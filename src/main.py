from __future__ import annotations
"""CLI: python -m src.main --stage <base|rm|rmeval|rl|eval|all> \
        [--cfg f.json] [--set k=v] [--dataset d] [--run-name r] [--device cpu|cuda]"""
import argparse, json, os
import numpy as np
import torch

from .config import Config
from .data import UserData, prepare_kion, prepare_synthetic, prepare_yambda
from .model import PolicyModel, RewardModel
from . import metrics as M
from . import train as T


def load_cfg(args):
    cfg = Config.load(args.cfg) if args.cfg else Config()
    if args.run_name:
        cfg.run_name = args.run_name
    if args.dataset:
        cfg.dataset = args.dataset
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true")
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        setattr(cfg, k, v)
    return cfg


def get_data(cfg, force_prep=False):
    if not os.path.exists(cfg.processed_npz_path) or force_prep:
        if cfg.dataset == "kion":
            prepare_kion(cfg)
        elif cfg.dataset == "synthetic":
            prepare_synthetic(cfg)
        elif cfg.dataset == "yambda":
            prepare_yambda(cfg)
        else:
            raise ValueError(cfg.dataset)
    ud = UserData(cfg)
    cfg.vocab = ud.vocab
    return ud


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def eval_rm(cfg, data, rm, device):
    """Hold-out quality: last event per user (target unseen in the prefix).

    Also reports the correlation of the true label with item length proxies and
    a naive per-item-mean baseline for reference.
    """
    users = data.evaluable_users()
    rm.eval()
    global_mean = float(data.rewards.mean())
    im = np.zeros(data.vocab + 1, dtype=np.float64)
    c = np.zeros(data.vocab + 1, dtype=np.int64)
    for u in range(data.n_users):
        s, r = data.user_seq(u)
        s_tr, r_tr = s[:-2], r[:-2]
        np.add.at(im, s_tr, r_tr)
        np.add.at(c, s_tr, 1)
    im = np.where(c > 0, im / np.maximum(c, 1), global_mean)
    preds, trues, item_means = [], [], []
    for s0 in range(0, len(users), 256):
        ub = users[s0:s0 + 256]
        inp, tgt, _ = data.eval_tensors(ub, "test", cfg.max_len)
        ext = np.concatenate([inp, tgt.reshape(-1, 1)], 1)
        with torch.no_grad():
            p = rm.reward(torch.as_tensor(ext, device=device))[:, -1].cpu().numpy()
        preds.append(p)
        trues.append(np.array([data.user_seq(u)[1][-1] for u in ub]))
        item_means.append(im[tgt])
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    item_means = np.concatenate(item_means)
    return dict(n=len(preds),
                rmse=float(np.sqrt(((preds - trues) ** 2).mean())),
                spearman_pred=spearman(preds, trues),
                rmse_popbaseline=float(np.sqrt(((item_means - trues) ** 2).mean())),
                spearman_popbaseline=spearman(item_means, trues),
                mean_pred=float(preds.mean()), mean_true=float(trues.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--set", action="append")
    ap.add_argument("--run-name", dest="run_name", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prep", action="store_true", help="force re-preprocessing")
    args = ap.parse_args()
    cfg = load_cfg(args)
    os.makedirs(cfg.run_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.run_dir, "config.json"))
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    data = get_data(cfg, force_prep=args.prep)
    log = print
    torch.set_num_threads(os.cpu_count() or 4)
    log(f"[cfg] dataset={cfg.dataset} users={data.n_users} events={len(data.seqs)} "
        f"vocab={data.vocab} device={device}")

    if args.stage in ("prep", "all") and args.prep:
        pass  # done by get_data

    if args.stage in ("base", "all"):
        T.train_base(cfg, data, device, log)
    if args.stage in ("rm", "all"):
        T.train_rm(cfg, data, device, log)
    if args.stage in ("rmeval", "all"):
        rm = RewardModel(cfg).to(device)
        rm.load_state_dict(torch.load(cfg.rm_path, map_location=device))
        res = eval_rm(cfg, data, rm, device)
        log(f"[rmeval] {json.dumps(res)}")
        json.dump(res, open(os.path.join(cfg.run_dir, "rmeval.json"), "w"), indent=1)
    if args.stage in ("rl", "all"):
        T.rl_finetune(cfg, data, device, log)
    if args.stage in ("eval", "all"):
        results = {}
        pol = PolicyModel(cfg).to(device)
        pol.load_state_dict(torch.load(cfg.policy_path, map_location=device))
        rm = RewardModel(cfg).to(device)
        rm.load_state_dict(torch.load(cfg.rm_path, map_location=device))
        results["base"] = T.evaluate_policy(cfg, data, pol, "test", device,
                                            extra_rm_reward=True, rm=rm)
        pth = os.path.join(cfg.run_dir, "policy_rl.pt")
        if os.path.exists(pth):
            pol_rl = PolicyModel(cfg).to(device)
            pol_rl.load_state_dict(torch.load(pth, map_location=device))
            results["rl"] = T.evaluate_policy(cfg, data, pol_rl, "test", device,
                                              extra_rm_reward=True, rm=rm)
            results["gen_base"] = M.multi_step_generate(cfg, pol, data, rm, device)
            results["gen_rl"] = M.multi_step_generate(cfg, pol_rl, data, rm, device)
        json.dump(results, open(os.path.join(cfg.run_dir, "eval.json"), "w"), indent=1)
        log(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
