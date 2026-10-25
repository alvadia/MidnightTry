from __future__ import annotations
import glob, json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.config import Config          # noqa
from src.data import UserData          # noqa
from src.model import PolicyModel, RewardModel  # noqa

cfg = Config(dataset="kion", processed_dir="data/processed_kion",
             d_model=64, n_layer=2, n_head=4, d_ff=256, max_len=30, vocab=6459)
data = UserData(cfg)
st = np.load(os.path.join(ROOT, "data/processed_kion/item_stats.npz"))
mean_dur, mean_pct, pop = st["mean_dur"], st["mean_pct"], st["pop"]

users = data.evaluable_users()
rng = np.random.default_rng(0)
users = users[rng.choice(len(users), size=5000, replace=False)]
inp, tgt, seen = data.eval_tensors(users, "test", cfg.max_len)

def top10(pol):
    outs = []
    for s in range(0, len(users), 250):
        x = torch.as_tensor(inp[s:s+250])
        with torch.no_grad():
            lg = pol.logits(x)[:, -1].clone()
        lg[:, 0] = float("-inf")
        _, ids = lg.topk(10, dim=-1)
        outs.append((ids + 1).numpy())
    return np.concatenate(outs)  # [U,10]

def summarize(name, tops, base_tops=None):
    d = dict(model=name,
             mean_item_dur=float(mean_dur[tops].mean()),
             mean_item_pct=float(mean_pct[tops].mean()),
             mean_log_pop=float(np.log(pop[tops] + 1).mean()),
             uniq_items=int(len(np.unique(tops))),
             gini_like=float(np.sort(np.bincount(tops.reshape(-1), minlength=cfg.vocab + 1)[1:])[::-1][:100].sum() / tops.size))
    if base_tops is not None:
        same = (tops[:, None, :] == base_tops[None, :, :]).any(2).mean(1)
        d["overlap_with_base@10"] = float(same.mean())
        changed = []
        for i in range(len(tops)):
            m = ~(tops[i][:, None] == base_tops[i][None, :]).any(1)
            changed += list(tops[i][m])
        if changed:
            c = np.array(changed)
            d["new_items_mean_dur"] = float(mean_dur[c].mean())
            d["new_items_mean_pct"] = float(mean_pct[c].mean())
            d["new_items_log_pop"] = float(np.log(pop[c] + 1).mean())
    return d

results = []
pol = PolicyModel(cfg); pol.load_state_dict(torch.load(cfg.policy_path, map_location="cpu")); pol.eval()
base_tops = top10(pol)
results.append(summarize("base", base_tops))
for pth in sorted(glob.glob(os.path.join(ROOT, "runs/kion_*/policy_rl.pt"))):
    name = os.path.basename(os.path.dirname(pth))
    p = PolicyModel(cfg); p.load_state_dict(torch.load(pth, map_location="cpu")); p.eval()
    t = top10(p)
    results.append(summarize(name, t, base_tops))

rm = RewardModel(cfg); rm.load_state_dict(torch.load(cfg.rm_path, map_location="cpu")); rm.eval()
ext = np.concatenate([inp, tgt.reshape(-1, 1)], 1)
preds, trues = [], []
for s in range(0, len(users), 250):
    with torch.no_grad():
        preds.append(rm.reward(torch.as_tensor(ext[s:s+250]))[:, -1].numpy())
    trues.append(np.array([data.user_seq(u)[1][-1] for u in users[s:s+250]]))
preds = np.concatenate(preds); trues = np.concatenate(trues)
dur_items = mean_dur[tgt]
qs = np.quantile(dur_items, [0, .25, .5, .75, 1.0])
calib = []
for a, b in zip(qs[:-1], qs[1:]):
    m = (dur_items >= a) & (dur_items <= b)
    calib.append(dict(dur_range=[float(a), float(b)], n=int(m.sum()),
                      pred=float(preds[m].mean()), true=float(trues[m].mean())))

out = dict(models=results, rm_calibration_by_duration=calib,
           corr_pred_true=float(np.corrcoef(preds, trues)[0, 1]))
os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "report/bias_analysis.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
