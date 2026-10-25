from __future__ import annotations
"""Stage runners: pretrain the policy (next-item), train the reward model,
RL fine-tune the policy against the RM, and evaluate."""
import copy, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import UserData
from .model import PolicyModel, RewardModel
from . import metrics as M


def _to_dev(t, device):
    return torch.as_tensor(t, device=device)


def _batches(arr, bs, rng, shuffle=True):
    idx = rng.permutation(len(arr)) if shuffle else np.arange(len(arr))
    for i in range(0, len(idx), bs):
        yield arr[idx[i:i + bs]]


# --------------------------------------------------------------------- base
def train_base(cfg: Config, data: UserData, device="cpu", log=print):
    torch.manual_seed(cfg.seed)
    model = PolicyModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=0.01)
    users = np.arange(data.n_users)
    rng = np.random.default_rng(cfg.seed)
    hist = []
    best = None
    for ep in range(cfg.base_epochs):
        model.train()
        tot, nb = 0.0, 0
        for ub in _batches(users, cfg.base_batch, rng):
            packed = data.train_batch_inputs(ub, cfg.max_len)
            if packed[0].shape[1] == 0:
                continue
            inp, tgt, mask = map(lambda x: _to_dev(x, device), packed)
            logits = model.logits(inp)                           # [B,L,V]
            tgt_idx = (tgt - 1).clamp(min=0)                     # ids 1..V -> cols 0..V-1
            loss = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     tgt_idx.reshape(-1), reduction="none")
                    * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
            if cfg.base_max_batches and nb >= cfg.base_max_batches:
                break
        v = evaluate_policy(cfg, data, model, "valid", device,
                            users=_subset(users, 3000, rng))
        log(f"[base] epoch {ep+1}/{cfg.base_epochs} loss={tot/max(nb,1):.4f} valid={v}")
        hist.append(tot / max(nb, 1))
        if best is None or v["ndcg"] > best[0]:
            best = (v["ndcg"], {k: t.cpu() for k, t in model.state_dict().items()})
        model.train()
    os.makedirs(cfg.processed_dir, exist_ok=True)
    torch.save(best[1], cfg.policy_path)
    os.makedirs(cfg.run_dir, exist_ok=True)
    json.dump({"loss_hist": hist}, open(os.path.join(cfg.run_dir, "base.json"), "w"))
    return model


def _subset(arr, n, rng):
    if len(arr) <= n:
        return arr
    return arr[rng.choice(len(arr), size=n, replace=False)]


# ------------------------------------------------------------------ reward
def train_rm(cfg: Config, data: UserData, device="cpu", log=print):
    torch.manual_seed(cfg.seed + 1)
    rm = RewardModel(cfg).to(device)
    opt = torch.optim.AdamW(rm.parameters(), lr=cfg.rm_lr, weight_decay=0.01)
    users = np.arange(data.n_users)
    rng = np.random.default_rng(cfg.seed + 1)
    for ep in range(cfg.rm_epochs):
        rm.train()
        tot, nb = 0.0, 0
        for ub in _batches(users, cfg.rm_batch, rng):
            inp, lab, m = data.rm_tensors(ub, cfg.max_len)
            if inp.shape[0] == 0 or inp.shape[1] == 0:
                continue
            inp, lab, m = map(lambda x: _to_dev(x, device), (inp, lab, m))
            keep = m.sum(1) > 0
            if not keep.any():
                continue
            inp, lab, m = inp[keep], lab[keep], m[keep]
            pred = rm.reward(inp)                                # [B,L]
            loss = ((pred - lab) ** 2 * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
            if cfg.rm_max_batches and nb >= cfg.rm_max_batches:
                break
        log(f"[rm] epoch {ep+1}/{cfg.rm_epochs} mse={tot/max(nb,1):.5f}")
    torch.save(rm.state_dict(), cfg.rm_path)
    return rm


# ---------------------------------------------------------------------- RL
def rl_finetune(cfg: Config, data: UserData, device="cpu", log=print):
    """REINFORCE / GRPO against the frozen RM with KL anchor to the base policy."""
    policy = PolicyModel(cfg).to(device)
    policy.load_state_dict(torch.load(cfg.policy_path, map_location="cpu"))
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    rm = RewardModel(cfg).to(device)
    rm.load_state_dict(torch.load(cfg.rm_path, map_location="cpu"))
    rm.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.rl_lr)
    rng = np.random.default_rng(cfg.seed + 7)
    users_all = np.arange(data.n_users)

    def raw_reward(hist_ids, cand):
        """RM score + shaping variants. hist_ids [B,L] padded, cand [B] ids 1..V."""
        r = rm.score_pairs(hist_ids, cand)                       # [B] in (0,1)
        if cfg.rl_reward == "completion_norep":
            r = r - cfg.rep_penalty * _in_history(hist_ids, cand, device)
        elif cfg.rl_reward == "completion_pop_pen":
            freq = _to_dev(data.item_freq, device)[cand].float().clamp(min=1)
            r = r - cfg.pop_pen_alpha * (freq.log() / math.log(data.item_freq.sum() + 1))
        return r

    log_lines, step, t0 = [], 0, time.time()
    for ub in _batches(users_all, cfg.rl_batch, rng):
        if step >= cfg.rl_steps:
            break
        policy.train()
        inp_np = data.rl_prompts(ub, cfg.max_len)
        if inp_np.shape[1] == 0:
            continue
        inp = _to_dev(inp_np, device)
        lp = policy.logprobs_last(inp)                           # [B,V]
        if cfg.rl_algo == "grpo":
            G = cfg.grpo_group
            idxs = torch.cat([torch.multinomial(lp.exp(), 1) for _ in range(G)], 1)  # [B,G] 0-based
            cand = idxs + 1
            reward = torch.stack([raw_reward(inp, cand[:, g]) for g in range(G)], 1)
            adv = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
            sel_lp = lp.gather(1, idxs)
            with torch.no_grad():
                ref_lp = ref.logprobs_last(inp).gather(1, idxs)
            kl = (sel_lp - ref_lp).mean()
            loss = -(adv.detach() * sel_lp).mean() + cfg.kl_beta * kl
            r_mean = reward.mean().item()
        else:                                                    # REINFORCE
            idx = torch.multinomial(lp.exp(), 1).squeeze(1)      # [B] 0-based
            cand = idx + 1
            reward = raw_reward(inp, cand)
            with torch.no_grad():
                base_lp = ref.logprobs_last(inp).gather(1, idx.unsqueeze(1)).squeeze(1)
            adv = reward - reward.mean().detach()
            sel_lp = lp.gather(1, idx.unsqueeze(1)).squeeze(1)
            kl = (sel_lp - base_lp).mean()
            loss = -(adv.detach() * sel_lp).mean() + cfg.kl_beta * kl
            r_mean = reward.mean().item()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0 or step == cfg.rl_steps:
            with torch.no_grad():
                ent = float(-(lp.exp() * lp).sum(1).mean())
            entry = dict(step=step, reward=r_mean, kl=float(kl.detach()),
                         entropy=ent, loss=float(loss.detach()))
            log_lines.append(entry)
            log(f"[rl] step {step} reward={r_mean:.4f} kl={entry['kl']:.4f} "
                f"ent={ent:.3f} loss={float(loss.detach()):.4f} ({time.time()-t0:.0f}s)")
        if cfg.eval_every and step % cfg.eval_every == 0:
            ev = evaluate_policy(cfg, data, policy, "test", device,
                                 users=_subset(np.arange(data.n_users),
                                               cfg.rl_eval_users, rng),
                                 extra_rm_reward=True, rm=rm)
            log(f"[rl] eval step {step}: {ev}")
            if log_lines and log_lines[-1].get("step") == step:
                log_lines[-1].update(ev)
            else:
                log_lines.append(dict(step=step, **ev))
    os.makedirs(cfg.run_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(cfg.run_dir, "policy_rl.pt"))
    json.dump(log_lines, open(os.path.join(cfg.run_dir, "rl_log.json"), "w"), indent=1)
    return policy


def _in_history(hist_ids, cand, device):
    """1.0 where cand id appears in the visible (non-pad) history."""
    eq = hist_ids == cand.unsqueeze(1)
    real = hist_ids != 0
    return (eq & real).any(1).float()


# ------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate_policy(cfg, data, policy, which, device="cpu", users=None,
                    extra_rm_reward=False, rm=None):
    policy.eval()
    if users is None:
        users = data.evaluable_users()
    inp, tgt, seen_sets = data.eval_tensors(users, which, cfg.max_len)
    out = M.rank_metrics(cfg, policy, data, inp, tgt, seen_sets, device,
                         topk=cfg.topk, extra_rm_reward=extra_rm_reward, rm=rm)
    policy.train()
    return out
