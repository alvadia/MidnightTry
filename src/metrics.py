from __future__ import annotations
"""Ranking metrics, diversity, RM-reward evaluation and multi-step generation."""
import numpy as np
import torch


def _to_dev(t, device):
    return torch.as_tensor(t, device=device)


@torch.no_grad()
def rank_metrics(cfg, policy, data, inp, tgt, seen_sets, device="cpu", topk=10,
                 extra_rm_reward=False, rm=None, chunk=256):
    """Full-catalogue ranking.

    Main metrics mask the user's history (standard offline protocol). In
    addition we rank WITHOUT masking to measure re-watch recommendations and
    the RM reward of the head of the ranking actually shown to the user.
    """
    policy.eval()
    B = inp.shape[0]
    V = data.vocab
    hit = np.zeros(B); rank_of = np.full(B, 10 ** 6)
    top_ids = np.zeros((B, topk), dtype=np.int64)
    unmask_top = np.zeros((B, topk), dtype=np.int64)
    rm_rewards, rm_rewards_unmasked, rewatch = [], [], []
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        x = _to_dev(inp[s:e], device)
        logits = policy.logits(x)[:, -1]                          # [b,V]
        logits[:, 0] = float("-inf")                              # dead pad column
        seen_np = [np.asarray(sorted(seen_sets[s + i])) for i in range(e - s)]
        b = e - s
        # --- unmasked ranking first ---
        sc, ids = logits.topk(topk, dim=-1)
        um = ids + 1
        unmask_top[s:e] = um.cpu().numpy()
        rep = np.zeros(b)
        for i in range(b):
            sn = set(seen_np[i].tolist())
            rep[i] = np.mean([int(it) in sn for it in um[i].tolist()])
        rewatch.append(rep)
        if extra_rm_reward and rm is not None:
            ext = torch.cat([x, um[:, :1]], 1)
            rm_rewards_unmasked.append(rm.reward(ext)[:, -1].cpu().numpy())
        # --- masked ranking (main metrics) ---
        for i, sn in enumerate(seen_np):
            if len(sn):
                logits[i, sn - 1] = float("-inf")
        sc, ids = logits.topk(topk, dim=-1)
        ids = ids + 1
        top_ids[s:e] = ids.cpu().numpy()
        tgt_t = _to_dev(tgt[s:e], device).unsqueeze(1)
        pos = (ids == tgt_t).float()
        hit[s:e] = pos.any(1).cpu().numpy()
        r = (pos.cumsum(1) > 0).float()
        first = r.argmax(1) + 1
        rank_of[s:e] = torch.where(pos.any(1), first,
                                   torch.full_like(first, 10 ** 6)).cpu().numpy()
        if extra_rm_reward and rm is not None:
            ext = torch.cat([x, ids[:, :1]], 1)
            rm_rewards.append(rm.reward(ext)[:, -1].cpu().numpy())
    D = 1.0 / np.log2(np.arange(2, topk + 2))
    recall = hit.mean()
    ndcg = np.where(rank_of <= topk, D[np.clip(rank_of - 1, 0, topk - 1)], 0.0).mean()
    mapk = np.where(rank_of <= topk, 1.0 / np.clip(rank_of, 1, topk), 0.0).mean()
    out = {f"recall@{topk}": float(recall), f"ndcg@{topk}": float(ndcg),
           f"map@{topk}": float(mapk), "recall": float(recall),
           "ndcg": float(ndcg), "map": float(mapk)}
    # diversity of recommended top-K (masked ranking)
    flat = top_ids.reshape(-1)
    out["coverage@k"] = float(len(np.unique(flat)) / V)
    counts = np.bincount(flat, minlength=V + 1)[1:]
    p = counts / counts.sum()
    out["entropy_norm"] = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(V))
    out["top1_share"] = float(counts.max() / counts.sum())
    # unmasked list stats (what a naive deployment would show)
    flat_u = unmask_top.reshape(-1)
    counts_u = np.bincount(flat_u, minlength=V + 1)[1:]
    p_u = counts_u / counts_u.sum()
    out["unmask_entropy_norm"] = float(-(p_u[p_u > 0] * np.log(p_u[p_u > 0])).sum() / np.log(V))
    out["unmask_top1_share"] = float(counts_u.max() / counts_u.sum())
    out["rewatch@k"] = float(np.concatenate(rewatch).mean())
    if extra_rm_reward and rm is not None:
        rm.eval()
        out["rm_reward_top1"] = float(np.concatenate(rm_rewards).mean())
        out["rm_reward_top1_unmasked"] = float(np.concatenate(rm_rewards_unmasked).mean())
    return out


@torch.no_grad()
def multi_step_generate(cfg, policy, data, rm, device="cpu", users=None,
                        steps=None, temperature=1.0, top_k=50):
    """Generate `steps` items, appending each to the context.

    Tracks per step: mean RM reward (on the generated, increasingly synthetic
    context), re-watch rate, and mean log-prob the policy assigns to its own
    generated item — a drift signal (falling prob = the context leaves the
    manifold of real histories).
    """
    steps = steps or cfg.gen_steps
    policy.eval(); rm.eval()
    if users is None:
        users = data.evaluable_users()[:cfg.gen_users]
    per_step = [dict(rm_reward=[], repeat_rate=[], logprob=[]) for _ in range(steps)]
    for u in users:
        s = data.user_seq(u)[0][:-2][-cfg.max_len:]
        if len(s) < 3:
            continue
        ctx = torch.as_tensor(np.asarray([s], dtype=np.int64), device=device)
        for t in range(steps):
            logits = policy.logits(ctx)[:, -1]
            logits[:, 0] = -1e9
            if top_k:
                v, _ = logits.topk(min(top_k, logits.shape[-1] - 1))
                thr = v[:, -1:].expand_as(logits)
                logits = torch.where(logits < thr, torch.full_like(logits, -1e9), logits)
            logp_all = torch.log_softmax(logits, -1)
            probs = (logits / temperature).softmax(-1)
            idx = torch.multinomial(probs, 1)                    # [1,1] column
            item = idx + 1                                       # item id
            per_step[t]["logprob"].append(logp_all.gather(1, idx).item())
            ext = torch.cat([ctx, item], 1)[:, -cfg.max_len:]
            per_step[t]["rm_reward"].append(float(rm.reward(ext)[0, -1]))
            per_step[t]["repeat_rate"].append(
                int(item.item() in set(ctx.flatten().tolist())))
            ctx = ext
    out = {}
    for t, d in enumerate(per_step):
        if not d["rm_reward"]:
            continue
        out[f"step{t+1}"] = dict(rm_reward=float(np.mean(d["rm_reward"])),
                                 repeat_rate=float(np.mean(d["repeat_rate"])),
                                 policy_logprob=float(np.mean(d["logprob"])))
    return out
