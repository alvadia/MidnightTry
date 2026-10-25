from __future__ import annotations
"""Dataset preparation: KION, Yambda (subsample), synthetic.

Common output format (npz + json meta):
  seqs      int32 [total]     local item ids (1..V), concatenated per user,
                              chronological inside a user
  offsets   int32 [U+1]       per-user ranges
  rewards   float32 [total]   watched ratio in [0,1] (KION watched_pct/100,
                              Yambda min(played_ratio_pct,100)/100)
  item_freq int32 [V+1]       frequency over kept events, index 0 unused

Split protocol: per-user chronological leave-last-out — the last event is the
test target, the penultimate one is validation, everything earlier is
training. Because each user's events are ordered by time, no future event can
leak into any prefix.
"""
import csv, json, os
import numpy as np

from .config import Config


class UserData:
    """In-memory view over a processed npz dataset."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        d = np.load(cfg.processed_npz_path, allow_pickle=False)
        self.seqs = d["seqs"].astype(np.int64)
        self.offsets = d["offsets"].astype(np.int64)
        self.rewards = d["rewards"].astype(np.float32)
        self.item_freq = d["item_freq"].astype(np.int64)
        with open(cfg.processed_meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.vocab = int(self.meta["vocab"])          # ids are 1..vocab
        self.n_users = len(self.offsets) - 1

    def user_seq(self, u: int):
        a, b = self.offsets[u], self.offsets[u + 1]
        return self.seqs[a:b], self.rewards[a:b]

    def evaluable_users(self):
        return np.array([u for u in range(self.n_users)
                         if self.offsets[u + 1] - self.offsets[u] >= 3])

    # --- splits (per-user chronological leave-last-out) ---
    def train_batch_inputs(self, users: np.ndarray, max_len: int):
        """Teacher forcing on the train part s1..s_{n-2}: input s[:-1], target s[1:]."""
        seqs = [self.user_seq(u)[0][:-2] for u in users]
        seqs = [s for s in seqs if len(s) >= 2]
        return self._pack(seqs, max_len)

    def eval_tensors(self, users: np.ndarray, which: str, max_len: int):
        """which = 'valid' | 'test'. Input s1..s_{n-2|n-1}, one target item."""
        inputs, targets, seen = [], [], []
        for u in users:
            s = self.user_seq(u)[0]
            if which == "valid":
                inp, tgt = s[:-2], s[-2]
            else:
                inp, tgt = s[:-1], s[-1]
            inputs.append(inp[-max_len:])
            targets.append(tgt)
            seen.append(np.unique(s[:-1] if which == "test" else s[:-2]))
        packed = self._pack(inputs, max_len, with_targets=False)
        return packed[0], np.asarray(targets, dtype=np.int64), seen

    def rl_prompts(self, users: np.ndarray, max_len: int):
        """Context = full train history s1..s_{n-2}; the policy generates its next item."""
        seqs = [self.user_seq(u)[0][:-2] for u in users]
        seqs = [s for s in seqs if len(s) >= 1]
        return self._pack(seqs, max_len, with_targets=False)[0]

    def rm_tensors(self, users: np.ndarray, max_len: int):
        """RM training: position t predicts reward of s_{t+1} (train events only)."""
        seqs, rw = [], []
        for u in users:
            s, r = self.user_seq(u)
            s_tr, r_tr = s[:-2], r[:-2]           # targets inside train part
            if len(s_tr) < 2:
                continue
            seqs.append(s_tr)
            rw.append(r_tr)
        if not seqs:
            return (np.zeros((0, 1), dtype=np.int64),
                    np.zeros((0, 1), dtype=np.float32),
                    np.zeros((0, 1), dtype=np.float32))
        B = len(seqs)
        L = min(max_len, max(len(s) for s in seqs) - 1)
        inp = np.zeros((B, L), dtype=np.int64)
        lab = np.zeros((B, L), dtype=np.float32)
        m = np.zeros((B, L), dtype=np.float32)
        for i, (s, r) in enumerate(zip(seqs, rw)):
            src_full, dst_full = s[:-1], r[1:]
            k = min(len(src_full), L)
            inp[i, L - k:] = src_full[-k:]
            lab[i, L - k:] = dst_full[-k:]
            m[i, L - k:] = 1.0
        return inp, lab, m

    def _pack(self, seqs, max_len, with_targets=True):
        """Left-pad sequences to the batch max (capped at max_len)."""
        seqs = [s[-max_len:] for s in seqs]
        L = max(len(s) for s in seqs)
        B = len(seqs)
        inp = np.zeros((B, L), dtype=np.int64)
        tgt = np.zeros((B, L), dtype=np.int64)
        mask = np.zeros((B, L), dtype=np.float32)
        for i, s in enumerate(seqs):
            if with_targets:
                src, dst = (s[:-1], s[1:]) if len(s) > 1 else (np.array([], np.int64),) * 2
                k = len(dst)
                inp[i, L - k:] = src
                tgt[i, L - k:] = dst
                mask[i, L - k:] = 1.0
            else:
                inp[i, L - len(s):] = s
                mask[i, L - len(s):] = 1.0
        if with_targets:
            return inp, tgt, mask
        return inp, mask


def save_processed(cfg: Config, seqs, offsets, rewards, item_freq, meta):
    os.makedirs(cfg.processed_dir, exist_ok=True)
    np.savez_compressed(cfg.processed_npz_path,
                        seqs=seqs.astype(np.int32),
                        offsets=offsets.astype(np.int32),
                        rewards=rewards.astype(np.float32),
                        item_freq=item_freq.astype(np.int32))
    with open(cfg.processed_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[data] saved {cfg.processed_npz_path}: users={len(offsets)-1}, "
          f"events={len(seqs)}, vocab={meta['vocab']}")


# ---------------------------------------------------------------- KION
def prepare_kion(cfg: Config, raw_csv: str | None = None):
    """KION interactions.csv: user_id,item_id,last_watch_dt,total_dur,watched_pct.

    Chronological per-user ordering, iterative core filtering, id remapping.
    """
    raw_csv = raw_csv or os.path.join(cfg.data_dir, "kion", "interactions.csv")
    rows = []
    with open(raw_csv, encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        header = next(r)
        assert header[:2] == ["user_id", "item_id"], header
        for line in r:
            if len(line) < 5:
                continue
            try:
                u = int(line[0]); i = int(line[1]); dt = line[2]
                pct = float(line[4])
            except ValueError:
                continue
            if not dt or len(dt) != 10:      # truncated tail of the file
                continue
            rows.append((dt, u, i, min(max(pct, 0.0), 100.0) / 100.0))
    print(f"[kion] parsed {len(rows)} events")

    while True:
        uc, ic = {}, {}
        for _, u, i, _ in rows:
            uc[u] = uc.get(u, 0) + 1
            ic[i] = ic.get(i, 0) + 1
        kept = [(d, u, i, r) for d, u, i, r in rows
                if uc[u] >= cfg.min_user_events and ic[i] >= cfg.min_item_freq]
        if len(kept) == len(rows):
            break
        rows = kept
        if len(rows) < 1000:
            raise RuntimeError("filtering collapsed the dataset")

    rows.sort(key=lambda x: (x[1], x[0]))    # user, then date (day granularity)
    uniq = []
    for rec in rows:
        if uniq and uniq[-1][1] == rec[1] and uniq[-1][2] == rec[2]:
            continue                          # duplicate user+item: keep one
        uniq.append(rec)
    rows = uniq

    items = sorted({i for _, _, i, _ in rows})
    imap = {it: k + 1 for k, it in enumerate(items)}
    freq = np.zeros(len(items) + 1, dtype=np.int32)
    for _, _, i, _ in rows:
        freq[imap[i]] += 1

    seqs, offsets, rewards = [], [0], []
    cur_user = None
    for d, u, i, r in rows:
        if cur_user is None:
            cur_user = u
        if u != cur_user:
            offsets.append(len(seqs))
            cur_user = u
        seqs.append(imap[i])
        rewards.append(r)
    offsets.append(len(seqs))
    meta = dict(dataset="kion", vocab=len(items), n_events=len(seqs),
                min_user_events=cfg.min_user_events, min_item_freq=cfg.min_item_freq,
                date_range=[min(d for d, *_ in rows), max(d for d, *_ in rows)])
    save_processed(cfg, np.asarray(seqs), np.asarray(offsets),
                   np.asarray(rewards), freq, meta)


# ---------------------------------------------------------------- Yambda
def prepare_yambda(cfg: Config, n_users_target=30_000, min_events=10,
                   max_events=3_000_000, per_user_cap=0):
    """Yambda sequential/50m/listens.parquet — one row per user with parallel
    arrays (timestamp, item_id, played_ratio_pct, track_length_seconds).
    Streaming stops after n_users_target qualifying users / max_events rows.
    per_user_cap: keep only the LAST N events of each user (the head of the
    file is dominated by multi-year power users; capping spreads the budget
    over many more users).
    Requires: pip install datasets
    """
    from datasets import load_dataset  # noqa
    ds = load_dataset("yandex/yambda", data_files="sequential/50m/listens.parquet",
                      split="train", streaming=True)
    recs, users_taken = [], 0
    for row in ds:
        n = len(row["item_id"])
        if n < min_events:
            continue
        ts = row["timestamp"]
        items = row["item_id"]
        ratios = row["played_ratio_pct"]
        pairs = [(a, b, c) for a, b, c in zip(ts, items, ratios) if b is not None]
        pairs.sort(key=lambda x: x[0])          # ensure chronological
        if per_user_cap and len(pairs) > per_user_cap:
            pairs = pairs[-per_user_cap:]
        for t, i, r in pairs:
            recs.append((int(t), users_taken, int(i),
                         min(max(r or 0, 0), 100) / 100.0))
        users_taken += 1
        if users_taken >= n_users_target or len(recs) >= max_events:
            break
    print(f"[yambda] took {users_taken} users, {len(recs)} events")
    _save_from_records(cfg, recs, cfg.dataset, min_events, cfg.min_item_freq)


# ---------------------------------------------------------------- synthetic
def prepare_synthetic(cfg: Config, n_users=3000, vocab=500):
    """Latent-factor synthetic data with popularity + completion bias."""
    rng = np.random.default_rng(cfg.seed)
    user_pref = rng.normal(0, 1, (n_users, 32))
    item_lat = rng.normal(0, 1, (vocab, 32))
    item_short = rng.random(vocab)                       # fake duration proxy
    pop = rng.pareto(1.5, vocab) + 1
    pop = pop / pop.sum()
    seqs, rewards, offsets = [], [], [0]
    freq = np.zeros(vocab + 1, dtype=np.int32)
    for u in range(n_users):
        n = int(rng.poisson(12)) + 5
        scores = item_lat @ user_pref[u]
        p = np.exp((scores - scores.max()) / 2.0) * pop
        p /= p.sum()
        chosen = rng.choice(vocab, size=n, p=p, replace=False)
        for it in chosen:
            r = float(np.clip(0.5 + 0.03 * (item_lat[it] @ user_pref[u])
                              + 0.35 * (0.5 - item_short[it]) + rng.normal(0, .15), 0, 1))
            seqs.append(int(it) + 1)
            rewards.append(r)
            freq[int(it) + 1] += 1
        offsets.append(len(seqs))
    meta = dict(dataset="synthetic", vocab=vocab, n_events=len(seqs),
                min_user_events=0, min_item_freq=0, date_range=["synthetic", ""])
    save_processed(cfg, np.asarray(seqs), np.asarray(offsets),
                   np.asarray(rewards), freq, meta)


def _save_from_records(cfg, recs, name, min_events, min_item_freq):
    import collections
    by_user = collections.defaultdict(list)
    for ts, u, i, r in recs:
        by_user[u].append((ts, i, r))
    ic = collections.Counter()
    for v in by_user.values():
        for _, i, _ in v:
            ic[i] += 1
    while True:
        keep_i = {i for i, c in ic.items() if c >= min_item_freq}
        new_by_user = {}
        for u, v in by_user.items():
            fv = [(ts, i, r) for ts, i, r in v if i in keep_i]
            if len(fv) >= min_events:
                new_by_user[u] = fv
        new_ic = collections.Counter()
        for v in new_by_user.values():
            for _, i, _ in v:
                new_ic[i] += 1
        if len(new_by_user) == len(by_user):
            by_user = new_by_user
            break
        by_user, ic = new_by_user, new_ic

    items = sorted({i for v in by_user.values() for _, i, _ in v})
    imap = {it: k + 1 for k, it in enumerate(items)}
    freq = np.zeros(len(items) + 1, dtype=np.int32)
    seqs, rewards, offsets = [], [], [0]
    for u in sorted(by_user):
        v = sorted(by_user[u])
        for ts, i, r in v:
            seqs.append(imap[i]); rewards.append(r); freq[imap[i]] += 1
        offsets.append(len(seqs))
    meta = dict(dataset=name, vocab=len(items), n_events=len(seqs),
                min_user_events=min_events, min_item_freq=cfg.min_item_freq,
                date_range=["", ""])
    save_processed(cfg, np.asarray(seqs), np.asarray(offsets),
                   np.asarray(rewards), freq, meta)
