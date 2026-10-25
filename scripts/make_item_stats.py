from __future__ import annotations
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config  # noqa

RAW = "data/kion/interactions.csv"
cfg = Config(dataset="kion", processed_dir="data/processed_kion")

rows = []
with open(RAW, encoding="utf-8", errors="replace") as f:
    r = csv.reader(f); next(r)
    for line in r:
        if len(line) < 5:
            continue
        try:
            u = int(line[0]); i = int(line[1]); dt = line[2]; dur = float(line[3]); pct = float(line[4])
        except ValueError:
            continue
        if not dt or len(dt) != 10:
            continue
        rows.append((dt, u, i, dur, pct))

while True:
    uc, ic = {}, {}
    for _, u, i, _, _ in rows:
        uc[u] = uc.get(u, 0) + 1
        ic[i] = ic.get(i, 0) + 1
    kept = [x for x in rows if uc[x[1]] >= cfg.min_user_events and ic[x[2]] >= cfg.min_item_freq]
    if len(kept) == len(rows):
        break
    rows = kept

rows.sort(key=lambda x: (x[1], x[0]))
uniq = []
for rec in rows:
    if uniq and uniq[-1][1] == rec[1] and uniq[-1][2] == rec[2]:
        continue
    uniq.append(rec)
items = sorted({i for _, _, i, _, _ in uniq})
imap = {it: k + 1 for k, it in enumerate(items)}
assert len(items) == json.load(open(cfg.processed_meta_path, encoding="utf-8"))["vocab"], \
    "mapping mismatch vs kion.npz"

dur_sum = np.zeros(len(items) + 1); dur_n = np.zeros(len(items) + 1)
pct_sum = np.zeros(len(items) + 1); pop = np.zeros(len(items) + 1, dtype=np.int64)
for _, u, i, d, p in uniq:
    k = imap[i]
    dur_sum[k] += d; dur_n[k] += 1; pct_sum[k] += min(max(p, 0), 100) / 100.0; pop[k] += 1
mean_dur = np.where(dur_n > 0, dur_sum / np.maximum(dur_n, 1), 0)
mean_pct = np.where(dur_n > 0, pct_sum / np.maximum(dur_n, 1), 0)
orig = np.zeros(len(items) + 1, dtype=np.int64)
for it, k in imap.items():
    orig[k] = it
np.savez_compressed(os.path.join(cfg.processed_dir, "item_stats.npz"),
                    mean_dur=mean_dur.astype(np.float32),
                    mean_pct=mean_pct.astype(np.float32),
                    pop=pop, orig_id=orig)
print("saved item_stats.npz; items:", len(items))
