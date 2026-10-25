from __future__ import annotations
import glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "report", "tables.md")

KEYS = [("recall@10", "Recall@10"), ("ndcg@10", "NDCG@10"), ("map@10", "MAP@10"),
        ("rm_reward_top1", "RM-reward top1"), ("rm_reward_top1_unmasked", "RM top1 (без маски)"),
        ("rewatch@k", "Rewatch@10"), ("coverage@k", "Coverage@10"),
        ("entropy_norm", "Entropy(top10)"), ("top1_share", "Top-1 share")]

COLS = ["kion_completion", "kion_completion_nokl", "kion_completion_norep",
        "kion_completion_poppen", "kion_grpo"]


def load(p):
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return None


def fmt(x, w=4):
    return "-" if x is None else f"{x:.{w}f}"


def main():
    lines = []
    # 1) reward model
    rm = None
    for c in COLS:
        rm = load(os.path.join(RUNS, c, "rmeval.json")) or rm
    if rm:
        lines += ["### Качество reward model (отложенная выборка = последний событие пользователя)", "",
                  "| метрика | значение |", "|---|---|",
                  f"| RMSE | {fmt(rm['rmse'])} |",
                  f"| Spearman (RM vs true) | {fmt(rm['spearman_pred'])} |",
                  f"| RMSE baseline (среднее по item) | {fmt(rm['rmse_popbaseline'])} |",
                  f"| Spearman baseline | {fmt(rm['spearman_popbaseline'])} |",
                  f"| n | {rm['n']} |", ""]
    # 2) main table: base vs RL variants
    base = None
    rows = []
    for c in COLS:
        ev = load(os.path.join(RUNS, c, "eval.json"))
        if not ev:
            continue
        if base is None and "base" in ev:
            base = ev["base"]
        rows.append((c, ev.get("rl")))
    hdr = "| модель | " + " | ".join(l for _, l in KEYS) + " |"
    sep = "|" + "---|" * (len(KEYS) + 1)
    lines += ["### Recall/NDCG/MAP и поведение top-10", "", hdr, sep]
    if base:
        lines.append("| base (без RL) | " + " | ".join(fmt(base.get(k)) for k, _ in KEYS) + " |")
    for name, r in rows:
        if r:
            lines.append(f"| {name} | " + " | ".join(fmt(r.get(k)) for k, _ in KEYS) + " |")
    lines.append("")
    # 3) multi-step generation
    lines += ["### Многошаговая генерация (5 шагов, контекст = train-история)", "",
              "| модель | шаг | RM reward | повтор в контексте | logprob политики |", "|---|---|---|---|---|"]
    for name, r in rows:
        ev = load(os.path.join(RUNS, name, "eval.json")) or {}
        for tag in ["gen_base", "gen_rl"]:
            g = ev.get(tag)
            if not g:
                continue
            label = "base" if tag == "gen_base" else name
            for st, d in sorted(g.items()):
                lines.append(f"| {label} | {st} | {fmt(d['rm_reward'])} | {fmt(d['repeat_rate'])} | {fmt(d['policy_logprob'], 3)} |")
    # 4) RL training curves (last entries)
    lines += ["", "### Динамика RL (по логам rl_log.json)", "",
              "| run | step | reward | kl | entropy | reward_top1_eval |", "|---|---|---|---|---|---|"]
    for name in COLS:
        lg = load(os.path.join(RUNS, name, "rl_log.json"))
        if not lg:
            continue
        picks = [lg[0], lg[len(lg)//2], lg[-1]]
        for e in picks:
            lines.append(f"| {name} | {e.get('step')} | {fmt(e.get('reward'))} | {fmt(e.get('kl'))} "
                         f"| {fmt(e.get('entropy'),3)} | {fmt(e.get('rm_reward_top1'))} |")
    txt = "\n".join(lines) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
