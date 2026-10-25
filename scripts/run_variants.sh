#!/usr/bin/env bash
# Queue RL-variants after the main run; reuses trained base policy + RM.
set -e
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
COMMON="--dataset kion --set processed_dir=data/processed_kion --set d_model=64 --set n_layer=2 --set n_head=4 --set d_ff=256 --set max_len=30 --set rl_steps=800 --set rl_batch=64 --set eval_every=200 --set rl_eval_users=3000 --set gen_users=300"

run_variant () {
  local name=$1; shift
  echo "=== variant $name ==="
  PYTHONUNBUFFERED=1 $PY -m src.main --stage rl  --run-name "$name" $COMMON "$@" > "runs/$name.log" 2>&1
  PYTHONUNBUFFERED=1 $PY -m src.main --stage eval --run-name "$name" $COMMON "$@" >> "runs/$name.log" 2>&1
}

run_variant kion_completion_nokl   --set rl_algo=reinforce --set rl_reward=completion --set kl_beta=0.0
run_variant kion_completion_norep  --set rl_algo=reinforce --set rl_reward=completion_norep --set kl_beta=0.05
run_variant kion_completion_poppen --set rl_algo=reinforce --set rl_reward=completion_pop_pen --set kl_beta=0.05
run_variant kion_grpo              --set rl_algo=grpo --set rl_reward=completion --set kl_beta=0.05
echo "ALL VARIANTS DONE"
