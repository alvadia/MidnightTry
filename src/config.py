from __future__ import annotations
"""Experiment configuration for RL-finetuned sequential recommender."""
import json, os
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- data ---
    dataset: str = "kion"                # kion | yambda | synthetic
    data_dir: str = "data"
    processed_dir: str = "data/processed"
    min_user_events: int = 5
    min_item_freq: int = 5
    max_len: int = 50                    # transformer context
    pad_token_id: int = 0                # 0 = pad, items are 1..vocab

    # --- base model (GPT-2 style) ---
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 512
    dropout: float = 0.2
    base_lr: float = 3e-4
    base_epochs: int = 5
    base_batch: int = 256
    base_max_batches: int = 0            # 0 = unlimited (for quick runs)
    seed: int = 42

    # --- reward model ---
    rm_epochs: int = 5
    rm_lr: float = 3e-4
    rm_batch: int = 256
    rm_max_batches: int = 0

    # --- RL ---
    rl_algo: str = "reinforce"           # reinforce | grpo
    rl_reward: str = "completion"        # completion | completion_norep | completion_pop_pen
    rl_steps: int = 2000
    rl_batch: int = 64
    rl_lr: float = 1e-5
    kl_beta: float = 0.05
    grpo_group: int = 4
    rl_temp: float = 1.0
    rep_penalty: float = 1.0             # penalty weight for already-watched items
    pop_pen_alpha: float = 0.5           # popularity penalty strength (variant 3)
    eval_every: int = 250
    rl_eval_users: int = 3000            # subsample of test users for in-loop eval

    # --- eval ---
    topk: int = 10
    gen_steps: int = 5
    gen_users: int = 500

    # --- paths ---
    out_dir: str = "runs"
    run_name: str = "default"
    vocab: int = 0                       # filled from meta after prep

    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_dir, self.run_name)

    @property
    def vocab_padded(self) -> int:
        return self.vocab + 1

    @property
    def processed_npz_path(self) -> str:
        return os.path.join(self.processed_dir, f"{self.dataset}.npz")

    @property
    def processed_meta_path(self) -> str:
        return os.path.join(self.processed_dir, f"{self.dataset}.json")

    @property
    def policy_path(self) -> str:
        return os.path.join(self.processed_dir, f"{self.dataset}_policy.pt")

    @property
    def rm_path(self) -> str:
        return os.path.join(self.processed_dir, f"{self.dataset}_rm.pt")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return Config(**json.load(f))
