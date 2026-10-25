from __future__ import annotations
"""GPT-2 style sequential recommender + reward model.

Tokens are item ids (1..V); 0 is left-padding. Both models share the trunk;
the policy has a tied item-embedding output head over columns 0..V-1
(column j = item id j+1), the reward model has a scalar head predicting the
watched ratio of the *next* item after each prefix position.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class CausalBlock(nn.Module):
    """Pre-LN transformer block with hand-rolled causal MHA.

    Hand-rolled because causal mask + left-padding makes fully masked query
    rows (all-(-inf) softmax -> NaN). The diagonal is always allowed; padded
    positions are never read downstream.
    """

    def __init__(self, d: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.dk = d // n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x, key_pad):                       # key_pad [B,L] bool True=pad
        B, L, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, self.dk).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.dk).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        allow = causal_allowed(L, x.device) & (~key_pad)[:, None, None, :]
        eye = torch.eye(L, dtype=torch.bool, device=x.device)
        allow = allow | eye[None]
        att = att.masked_fill(~allow, float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        o = (att @ v).transpose(1, 2).reshape(B, L, d)
        x = x + self.proj(o)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


_ALLOWED_CACHE = {}


def causal_allowed(L: int, device) -> torch.Tensor:
    key = (L, device)
    if key not in _ALLOWED_CACHE:
        m = torch.ones((L, L), dtype=torch.bool, device=device).tril()
        _ALLOWED_CACHE[key] = m
    return _ALLOWED_CACHE[key]


class TransformerTrunk(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.vocab_padded, cfg.d_model,
                                     padding_idx=cfg.pad_token_id)
        self.pos_emb = nn.Embedding(cfg.max_len + 1, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            CausalBlock(cfg.d_model, cfg.n_head, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, input_ids: torch.Tensor):
        B, L = input_ids.shape
        pad_mask = input_ids == self.cfg.pad_token_id           # [B,L] True=pad
        pos = (input_ids != self.cfg.pad_token_id).long().cumsum(-1) - 1
        pos = pos.clamp(min=0)
        x = self.item_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.ln_f(x)                                     # [B,L,d]


class PolicyModel(nn.Module):
    """Next-item LM; logits over columns 0..V-1 = item ids 1..V."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.trunk = TransformerTrunk(cfg)
        self.ln_tied = nn.LayerNorm(cfg.d_model)                # GPT-2 pre-head LN

    def logits(self, input_ids):
        h = self.ln_tied(self.trunk(input_ids))
        return h @ self.trunk.item_emb.weight[1:].T             # [B,L,V]

    def logprobs_last(self, input_ids):
        return F.log_softmax(self.logits(input_ids)[:, -1], dim=-1)   # [B,V]

    @torch.no_grad()
    def topk_last(self, input_ids, k):
        return self.logits(input_ids)[:, -1].topk(k, dim=-1)


class RewardModel(nn.Module):
    """Predicts watched ratio of the next item given the prefix."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.trunk = TransformerTrunk(cfg)
        self.head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model),
                                  nn.GELU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.d_model, 1))

    def reward_logits(self, input_ids):
        h = self.trunk(input_ids)
        return self.head(h).squeeze(-1)                         # [B,L] raw

    def reward(self, input_ids):
        return torch.sigmoid(self.reward_logits(input_ids))

    @torch.no_grad()
    def score_pairs(self, history_ids, cand_ids):
        """history_ids: [B,L] left-padded prefix; cand_ids: [B] item id (1..V)."""
        ext = torch.cat([history_ids, cand_ids.unsqueeze(1)], dim=1)
        return self.reward(ext)[:, -1]
