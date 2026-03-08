"""
RNN Encoder — LSTM-based temporal memory encoder.

Encodes sequences of episodic memories into temporal context vectors that
capture recency, repetition, and sequential patterns in conversation history.

Degrades gracefully when PyTorch is unavailable (returns heuristic fallbacks).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from cortana import config

VOCAB_SIZE = 8192
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
_MODEL_PATH = Path(config.AGENT_WORKSPACE) / "models" / "rnn_encoder.pt"


def _tokenize(text: str) -> List[int]:
    """Hash-trick word tokenizer — no vocab needed, deterministic."""
    tokens = []
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        tokens.append(h % VOCAB_SIZE)
    return tokens or [0]


if _TORCH_AVAILABLE:
    class _TemporalLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed   = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
            self.lstm    = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS,
                                   batch_first=True, dropout=0.1)
            self.project = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        def forward(self, token_ids: "torch.Tensor") -> "torch.Tensor":
            emb = self.embed(token_ids)           # (B, T, embed_dim)
            _, (h_n, _) = self.lstm(emb)          # h_n: (layers, B, hidden_dim)
            return self.project(h_n[-1])           # (B, hidden_dim)
else:
    _TemporalLSTM = None  # type: ignore


class RNNEncoder:
    """
    Wraps _TemporalLSTM for temporal episode encoding.
    All public methods return safe fallbacks if PyTorch is missing.
    """

    def __init__(self) -> None:
        self.available = _TORCH_AVAILABLE
        self._model: Optional["_TemporalLSTM"] = None
        if self.available:
            self._load_or_init()

    # ------------------------------------------------------------------
    # Init / persistence
    # ------------------------------------------------------------------

    def _load_or_init(self) -> None:
        import torch
        self._model = _TemporalLSTM()
        self._model.eval()
        if _MODEL_PATH.exists():
            try:
                state = torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=True)
                self._model.load_state_dict(state)
            except Exception:
                pass  # fresh random init is fine
        else:
            _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        if self._model is not None:
            import torch
            torch.save(self._model.state_dict(), str(_MODEL_PATH))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_episode(self, text: str) -> Optional[List[float]]:
        """Single episode → hidden vector (list[float] len=HIDDEN_DIM)."""
        if not self.available or self._model is None:
            return None
        import torch
        tokens = _tokenize(text)[:512]
        with torch.no_grad():
            t = torch.tensor([tokens], dtype=torch.long)
            return self._model(t)[0].tolist()

    def encode_sequence(self, episodes: List[str]) -> Optional[List[float]]:
        """
        Encode a time-ordered list of episodes as one temporal context vector.
        Concatenates the last 20 episodes (200 chars each) then runs LSTM.
        """
        if not self.available or self._model is None or not episodes:
            return None
        import torch
        combined = " [SEP] ".join(ep[:200] for ep in episodes[-20:])
        tokens = _tokenize(combined)[:1024]
        with torch.no_grad():
            t = torch.tensor([tokens], dtype=torch.long)
            return self._model(t)[0].tolist()

    def score_concepts(
        self,
        temporal_ctx: List[float],
        concept_topics: List[str],
    ) -> List[float]:
        """
        Cosine similarity between temporal context vector and each concept
        embedding. Returns scores in [0, 1] per concept.
        """
        if not self.available or self._model is None or not concept_topics:
            return [0.5] * len(concept_topics)
        import torch
        ctx = torch.tensor(temporal_ctx, dtype=torch.float32)
        ctx_norm = ctx / (ctx.norm() + 1e-8)
        scores: List[float] = []
        with torch.no_grad():
            for topic in concept_topics:
                tokens = _tokenize(topic)[:64]
                t = torch.tensor([tokens], dtype=torch.long)
                vec = self._model(t)[0]
                vec_norm = vec / (vec.norm() + 1e-8)
                sim = float((ctx_norm * vec_norm).sum())
                scores.append((sim + 1.0) / 2.0)   # [-1,1] → [0,1]
        return scores
