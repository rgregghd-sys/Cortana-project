"""
GNN Reasoner — Graph Attention Network for concept relationship reasoning.

Operates on the Tier 3 logic matrix (concept nodes + relationship edges).
Message passing propagates query relevance through the graph, surfacing
the most contextually important concepts for the current conversation.

Pure PyTorch implementation — no PyG or DGL dependency.
Degrades gracefully when PyTorch is unavailable.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from cortana import config
from cortana.models.schemas import ConceptNode, RelationEdge

NODE_DIM   = 64
HIDDEN_DIM = 128
NUM_HEADS  = 4
EDGE_DIM   = 8
NUM_LAYERS = 2

RELATION_TYPES = [
    "includes", "depends_on", "contradicts", "leads_to",
    "part_of", "related_to", "enables", "other",
]

_MODEL_PATH = Path(config.AGENT_WORKSPACE) / "models" / "gnn_reasoner.pt"


# ------------------------------------------------------------------
# Text → fixed-size bag-of-words vector (no neural net needed for feats)
# ------------------------------------------------------------------

def _bow_embed(text: str, dim: int = NODE_DIM) -> List[float]:
    vec = [0.0] * dim
    words = text.lower().split()
    if not words:
        return vec
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 / len(words)
    return vec


# ------------------------------------------------------------------
# GAT layer (pure PyTorch)
# ------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class _GATLayer(nn.Module):
        """Single Graph Attention layer with multi-head attention + edge bias."""

        def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int):
            super().__init__()
            assert out_dim % heads == 0
            self.heads    = heads
            self.head_dim = out_dim // heads
            self.W_node   = nn.Linear(in_dim, out_dim, bias=False)
            self.W_edge   = nn.Linear(edge_dim, heads, bias=False)
            self.attn     = nn.Linear(2 * self.head_dim, 1, bias=False)
            self.leaky    = nn.LeakyReLU(0.2)
            self.out_proj = nn.Linear(out_dim, out_dim)
            self.norm     = nn.LayerNorm(out_dim)
            self._res     = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        def forward(
            self,
            h: "torch.Tensor",          # (N, in_dim)
            adj: "torch.Tensor",        # (N, N)
            edge_feats: Optional["torch.Tensor"],  # (N, N, edge_dim)
        ) -> "torch.Tensor":
            N = h.size(0)
            res = self._res(h)
            H = self.W_node(h).view(N, self.heads, self.head_dim)     # (N, heads, hd)
            Hi = H.unsqueeze(1).expand(N, N, self.heads, self.head_dim)
            Hj = H.unsqueeze(0).expand(N, N, self.heads, self.head_dim)
            pair = torch.cat([Hi, Hj], dim=-1)                        # (N, N, heads, 2*hd)
            # attention logits: (N, N, heads) → (heads, N, N)
            al = self.leaky(self.attn(pair).squeeze(-1)).permute(2, 0, 1)
            if edge_feats is not None:
                al = al + self.W_edge(edge_feats).permute(2, 0, 1)
            mask = (adj == 0).unsqueeze(0).expand(self.heads, N, N)
            al = al.masked_fill(mask, -1e9)
            w = F.softmax(al, dim=-1)                                  # (heads, N, N)
            Hp = H.permute(1, 0, 2)                                    # (heads, N, hd)
            out = torch.bmm(w, Hp).permute(1, 0, 2).contiguous().view(N, -1)
            return self.norm(self.out_proj(out) + res)

    class _ConceptGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj  = nn.Linear(NODE_DIM, HIDDEN_DIM)
            self.edge_embed  = nn.Embedding(len(RELATION_TYPES), EDGE_DIM)
            self.gat_layers  = nn.ModuleList([
                _GATLayer(HIDDEN_DIM, HIDDEN_DIM, NUM_HEADS, EDGE_DIM)
                for _ in range(NUM_LAYERS)
            ])
            self.output_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.relu        = nn.ReLU()

        def forward(
            self,
            node_feats:  "torch.Tensor",   # (N, NODE_DIM)
            adj:         "torch.Tensor",   # (N, N)
            edge_types:  "torch.Tensor",   # (N, N) int
        ) -> "torch.Tensor":               # (N, HIDDEN_DIM)
            h = self.relu(self.input_proj(node_feats))
            ef = self.edge_embed(edge_types)    # (N, N, EDGE_DIM)
            for layer in self.gat_layers:
                h = self.relu(layer(h, adj, ef))
            return self.output_proj(h)

        def embed_query(self, text_feat: "torch.Tensor") -> "torch.Tensor":
            """Project a single BOW vector into embedding space."""
            with torch.no_grad():
                return self.output_proj(
                    self.relu(self.input_proj(text_feat.unsqueeze(0)))
                )[0]

else:
    _ConceptGNN = None  # type: ignore


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

class GNNReasoner:
    """
    Wraps _ConceptGNN for concept graph reasoning.
    All public methods return safe fallbacks if PyTorch is missing.
    """

    def __init__(self) -> None:
        self.available = _TORCH_AVAILABLE
        self._model: Optional["_ConceptGNN"] = None
        if self.available:
            self._load_or_init()

    def _load_or_init(self) -> None:
        import torch
        self._model = _ConceptGNN()
        self._model.eval()
        if _MODEL_PATH.exists():
            try:
                state = torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=True)
                self._model.load_state_dict(state)
            except Exception:
                pass
        else:
            _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        if self._model is not None:
            import torch
            torch.save(self._model.state_dict(), str(_MODEL_PATH))

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(
        self,
        concepts: List[ConceptNode],
        relations: List[RelationEdge],
    ) -> Optional[Tuple]:
        if not concepts:
            return None
        import torch
        N = len(concepts)
        idx = {c.topic.lower(): i for i, c in enumerate(concepts)}

        node_feats = torch.tensor(
            [[f * c.confidence for f in _bow_embed(f"{c.topic} {c.summary}")]
             for c in concepts],
            dtype=torch.float32,
        )
        adj        = torch.eye(N)                              # self-loops
        edge_types = torch.zeros(N, N, dtype=torch.long)

        for rel in relations:
            si = idx.get(rel.source.lower())
            ti = idx.get(rel.target.lower())
            if si is not None and ti is not None:
                adj[si, ti] = rel.confidence
                adj[ti, si] = rel.confidence
                rt = RELATION_TYPES.index(rel.relation) \
                     if rel.relation in RELATION_TYPES else len(RELATION_TYPES) - 1
                edge_types[si, ti] = rt
                edge_types[ti, si] = rt

        return node_feats, adj, edge_types, idx

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute_embeddings(
        self,
        concepts: List[ConceptNode],
        relations: List[RelationEdge],
    ) -> Optional[Dict[str, List[float]]]:
        """Run GNN forward pass → {topic: embedding}."""
        if not self.available or self._model is None or not concepts:
            return None
        graph = self._build_graph(concepts, relations)
        if graph is None:
            return None
        node_feats, adj, edge_types, idx = graph
        with torch.no_grad():
            embs = self._model(node_feats, adj, edge_types)
        rev = {v: k for k, v in idx.items()}
        return {rev[i]: embs[i].tolist() for i in range(len(concepts))}

    def rank_concepts_by_query(
        self,
        query: str,
        concepts: List[ConceptNode],
        relations: List[RelationEdge],
    ) -> List[Tuple[str, float]]:
        """
        Rank concept topics by GNN relevance to query.
        Falls back to confidence-based ranking if torch unavailable.
        """
        if not self.available or not concepts:
            return [(c.topic, c.confidence) for c in concepts]

        embs = self.compute_embeddings(concepts, relations)
        if embs is None:
            return [(c.topic, c.confidence) for c in concepts]

        import torch
        q_feat = torch.tensor(_bow_embed(query, NODE_DIM), dtype=torch.float32)
        q_emb  = self._model.embed_query(q_feat)
        q_norm = q_emb / (q_emb.norm() + 1e-8)

        ranked = []
        for topic, emb_list in embs.items():
            emb  = torch.tensor(emb_list, dtype=torch.float32)
            norm = emb / (emb.norm() + 1e-8)
            score = float((q_norm * norm).sum())
            ranked.append((topic, (score + 1.0) / 2.0))

        return sorted(ranked, key=lambda x: x[1], reverse=True)
