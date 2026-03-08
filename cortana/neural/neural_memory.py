"""
Neural Memory — unified interface combining RNN + GNN.

Provides neural-augmented recall scores to Layer 2 and
the Cognitive Architecture (Layer 17).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from cortana.neural.rnn_encoder import RNNEncoder
from cortana.neural.gnn_reasoner import GNNReasoner
from cortana.models.schemas import ConceptNode, RelationEdge


class NeuralMemory:
    """
    Combines RNN temporal encoding with GNN graph reasoning to
    produce neural-augmented relevance scores for memory items.
    """

    def __init__(self) -> None:
        self.rnn = RNNEncoder()
        self.gnn = GNNReasoner()
        self.available = self.rnn.available or self.gnn.available

    # ------------------------------------------------------------------
    # Augmented recall
    # ------------------------------------------------------------------

    def augment_recall(
        self,
        query: str,
        episodes: List[str],
        concepts: List[ConceptNode],
        relations: List[RelationEdge],
        base_results: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Re-rank base_results using RNN temporal context + GNN concept scores.

        Returns list of (text, score) sorted descending.
        base_results that are not in the scored list retain score 0.5.
        """
        if not base_results:
            return []

        # 1. RNN: temporal context vector from recent episodes
        temporal_ctx = self.rnn.encode_sequence(episodes) if episodes else None

        # 2. GNN: topic relevance scores
        concept_scores: Dict[str, float] = {}
        if concepts:
            for topic, score in self.gnn.rank_concepts_by_query(query, concepts, relations):
                concept_scores[topic.lower()] = score

        scored: List[Tuple[str, float]] = []
        for text in base_results:
            score = 0.5  # base

            # RNN component: cosine sim between temporal ctx and text embedding
            if temporal_ctx is not None:
                rnn_s = self.rnn.score_concepts(temporal_ctx, [text[:120]])
                score = 0.4 * score + 0.35 * rnn_s[0]

            # GNN component: boost if text mentions high-ranking concept topics
            if concept_scores:
                text_lower = text.lower()
                boosts, hits = 0.0, 0
                for topic, cs in concept_scores.items():
                    if topic in text_lower:
                        boosts += cs
                        hits   += 1
                if hits:
                    score += 0.25 * (boosts / hits)

            scored.append((text, min(score, 1.0)))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def propagate_concept_graph(
        self,
        seed_topics: List[str],
        concepts: List[ConceptNode],
        relations: List[RelationEdge],
        hops: int = 2,
    ) -> List[str]:
        """Walk concept graph from seed_topics, return reachable topics."""
        if not concepts or not relations:
            return []

        adj: Dict[str, List[str]] = {}
        for rel in relations:
            adj.setdefault(rel.source.lower(), []).append(rel.target.lower())
            adj.setdefault(rel.target.lower(), []).append(rel.source.lower())

        visited = {s.lower() for s in seed_topics}
        frontier = list(visited)

        for _ in range(hops):
            nxt = []
            for node in frontier:
                for nbr in adj.get(node, []):
                    if nbr not in visited:
                        visited.add(nbr)
                        nxt.append(nbr)
            frontier = nxt

        seeds_lower = {s.lower() for s in seed_topics}
        return [t for t in visited if t not in seeds_lower][:10]

    # ------------------------------------------------------------------
    # Temporal summary (no LLM call)
    # ------------------------------------------------------------------

    def get_temporal_context_summary(self, episodes: List[str]) -> str:
        """Heuristic: extract recurring themes from recent episodes."""
        if not episodes:
            return ""
        counts: Dict[str, int] = {}
        for ep in episodes[-10:]:
            for word in ep.lower().split():
                if len(word) > 4:
                    counts[word] = counts.get(word, 0) + 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]
        if not top:
            return ""
        return "Recurring themes: " + ", ".join(w for w, _ in top)
