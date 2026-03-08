"""
Cross-Domain Generalization — structural analogy engine.

Maintains a curated library of domain bridges (structural mappings between
fields). Given a query, detects which domains are present and surfaces
relevant analogies to enrich reasoning.

No LLM required for the static library — pure keyword matching.
An optional LLM path generates novel analogies for complex queries
and caches results in SQLite.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cortana import config

_DB_PATH = config.SQLITE_PATH


# ---------------------------------------------------------------------------
# Domain registry  (domain → (core concepts, key terms))
# ---------------------------------------------------------------------------

_DOMAINS: Dict[str, Tuple[List[str], List[str]]] = {
    "mathematics": (
        ["proof", "theorem", "axiom", "function", "set", "mapping", "topology"],
        ["equation", "derivative", "integral", "matrix", "vector", "eigenvalue"],
    ),
    "biology": (
        ["organism", "evolution", "cell", "gene", "ecosystem", "adaptation", "reproduction"],
        ["protein", "dna", "mutation", "selection", "metabolism", "neuron", "synapse"],
    ),
    "physics": (
        ["force", "energy", "field", "particle", "wave", "entropy", "equilibrium"],
        ["momentum", "quantum", "relativity", "thermodynamics", "gravity", "photon"],
    ),
    "computer_science": (
        ["algorithm", "data structure", "recursion", "abstraction", "state", "network"],
        ["graph", "tree", "hash", "cache", "concurrency", "compiler", "complexity"],
    ),
    "economics": (
        ["supply", "demand", "equilibrium", "incentive", "market", "trade", "utility"],
        ["price", "capital", "investment", "risk", "scarcity", "competition", "externality"],
    ),
    "psychology": (
        ["behavior", "cognition", "memory", "attention", "learning", "motivation", "emotion"],
        ["perception", "reinforcement", "schema", "bias", "heuristic", "conditioning"],
    ),
    "philosophy": (
        ["truth", "knowledge", "ethics", "logic", "consciousness", "causation", "identity"],
        ["epistemology", "ontology", "dialectic", "realism", "phenomenology"],
    ),
    "systems_theory": (
        ["feedback", "emergence", "self-organization", "hierarchy", "resilience", "attractor"],
        ["nonlinear", "complex", "adaptive", "homeostasis", "entropy", "dynamics"],
    ),
    "linguistics": (
        ["syntax", "semantics", "pragmatics", "grammar", "meaning", "context", "sign"],
        ["morpheme", "phoneme", "metaphor", "ambiguity", "reference", "discourse"],
    ),
    "game_theory": (
        ["strategy", "payoff", "cooperation", "defection", "player", "information"],
        ["nash", "dominant", "zero-sum", "auction", "mechanism", "signaling"],
    ),
    "neuroscience": (
        ["brain", "neural", "synapse", "cortex", "plasticity", "circuit", "signal"],
        ["dopamine", "serotonin", "spike", "inhibition", "excitation", "oscillation"],
    ),
    "thermodynamics": (
        ["heat", "work", "entropy", "temperature", "pressure", "phase", "equilibrium"],
        ["carnot", "boltzmann", "free energy", "reversible", "irreversible"],
    ),
}


# ---------------------------------------------------------------------------
# Structural bridge library
# ---------------------------------------------------------------------------

@dataclass
class Bridge:
    concept: str                  # abstract structural concept
    domain_map: Dict[str, str]    # domain → how this concept manifests


_BRIDGES: List[Bridge] = [
    Bridge("optimization", {
        "mathematics":      "gradient descent / convex optimisation",
        "biology":          "natural selection",
        "economics":        "market equilibrium seeking",
        "computer_science": "search algorithms",
        "physics":          "energy minimisation",
        "psychology":       "reinforcement learning / habit formation",
        "thermodynamics":   "entropy minimisation toward equilibrium",
    }),
    Bridge("feedback_loop", {
        "biology":          "homeostasis",
        "economics":        "price adjustment mechanism",
        "computer_science": "PID controller / control systems",
        "physics":          "damped oscillation",
        "psychology":       "habit reinforcement cycle",
        "systems_theory":   "negative / positive feedback",
        "neuroscience":     "recurrent neural circuits",
    }),
    Bridge("emergence", {
        "biology":          "consciousness from neurons",
        "physics":          "phase transitions (liquid → solid)",
        "economics":        "market prices from individual trades",
        "computer_science": "complex behaviour from simple cellular automata rules",
        "systems_theory":   "self-organisation",
        "linguistics":      "meaning from symbol combinations",
    }),
    Bridge("compression", {
        "mathematics":      "lossy/lossless encoding",
        "biology":          "genetic regulatory networks",
        "computer_science": "data compression algorithms",
        "linguistics":      "abstraction / categorical perception",
        "physics":          "renormalisation group",
        "psychology":       "chunking in working memory",
        "neuroscience":     "sparse coding in cortex",
    }),
    Bridge("hierarchy", {
        "biology":          "taxonomic classification",
        "computer_science": "call stack / type hierarchy",
        "mathematics":      "group theory subgroups",
        "linguistics":      "syntactic parse tree",
        "economics":        "organisational structure",
        "philosophy":       "ontological categories",
    }),
    Bridge("information_flow", {
        "biology":          "gene expression / signalling cascade",
        "computer_science": "data pipeline / message passing",
        "neuroscience":     "action potential propagation",
        "economics":        "price signals in markets",
        "physics":          "wave propagation",
        "linguistics":      "speech act / discourse",
    }),
    Bridge("phase_transition", {
        "physics":          "solid ↔ liquid ↔ gas",
        "mathematics":      "bifurcation point in dynamical systems",
        "economics":        "market crash / regime change",
        "psychology":       "insight / paradigm shift",
        "biology":          "metamorphosis",
        "systems_theory":   "critical transition",
    }),
    Bridge("exploration_exploitation", {
        "computer_science": "multi-armed bandit / UCB algorithm",
        "biology":          "foraging vs territory defence",
        "economics":        "R&D investment vs production",
        "psychology":       "curiosity vs habit",
        "game_theory":      "mixed strategy equilibrium",
    }),
    Bridge("invariance", {
        "mathematics":      "symmetry / group invariant",
        "physics":          "conservation laws (Noether's theorem)",
        "computer_science": "abstraction / interface",
        "linguistics":      "semantic invariance across paraphrase",
        "philosophy":       "essence vs accident",
    }),
    Bridge("network_effects", {
        "economics":        "Metcalfe's law / platform markets",
        "biology":          "mutualistic ecosystem webs",
        "computer_science": "graph connectivity / viral spread",
        "sociology":        "social contagion",
        "neuroscience":     "connectome dynamics",
    }),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DomainDetection:
    domain: str
    score: float
    matched_terms: List[str]


@dataclass
class Analogy:
    concept: str
    source_domain: str
    target_domain: str
    source_form: str
    target_form: str
    confidence: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CrossDomainEngine:
    """Detects domains in queries and surfaces structural analogies."""

    def __init__(self, reasoning: Any = None) -> None:
        self.reasoning = reasoning
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(_DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agi_analogy_cache (
                    fingerprint TEXT PRIMARY KEY,
                    analogy     TEXT NOT NULL,
                    ts          REAL NOT NULL
                )
            """)

    def detect_domains(self, text: str) -> List[DomainDetection]:
        text_lower = text.lower()
        results: List[DomainDetection] = []
        for domain, (concepts, terms) in _DOMAINS.items():
            matched = [
                w for w in concepts + terms
                if re.search(r"\b" + re.escape(w) + r"\b", text_lower)
            ]
            if matched:
                score = len(matched) / (len(concepts) + len(terms))
                results.append(DomainDetection(domain, round(score, 3), matched))
        return sorted(results, key=lambda d: d.score, reverse=True)

    def find_analogies(self, text: str, max_analogies: int = 3) -> List[Analogy]:
        domains = self.detect_domains(text)
        if not domains:
            return []
        primary = domains[0].domain
        text_lower = text.lower()
        analogies: List[Analogy] = []

        for bridge in _BRIDGES:
            # Check if bridge concept appears in query
            concept_words = bridge.concept.replace("_", " ").split()
            if not any(w in text_lower for w in concept_words):
                continue
            if primary not in bridge.domain_map:
                continue
            source_form = bridge.domain_map[primary]
            for target_domain, target_form in bridge.domain_map.items():
                if target_domain == primary:
                    continue
                analogies.append(Analogy(
                    concept=bridge.concept,
                    source_domain=primary,
                    target_domain=target_domain,
                    source_form=source_form,
                    target_form=target_form,
                    confidence=0.72,
                ))
                if len(analogies) >= max_analogies:
                    return analogies
        return analogies

    def build_transfer_prompt(self, text: str) -> str:
        """Build a prompt augmentation string for cross-domain transfer."""
        domains = self.detect_domains(text)
        analogies = self.find_analogies(text)
        if not domains and not analogies:
            return ""

        lines = ["--- Cross-Domain Context ---"]
        if domains:
            top = domains[:2]
            lines.append(
                "Detected domains: "
                + ", ".join(f"{d.domain} ({len(d.matched_terms)} signals)" for d in top)
            )
        if analogies:
            lines.append("Structural analogies:")
            for a in analogies:
                lines.append(
                    f"  • [{a.concept}] {a.source_domain} ({a.source_form}) "
                    f"↔ {a.target_domain} ({a.target_form})"
                )
            lines.append(
                "Use these cross-domain mappings to enrich your reasoning — "
                "structural parallels often yield non-obvious insights."
            )
        return "\n".join(lines)

    def _fingerprint(self, text: str) -> str:
        return hashlib.md5(text[:200].encode()).hexdigest()
