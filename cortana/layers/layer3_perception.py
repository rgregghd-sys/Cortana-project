"""
Layer 3 — Perception
Normalize user input, extract metadata, detect intent and complexity.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List

from cortana.models.schemas import UserInput, PerceivedInput


# Keywords that steer intent classification
_CONVERSATIONAL_KEYWORDS = {
    "hello", "hi", "hey", "howdy", "greetings", "sup", "yo",
    "how are you", "how's it going", "what's up", "what are you",
    "who are you", "tell me about yourself", "what can you do",
    "do you", "are you", "can you", "what do you think", "opinion",
    "thanks", "thank you", "goodbye", "bye", "cool", "nice", "awesome",
    "what's your", "do you like", "favorite", "feel", "feelings",
}
_RESEARCH_KEYWORDS = {
    # Only fire on words that clearly signal a need for external information
    "research", "search", "look up", "news", "latest", "discover",
    "investigate", "wikipedia", "citation", "source", "current events",
    "today's", "recent", "breaking",
}
_CODE_KEYWORDS = {
    "code", "implement", "program", "function", "class", "script",
    "python", "javascript", "debug", "fix", "test", "run", "execute",
    "algorithm", "refactor", "build",
}
_ANALYSIS_KEYWORDS = {
    "analyze", "analyse", "compare", "contrast", "evaluate", "assess",
    "review", "pros", "cons", "trade-off", "breakdown", "metrics",
    "data", "table", "chart", "insight", "statistics",
}
_CREATIVE_KEYWORDS = {
    "story", "poem", "creative", "imagine", "draft", "compose",
    "brainstorm", "invent", "generate",
}
_SELF_DESIGN_KEYWORDS = {
    "redesign yourself", "design yourself", "change your appearance",
    "update your model", "create your model", "rebuild yourself",
    "new look", "change your look", "design your body", "make yourself",
    "design a new model", "design yourself", "change your 3d",
    "update your 3d", "new 3d model",
}
_DEVAI_KEYWORDS = {
    "devai", "dev ai", "scan my code", "scan code", "review my code",
    "code review", "code improvements", "improve my code", "check my code",
    "code analysis", "any bugs", "code quality", "approve #", "reject #",
    "devai status", "devai history", "devai scan", "pending proposals",
    "code suggestions", "code issues",
}


class PerceptionLayer:
    """
    Normalizes raw user input into a PerceivedInput with:
    - intent classification (simple / research / code / analysis / creative)
    - complexity score (0.0–1.0) driving whether planning is invoked
    - extracted keywords for memory recall
    - emotional tone detection
    """

    def perceive(self, user_input: UserInput) -> PerceivedInput:
        content = self._normalize(user_input)
        intent = self._classify_intent(content)
        complexity = self._estimate_complexity(content, intent)
        keywords = self._extract_keywords(content)
        emotional_tone = self._detect_emotional_tone(content)

        return PerceivedInput(
            content=content,
            intent=intent,
            complexity=complexity,
            keywords=keywords,
            emotional_tone=emotional_tone,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize(self, user_input: UserInput) -> str:
        """Strip excess whitespace; for file inputs, prepend a label."""
        raw = user_input.raw.strip()
        if user_input.input_type == "file":
            path = user_input.metadata.get("file_path", "unknown")
            return f"[File: {path}]\n{raw}"
        if user_input.input_type == "image":
            path = user_input.metadata.get("image_path", "unknown")
            return f"[Image: {path}]\n{raw}"
        return raw

    def _classify_intent(self, content: str) -> str:
        lower = content.lower()
        words = set(re.findall(r"\b\w+\b", lower))

        # Self-design: check multi-word phrases first (highest priority)
        for phrase in _SELF_DESIGN_KEYWORDS:
            if phrase in lower:
                return "self_design"

        # DevAI: code review / improvement requests
        for phrase in _DEVAI_KEYWORDS:
            if phrase in lower:
                return "devai"

        # Short messages (≤ 10 words) are conversational unless they clearly aren't
        if len(content.split()) <= 10:
            strong_signals = words & (_CODE_KEYWORDS | _ANALYSIS_KEYWORDS | _RESEARCH_KEYWORDS)
            if not strong_signals:
                return "conversational"

        # Check for conversational phrases (multi-word patterns)
        for phrase in _CONVERSATIONAL_KEYWORDS:
            if phrase in lower:
                return "conversational"

        scores = {
            "research":  len(words & _RESEARCH_KEYWORDS),
            "code":      len(words & _CODE_KEYWORDS),
            "analysis":  len(words & _ANALYSIS_KEYWORDS),
            "creative":  len(words & _CREATIVE_KEYWORDS),
        }

        best_intent = max(scores, key=lambda k: scores[k])
        if scores[best_intent] == 0:
            return "simple"

        # "code" and "creative" share some words — break tie by context
        if best_intent == "creative" and scores["code"] >= scores["creative"]:
            if any(kw in lower for kw in ("python", "function", "class", "script")):
                return "code"

        return best_intent

    def _estimate_complexity(self, content: str, intent: str) -> float:
        """
        Heuristic complexity score.
        Factors: length, question count, multi-step conjunctions, intent.
        """
        score = 0.0

        # Length factor (caps at 0.3)
        word_count = len(content.split())
        score += min(word_count / 200, 0.3)

        # Multiple sentences or questions → more complex
        sentence_count = len(re.split(r"[.!?]+", content))
        score += min((sentence_count - 1) * 0.05, 0.2)

        # Multi-step indicators
        multi_step = re.findall(
            r"\b(then|after that|next|also|additionally|and then|furthermore)\b",
            content.lower(),
        )
        score += min(len(multi_step) * 0.1, 0.3)

        # Intent bonus
        intent_bonus = {
            "simple": 0.0,
            "research": 0.15,
            "analysis": 0.2,
            "creative": 0.1,
            "code": 0.2,
            "self_design": 0.3,
        }
        score += intent_bonus.get(intent, 0.0)

        return round(min(score, 1.0), 3)

    def _detect_emotional_tone(self, content: str) -> str:
        """
        Heuristic emotional tone detection from user input.
        Returns one of: neutral | frustrated | curious | excited | confused | playful
        """
        lower = content.lower()

        # Playful — check first to avoid misclassifying jokes as frustrated
        if re.search(r'\b(haha|hehe|lol|lmao|rofl|funny|joking|jk|kidding|😂|🤣|:D|:P)\b', lower):
            return "playful"

        # Frustrated
        if (
            re.search(r'\b(ugh|argh|wtf|not working|broken|still (doesn\'t|not|broken)|why (won\'t|doesn\'t|isn\'t|can\'t)|so frustrating|keep(s)? (failing|breaking)|again\?|makes no sense)\b', lower)
            or content.count('!') >= 2
            or '???' in content
        ):
            return "frustrated"

        # Confused
        if re.search(r'\b(confused|don\'t understand|not sure (what|how|why|if)|lost|unclear|what (does|is|do) .{0,30} mean|can you (clarify|explain)|i don\'t get)\b', lower):
            return "confused"

        # Excited
        if (
            re.search(r'\b(amazing|awesome|incredible|love (this|it|that)|this is great|this is (so )?cool|wow|brilliant|perfect|exactly what)\b', lower)
            or (content.count('!') >= 1 and len(content.split()) <= 12)
        ):
            return "excited"

        # Curious
        if re.search(r'\b(wondering|curious|what if|how (does|do|would|could|come)|why (does|do|would|is)|i\'ve been thinking|just (want|wanted) to (know|understand)|out of curiosity)\b', lower):
            return "curious"

        return "neutral"

    def _extract_keywords(self, content: str) -> List[str]:
        """
        Simple keyword extraction: remove stopwords, return top-10 tokens.
        """
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "should", "may", "might", "must", "shall",
            "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "or", "but", "if", "as", "that", "this", "which",
            "what", "how", "why", "when", "where", "who", "please",
        }
        words = re.findall(r"\b[a-z]{3,}\b", content.lower())
        filtered = [w for w in words if w not in stopwords]
        # Deduplicate while preserving order
        seen: set = set()
        unique = [w for w in filtered if not (w in seen or seen.add(w))]  # type: ignore[func-returns-value]
        return unique[:10]
