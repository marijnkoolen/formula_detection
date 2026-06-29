"""
phrase_typing.py — Human-in-the-loop categorisation of formulaic phrases
into user-defined types.

While exploring a corpus's formulaic phrases, a user typically recognises
several distinct *types* of partial formula (e.g. "opening formula",
"decision formula", "resumption formula"), each of which may be realised
by one or more differently-worded phrases (a "community" of variants).
A phrase can plausibly belong to more than one type — this module treats
typing as a multi-label problem throughout: each type is scored
independently against each candidate phrase (binary relevance), rather
than forcing a single partition.

Two pieces support the iterative workflow:

- ``PhraseTypeRegistry`` — a simple, explicit record of user-confirmed
  type memberships. A phrase may belong to any number of types.
- ``TypeSuggester`` — given the registry's confirmed members for a type,
  ranks the remaining uncategorised phrases by how strongly they co-occur
  with that type's confirmed members (the primary signal — functional
  position is more diagnostic here than wording) and by context-word
  similarity (a secondary signal, most useful when a type has very few
  confirmed members so far).

``concordance()`` renders example text windows around a phrase's
occurrences, sampled across the whole position list rather than just the
first few, so usage drift across a corpus is visible while categorising.

Everything here operates on ``phrase_positions`` (as produced by
``PhraseCooccurrenceIndex``) and the flat token stream — no document
boundaries or additional indexing required.
"""
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple


class PhraseType:
    """A user-defined category of formulaic phrase.

    Attributes:
        name: Type name, e.g. "decision formula".
        members: Set of phrase strings the user has confirmed as
            belonging to this type.
        notes: Free-text notes the user can attach (e.g. why this type
            was created, or what distinguishes it from a similar type).
    """

    def __init__(self, name: str, members: Optional[Set[str]] = None, notes: str = ''):
        self.name = name
        self.members: Set[str] = set(members) if members else set()
        self.notes = notes

    def __repr__(self) -> str:
        return f"PhraseType(name={self.name!r}, n_members={len(self.members)})"

    def __len__(self) -> int:
        return len(self.members)


class PhraseTypeRegistry:
    """Tracks user-confirmed phrase-to-type assignments.

    Multi-label throughout: a phrase may be assigned to any number of
    types, and assigning it to one type has no effect on its membership
    in any other.
    """

    def __init__(self):
        self.types: Dict[str, PhraseType] = {}

    def __repr__(self) -> str:
        return f"PhraseTypeRegistry(types={list(self.types.keys())})"

    def create_type(self, name: str, notes: str = '') -> PhraseType:
        """Create a new, empty type. No-op (returns the existing type) if it already exists."""
        if name not in self.types:
            self.types[name] = PhraseType(name, notes=notes)
        return self.types[name]

    def assign(self, phrase: str, type_names) -> None:
        """Assign *phrase* to one or more types, creating any that don't exist yet.

        Args:
            phrase: Phrase string to assign.
            type_names: A single type name (str) or an iterable of type names.
        """
        if isinstance(type_names, str):
            type_names = [type_names]
        for type_name in type_names:
            self.create_type(type_name)
            self.types[type_name].members.add(phrase)

    def unassign(self, phrase: str, type_name: str) -> None:
        """Remove *phrase* from *type_name*'s membership, if present."""
        if type_name in self.types:
            self.types[type_name].members.discard(phrase)

    def types_of(self, phrase: str) -> List[str]:
        """Return all type names *phrase* is currently assigned to."""
        return [name for name, t in self.types.items() if phrase in t.members]

    def all_categorised(self) -> Set[str]:
        """Return the set of all phrases assigned to at least one type."""
        result: Set[str] = set()
        for t in self.types.values():
            result.update(t.members)
        return result

    def uncategorised(self, candidates: Sequence[str]) -> List[str]:
        """Return the subset of *candidates* not yet assigned to any type."""
        categorised = self.all_categorised()
        return [phrase for phrase in candidates if phrase not in categorised]

    def summary(self) -> str:
        """Human-readable one-line-per-type summary."""
        lines = []
        for name, t in sorted(self.types.items()):
            lines.append(f"{name} ({len(t.members)}): {sorted(t.members)}")
        return '\n'.join(lines)


def concordance(phrase: str, flat_stream: Sequence[str],
                positions: Optional[List[int]] = None,
                phrase_positions: Optional[Dict[str, List[int]]] = None,
                window: int = 10, max_examples: int = 10) -> List[str]:
    """Render example text windows around occurrences of *phrase*.

    Samples occurrences spread across the whole position list (evenly
    spaced indices) rather than just the first *max_examples*, so usage
    drift across the corpus (different scribes, different years) is more
    likely to show up while browsing.

    Args:
        phrase: Phrase string (space-separated tokens) to look up.
        flat_stream: The full token sequence the positions index into.
        positions: Explicit list of token offsets for *phrase*. Provide
            this or *phrase_positions* (not both).
        phrase_positions: Dict mapping phrase to its position list, as
            produced by ``PhraseCooccurrenceIndex.phrase_positions``; used
            to look up *phrase* if *positions* is not given directly.
        window: Number of tokens of context to show on each side.
        max_examples: Maximum number of example windows to return.

    Returns:
        List of rendered strings, the matched phrase wrapped in ``[...]``.
    """
    if positions is None:
        if phrase_positions is None:
            raise ValueError('must provide either positions or phrase_positions')
        positions = phrase_positions.get(phrase, [])
    if not positions:
        return []

    phrase_len = len(phrase.split())
    if len(positions) <= max_examples:
        sampled = positions
    else:
        step = len(positions) / max_examples
        sampled = [positions[int(i * step)] for i in range(max_examples)]

    lines = []
    n = len(flat_stream)
    for pos in sampled:
        start = max(0, pos - window)
        end = min(n, pos + phrase_len + window)
        pre = ' '.join(flat_stream[start:pos])
        match = ' '.join(flat_stream[pos:pos + phrase_len])
        post = ' '.join(flat_stream[pos + phrase_len:end])
        lines.append(f"{pre} [{match}] {post}")
    return lines


def _nearest_distance(sorted_positions: List[int], value: int) -> Optional[int]:
    """Distance from *value* to the nearest entry in *sorted_positions*."""
    if not sorted_positions:
        return None
    idx = bisect_left(sorted_positions, value)
    candidates = []
    if idx < len(sorted_positions):
        candidates.append(abs(sorted_positions[idx] - value))
    if idx > 0:
        candidates.append(abs(value - sorted_positions[idx - 1]))
    return min(candidates) if candidates else None


class TypeSuggester:
    """Ranks uncategorised phrases by affinity to a type's confirmed members.

    Two independent signals are computed per candidate phrase:

    - ``cooc_affinity`` (primary) — the fraction of the candidate's own
      occurrences that fall within the type's own characteristic
      recurrence window of *some* confirmed member's occurrence. The
      window is estimated automatically from the confirmed members'
      combined occurrence stream (the same single-link clustering idea
      used by ``WindowEstimator.cluster_based_window``), so a type
      seeded with day-scale phrases gets a day-scale window and one
      seeded with resolution-scale phrases gets a resolution-scale window
      — fitted from the user's own confirmed examples, not guessed.
    - ``context_similarity`` (secondary) — cosine similarity between the
      candidate's local context-word profile and the centroid of the
      type's confirmed members' profiles. Most informative when a type
      has very few confirmed members and ``cooc_affinity`` is not yet
      reliable.

    Both are returned rather than combined into one score, so the user
    can judge which signal to trust for a given type.

    Args:
        phrase_positions: Dict mapping phrase string to sorted token
            offsets, as produced by ``PhraseCooccurrenceIndex.phrase_positions``.
        flat_stream: The token stream the positions index into, used to
            build context-word profiles.
        context_window: Number of tokens on each side of an occurrence to
            count as context for ``context_similarity``.
    """

    def __init__(self, phrase_positions: Dict[str, List[int]],
                 flat_stream: Sequence[str],
                 context_window: int = 10):
        self.phrase_positions = phrase_positions
        self.flat_stream = flat_stream
        self.context_window = context_window
        self._context_profile_cache: Dict[str, Counter] = {}

    def _context_profile(self, phrase: str) -> Counter:
        if phrase in self._context_profile_cache:
            return self._context_profile_cache[phrase]
        profile: Counter = Counter()
        phrase_len = len(phrase.split())
        n = len(self.flat_stream)
        for pos in self.phrase_positions.get(phrase, []):
            start = max(0, pos - self.context_window)
            end = min(n, pos + phrase_len + self.context_window)
            for i in range(start, pos):
                profile[self.flat_stream[i]] += 1
            for i in range(pos + phrase_len, end):
                profile[self.flat_stream[i]] += 1
        self._context_profile_cache[phrase] = profile
        return profile

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[w] * b[w] for w in a if w in b)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def suggest_for_type(self, type_members: Set[str],
                         candidates: Sequence[str],
                         window: float = 100,
                         top_n: int = 20) -> List[Tuple[str, float, float]]:
        """Rank *candidates* by affinity to *type_members*.

        Args:
            type_members: Confirmed member phrases of the type to score against.
            candidates: Phrases to rank (typically the still-uncategorised set).
            window: Maximum token distance for a candidate occurrence to
                count as "near" a confirmed member's occurrence. There is
                deliberately no automatic estimate here: on this kind of
                corpus, every automatic window estimator we've tried
                (``WindowEstimator.cluster_based_window`` included) is
                unreliable once a type's members are individually very
                frequent, because the corpus no longer has a clean
                burst-vs-gap structure to exploit (see
                ``Document-Pattern-Detection-Findings.md``). Pick a window
                that matches the scale you're trying to capture for this
                particular type — e.g. tens of tokens for a same-decision
                relationship, hundreds for same-resolution, thousands for
                same-day — and sanity-check the results rather than
                trusting a single default across very different types.
            top_n: Maximum number of results to return.

        Returns:
            List of ``(phrase, cooc_affinity, context_similarity)`` tuples,
            sorted by ``cooc_affinity`` descending. ``cooc_affinity`` is in
            [0, 1]; ``context_similarity`` is cosine similarity in [0, 1].
        """
        member_positions = sorted(
            pos for member in type_members for pos in self.phrase_positions.get(member, [])
        )
        if not member_positions:
            return []

        type_profile: Counter = Counter()
        for member in type_members:
            type_profile.update(self._context_profile(member))

        results: List[Tuple[str, float, float]] = []
        for phrase in candidates:
            if phrase in type_members:
                continue
            positions = self.phrase_positions.get(phrase, [])
            if not positions:
                continue
            within = sum(
                1 for pos in positions
                if (d := _nearest_distance(member_positions, pos)) is not None and d <= window
            )
            cooc_affinity = within / len(positions)
            context_similarity = self._cosine(self._context_profile(phrase), type_profile)
            results.append((phrase, cooc_affinity, context_similarity))

        results.sort(key=lambda r: -r[1])
        return results[:top_n]

    def context_vectors(self, phrases: Sequence[str]):
        """Build a sparse tf-idf-like matrix of context vectors for *phrases*.

        Intended for clustering only the currently-uncategorised phrases
        (via ``formula_detection.clustering.cluster_phrases``) to bootstrap
        discovery of new candidate types, rather than only refining types
        the user has already noticed.

        Returns:
            ``(matrix, phrases)`` tuple, matrix is a
            ``scipy.sparse.csr_matrix`` of shape ``(len(phrases), n_features)``.
        """
        import numpy as np
        import scipy.sparse

        profiles = [self._context_profile(p) for p in phrases]
        vocab: Dict[str, int] = {}
        for profile in profiles:
            for word in profile:
                if word not in vocab:
                    vocab[word] = len(vocab)
        doc_freq: Counter = Counter()
        for profile in profiles:
            doc_freq.update(profile.keys())

        rows, cols, data = [], [], []
        n_docs = len(phrases) or 1
        for pi, profile in enumerate(profiles):
            total = sum(profile.values()) or 1
            for word, count in profile.items():
                tf = count / total
                idf = math.log((1 + n_docs) / (1 + doc_freq[word]))
                score = tf * idf
                if score != 0.0:
                    rows.append(pi)
                    cols.append(vocab[word])
                    data.append(score)
        matrix = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(phrases), len(vocab)), dtype=np.float32
        )
        return matrix, list(phrases)
