"""
phrase_cooccurrence.py — Phrase-level co-occurrence index.

Counts how often pairs of formulaic phrases occur near each other in an
arbitrarily long token stream, at multiple configurable distance scales.
Comparing LLR scores across scales lets you distinguish within-formula
co-occurrence (short window) from within-document co-occurrence (long
window) without needing pre-defined document boundaries.

Designed for single-pass, memory-efficient processing of large streams
(hundreds of millions of tokens) via a sliding deque.
"""
import math
from collections import Counter, defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple


def detect_communities(graph, weight: str = 'weight') -> Dict[str, int]:
    """Run Louvain community detection on a phrase co-occurrence graph.

    Shared by ``FormulaSearch.detect_document_patterns`` (community
    detection at a document/element scale) and
    ``formula_detection.motif.merge_overlapping_phrases`` (community
    detection at a much smaller, same-formula-variant scale) — the same
    underlying primitive, applied to graphs built at different scales.

    Args:
        graph: A networkx.Graph with phrases as nodes.
        weight: Edge attribute to use as edge weight.

    Returns:
        Dict mapping phrase -> community ID. Empty if the graph has no edges.
    """
    import networkx as nx
    if graph.number_of_edges() == 0:
        return {}
    communities = nx.algorithms.community.louvain_communities(graph, weight=weight, seed=0)
    community_map: Dict[str, int] = {}
    for community_id, phrases in enumerate(communities):
        for phrase in phrases:
            community_map[phrase] = community_id
    return community_map


def _llr(k11: int, k12: int, k21: int, k22: int) -> float:
    """Log-likelihood ratio for a 2×2 co-occurrence contingency table."""
    n = k11 + k12 + k21 + k22
    if n == 0:
        return 0.0

    def _safe_xlogx(observed: int, expected: float) -> float:
        if observed == 0 or expected <= 0:
            return 0.0
        return observed * math.log(observed / expected)

    e11 = (k11 + k12) * (k11 + k21) / n
    e12 = (k11 + k12) * (k12 + k22) / n
    e21 = (k21 + k22) * (k11 + k21) / n
    e22 = (k21 + k22) * (k12 + k22) / n
    return 2.0 * (
        _safe_xlogx(k11, e11) +
        _safe_xlogx(k12, e12) +
        _safe_xlogx(k21, e21) +
        _safe_xlogx(k22, e22)
    )


class PhraseCooccurrenceIndex:
    """Phrase-level co-occurrence index supporting multiple distance windows.

    After calling ``index_stream`` once, call ``llr_scores`` and
    ``build_graph`` to retrieve association statistics at any of the
    configured window sizes.

    Example usage::

        phrases = ["te weten", "mitsdien", "op heden"]
        index = PhraseCooccurrenceIndex(phrases, windows=[100, 500, 2000])
        index.index_stream(token_stream, variant_index=vi)
        graph = index.build_graph(window=500, min_llr=10.0)

    Args:
        phrases: List of phrase strings (space-separated tokens) to track.
            Multi-word phrases are matched as contiguous token sequences.
        windows: List of token-distance thresholds.  A co-occurrence is
            counted at scale *w* when two phrase occurrences are ≤ *w*
            tokens apart.  Providing multiple scales lets downstream code
            distinguish local from global co-occurrence.
    """

    def __init__(self, phrases: List[str],
                 windows: List[int] = (100, 500, 2000)):
        self.phrases: Set[str] = set(phrases)
        self.windows: List[int] = list(windows)
        self.phrase_freq: Counter = Counter()
        self.phrase_positions: Dict[str, List[int]] = defaultdict(list)
        # cooc_freq[window][(phrase_a, phrase_b)] = count
        # phrase pair is stored in lexicographic order
        self.cooc_freq: Dict[int, Counter] = {w: Counter() for w in windows}
        self._total_tokens: int = 0

    def index_stream(self, token_stream: Iterable[str],
                     variant_index=None) -> None:
        """Scan *token_stream* in a single pass and populate the index.

        Uses a sliding deque so memory usage is bounded by
        ``max(windows) + max_phrase_len`` regardless of stream length.

        Args:
            token_stream: Iterable of token strings (normalised or raw).
                The stream may be arbitrarily long; it is consumed once.
            variant_index: Optional ``VariantIndex`` instance.  When
                provided, each token is canonicalised before matching so
                that orthographic variants are counted under the same phrase.
        """
        phrase_tuple_map: Dict[Tuple[str, ...], str] = {}
        tuple_lengths: Set[int] = set()
        for phrase in self.phrases:
            tup = tuple(phrase.split())
            phrase_tuple_map[tup] = phrase
            tuple_lengths.add(len(tup))
        if not tuple_lengths:
            return
        max_tuple_len = max(tuple_lengths)
        max_window = max(self.windows) if self.windows else 0

        token_buffer: deque = deque(maxlen=max_tuple_len)
        # recent_hits stores (token_offset, phrase_string)
        recent_hits: deque = deque()
        offset = 0

        def _canon(tok: str) -> str:
            return variant_index.canonical(tok) if variant_index is not None else tok

        for raw_token in token_stream:
            token = _canon(raw_token)
            token_buffer.append(token)

            if len(token_buffer) == max_tuple_len:
                for tup_len in tuple_lengths:
                    if tup_len > len(token_buffer):
                        continue
                    candidate = tuple(list(token_buffer)[-tup_len:])
                    if candidate not in phrase_tuple_map:
                        continue
                    phrase = phrase_tuple_map[candidate]
                    # The phrase ends at the current offset; start is offset - tup_len + 1
                    hit_offset = offset - tup_len + 1
                    self.phrase_freq[phrase] += 1
                    self.phrase_positions[phrase].append(hit_offset)

                    # Prune hits that fall outside even the largest window
                    while recent_hits and hit_offset - recent_hits[0][0] > max_window:
                        recent_hits.popleft()

                    for prev_offset, prev_phrase in recent_hits:
                        dist = hit_offset - prev_offset
                        pair = (min(prev_phrase, phrase), max(prev_phrase, phrase))
                        for window in self.windows:
                            if dist <= window:
                                self.cooc_freq[window][pair] += 1

                    recent_hits.append((hit_offset, phrase))

            offset += 1

        self._total_tokens = offset

    def llr_scores(self, window: int) -> Counter:
        """Return log-likelihood ratio scores for all phrase pairs at *window*.

        The LLR measures how much more (or less) often two phrases co-occur
        than expected under independence.  Higher values indicate stronger
        association.

        Args:
            window: One of the window sizes passed to ``__init__``.

        Returns:
            Counter mapping ``(phrase_a, phrase_b)`` tuples to LLR floats.
        """
        scores: Counter = Counter()
        n = self._total_tokens
        if n == 0 or window not in self.cooc_freq:
            return scores
        for (p1, p2), k11 in self.cooc_freq[window].items():
            f1 = self.phrase_freq[p1]
            f2 = self.phrase_freq[p2]
            k12 = max(f1 - k11, 0)
            k21 = max(f2 - k11, 0)
            k22 = max(n - k11 - k12 - k21, 0)
            score = _llr(k11, k12, k21, k22)
            if score > 0:
                scores[(p1, p2)] = score
        return scores

    def build_graph(self, window: int, min_llr: float = 10.0):
        """Build a weighted undirected networkx graph of phrase associations.

        Nodes are phrase strings; edge weights are LLR scores.  Isolated
        phrases (those with no co-occurrence above *min_llr* at this scale)
        are still included as nodes so that community detection sees the
        full vocabulary.

        Args:
            window: Distance scale to use for edge weights.
            min_llr: Minimum LLR score for an edge to be included.

        Returns:
            A ``networkx.Graph`` instance.
        """
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(self.phrases)
        for (p1, p2), score in self.llr_scores(window).items():
            if score >= min_llr:
                graph.add_edge(p1, p2, weight=score)
        return graph
