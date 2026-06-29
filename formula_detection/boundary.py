"""
boundary.py — Document boundary detection via formula community transitions
and via mined phrase motifs.

A BoundaryDetector scores each token position in an unbounded text stream
as a probable document boundary, from either (or both) of two independent
sources of evidence:

1. ``score_stream`` — scores are high where a phrase from a 'closing'
   formula community (one that tends to appear late in the inter-phrase
   distance histogram) is followed within a configurable window by a
   phrase from an 'opening' formula community. No supervision is required:
   opening/closing tendency is inferred from the mean position of each
   community's phrases in the PhraseCooccurrenceIndex.

2. ``score_from_motifs`` — scores come directly from the (start, end)
   offsets of ``formula_detection.motif.Motif`` instances, weighted by how
   reliably templated the motif is (low order entropy, high lift). This
   needs no community detection or stream rescan, since
   ``PhraseMotifIndex.mine_motifs()`` already computed the instance
   offsets from ``phrase_positions``.

Both producers emit scores on the same [0, 1]-ish scale and the same
``(token_offset, score)`` shape, so their outputs can be concatenated and
fed to a single ``detect_boundaries`` call.

Usage::

    detector = BoundaryDetector(phrase_clusters, community_map)
    detector.infer_community_order(cooc_index)
    scores = detector.score_stream(token_stream, variant_index=vi)
    boundaries = detector.detect_boundaries(scores)

    # or, directly from mined motifs, no community detection needed:
    scores = BoundaryDetector.score_from_motifs(canonical_motifs.values())
    boundaries = detector.detect_boundaries(scores)
"""
import math
from bisect import bisect_left
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple

from formula_detection.patterns.pattern import FormulaSignature, PhraseClusters

if TYPE_CHECKING:
    from formula_detection.motif import Motif


class BoundaryDetector:
    """Scores token positions as probable document boundaries.

    The detector operates in two stages:

    1. ``infer_community_order`` — called once after
       ``PhraseCooccurrenceIndex.index_stream``.  Assigns each community a
       position tendency score (0.0 = stream-beginning, 1.0 = stream-end)
       from the mean occurrence offsets stored in the index.

    2. ``score_stream`` — scans a new (or the same) token stream and
       returns a list of ``(token_offset, boundary_score)`` pairs.
       A high score at position *p* means a late-community phrase was
       observed just before *p* and an early-community phrase just after.

    Args:
        phrase_clusters: ``PhraseClusters`` mapping each phrase to a
            cluster ID (from ``formula_detection.clustering``).
        community_map: Mapping from phrase string to community ID,
            produced by networkx community detection on the phrase
            co-occurrence graph.
    """

    def __init__(self, phrase_clusters: PhraseClusters,
                 community_map: Dict[str, int]):
        self.phrase_clusters = phrase_clusters
        self.community_map = community_map
        # community_id -> position tendency [0, 1]; higher = later in stream
        self._community_order: Dict[int, float] = {}

    def infer_community_order(self, cooc_index) -> Dict[int, float]:
        """Compute position tendency for each community from *cooc_index*.

        Args:
            cooc_index: A ``PhraseCooccurrenceIndex`` instance after
                ``index_stream`` has been called.

        Returns:
            Dict mapping community ID to mean relative position [0, 1].
        """
        community_positions: Dict[int, List[int]] = defaultdict(list)
        for phrase, positions in cooc_index.phrase_positions.items():
            comm = self.community_map.get(phrase)
            if comm is not None:
                community_positions[comm].extend(positions)
        total = cooc_index._total_tokens or 1
        self._community_order = {
            comm: (sum(positions) / len(positions)) / total
            for comm, positions in community_positions.items()
            if positions
        }
        return self._community_order

    def score_stream(self, token_stream: Iterable[str],
                     variant_index=None,
                     window: int = 200) -> List[Tuple[int, float]]:
        """Scan *token_stream* and return per-position boundary scores.

        A boundary score is computed between every pair of phrase hits
        ``(prev_phrase, curr_phrase)`` that occur within *window* tokens of
        each other, where *prev_phrase* belongs to a later-tending community
        than *curr_phrase*.  The score equals the difference in their
        position tendency values; the boundary position is placed midway
        between the two hits.

        Call ``infer_community_order`` before this method.

        Args:
            token_stream: Iterable of token strings.  May be the same
                stream used to build the co-occurrence index or a new
                stream from the same corpus.
            variant_index: Optional ``VariantIndex`` for canonicalisation.
            window: Maximum token distance between a phrase pair for the
                pair to contribute a boundary score.

        Returns:
            List of ``(token_offset, score)`` tuples, unsorted.
        """
        phrase_tuple_map: Dict[tuple, str] = {}
        tuple_lengths: Set[int] = set()
        for phrase in self.community_map:
            tup = tuple(phrase.split())
            phrase_tuple_map[tup] = phrase
            tuple_lengths.add(len(tup))
        if not tuple_lengths:
            return []
        max_tuple_len = max(tuple_lengths)

        token_buffer: deque = deque(maxlen=max_tuple_len)
        # recent_hits stores (offset, phrase_string, position_tendency)
        recent_hits: deque = deque()
        scores: List[Tuple[int, float]] = []
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
                    comm = self.community_map.get(phrase)
                    if comm is None:
                        continue
                    curr_order = self._community_order.get(comm, 0.5)
                    hit_offset = offset - tup_len + 1

                    while recent_hits and hit_offset - recent_hits[0][0] > window:
                        recent_hits.popleft()

                    for prev_offset, prev_phrase, prev_order in recent_hits:
                        if prev_order > curr_order:
                            # Previous hit from a later community than current hit:
                            # probable boundary between them.
                            boundary_pos = (prev_offset + hit_offset) // 2
                            score = prev_order - curr_order
                            scores.append((boundary_pos, score))

                    recent_hits.append((hit_offset, phrase, curr_order))

            offset += 1

        return scores

    @staticmethod
    def score_from_motifs(motifs: Iterable['Motif'],
                          boundary: str = 'both') -> List[Tuple[int, float]]:
        """Score positions as boundary candidates directly from Motif instances.

        Each Motif instance's start and/or end offset becomes a candidate
        boundary position, weighted by a confidence score derived from the
        motif's own statistics rather than from community detection — a
        ``Motif`` already carries everything needed:

        - ``order_confidence`` = ``1 - order_entropy / log2(support)``: a
          motif whose members always appear in the same relative order
          scores near 1; one with highly variable internal order scores
          near 0. (A motif with a single instance trivially gets 1.0.)
        - ``lift_weight`` = ``lift / (lift + 1)``: a saturating transform so
          very large lift values (extremely unlikely under independence)
          approach 1 without unbounded scores, while lift just above the
          mining threshold contributes proportionally less.

        ``confidence = order_confidence * lift_weight`` is bounded in
        [0, 1] — the same rough scale ``score_stream`` uses — so the two
        can be combined into a single list and passed to one
        ``detect_boundaries`` call if you want evidence from both sources.

        Args:
            motifs: ``Motif`` objects to use as boundary evidence —
                typically the canonical/maximal motifs you've identified as
                document- or element-defining, not every mined subset
                (mining returns subsets and supersets of a real motif too,
                and including all of them would just duplicate the same
                boundary evidence many times over).
            boundary: Which edge(s) of each instance to emit: ``'start'``,
                ``'end'``, or ``'both'`` (default). Use ``'start'`` for
                opening-only candidates, ``'end'`` for closing-only,
                ``'both'`` to mark the full extent of each instance.

        Returns:
            List of ``(token_offset, score)`` pairs, unsorted, one or two
            entries per motif instance depending on *boundary*.
        """
        if boundary not in ('start', 'end', 'both'):
            raise ValueError("boundary must be 'start', 'end', or 'both'")
        scores: List[Tuple[int, float]] = []
        for motif in motifs:
            if motif.support == 0:
                continue
            if motif.support <= 1:
                order_confidence = 1.0
            else:
                max_entropy = math.log2(motif.support)
                order_confidence = 1.0 - min(motif.order_entropy / max_entropy, 1.0)
            lift_weight = 1.0 if math.isinf(motif.lift) else motif.lift / (motif.lift + 1.0)
            confidence = order_confidence * lift_weight
            for inst in motif.instances:
                if boundary in ('start', 'both'):
                    scores.append((inst.start, confidence))
                if boundary in ('end', 'both'):
                    scores.append((inst.end, confidence))
        return scores

    def detect_boundaries(self, scores: List[Tuple[int, float]],
                          min_gap: int = 100,
                          threshold: float = 0.3) -> List[int]:
        """Apply non-maximum suppression to the raw score stream.

        Selects boundary positions greedily from highest to lowest score,
        discarding candidates that are within *min_gap* tokens of an
        already-accepted boundary.

        Args:
            scores: Output of ``score_stream``.
            min_gap: Minimum token distance between two accepted boundaries.
                Set to roughly the minimum expected document length.
            threshold: Minimum score for a candidate to be considered.

        Returns:
            Sorted list of boundary token offsets.
        """
        candidates = [(pos, s) for pos, s in scores if s >= threshold]
        if not candidates:
            return []
        candidates.sort(key=lambda x: -x[1])
        # Keep accepted positions in a sorted list and use bisect to find the
        # nearest neighbours, rather than scanning every already-accepted
        # position for every candidate. With real corpora routinely producing
        # tens of thousands of candidate scores (e.g. many overlapping mined
        # motifs), the naive O(n*k) scan is too slow to be usable.
        selected: List[int] = []
        for pos, _score in candidates:
            idx = bisect_left(selected, pos)
            too_close = False
            if idx > 0 and pos - selected[idx - 1] < min_gap:
                too_close = True
            if idx < len(selected) and selected[idx] - pos < min_gap:
                too_close = True
            if not too_close:
                selected.insert(idx, pos)
        return selected

    def build_signatures(self) -> Dict[int, FormulaSignature]:
        """Construct a FormulaSignature for each community.

        Returns:
            Dict mapping community ID to ``FormulaSignature``.
        """
        community_phrases: Dict[int, List[str]] = defaultdict(list)
        for phrase, comm in self.community_map.items():
            community_phrases[comm].append(phrase)
        signatures = {}
        for comm, phrases in community_phrases.items():
            signatures[comm] = FormulaSignature(
                community_id=comm,
                phrases=phrases,
                position_tendency=self._community_order.get(comm),
            )
        return signatures
