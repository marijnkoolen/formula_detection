"""
motif.py — Ordered, multi-phrase co-occurrence motifs and their relations.

Generalises the pairwise co-occurrence test in phrase_cooccurrence.py to
*sets* of 2–4 candidate phrases that recur together within a window, along
with the order they tend to appear in and the span of text they typically
cover. A tight span and low order-entropy are strong evidence of a
recurring structural unit (a document or a recurring document element);
its instances' start/end offsets are then direct boundary candidates —
more specific than single phrase-pair transitions because they require a
whole set to match.

Also provides classify_relation() to characterise how two motifs relate
positionally across the stream (sequential/adjacent, nested, overlapping,
or unrelated/disjoint), which helps decide whether two motifs are really
separate document elements, a hierarchical sub-structure, or two partial
views of the same element sharing a hinge phrase.

Operates entirely on the phrase_positions already collected by
PhraseCooccurrenceIndex — no need to rescan the token stream.
"""
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

from formula_detection.window_estimator import WindowEstimator


class MotifInstance:
    """A single occurrence of a motif (a set of co-occurring phrases).

    Attributes:
        start: Token offset of the earliest member in this occurrence.
        end: Token offset of the latest member in this occurrence.
        order: Tuple of phrase strings in the order they appeared.
    """

    __slots__ = ('start', 'end', 'order')

    def __init__(self, start: int, end: int, order: Tuple[str, ...]):
        self.start = start
        self.end = end
        self.order = order

    def __repr__(self) -> str:
        return f"MotifInstance(start={self.start}, end={self.end}, order={self.order})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, MotifInstance) and self.start == other.start
                and self.end == other.end and self.order == other.order)

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.order))


class Motif:
    """A statistically significant set of phrases that recur together.

    Attributes:
        phrase_set: The frozenset of phrase strings making up this motif.
        instances: All observed occurrences, sorted by start offset.
        support: Number of instances (after deduplication).
        lift: observed / expected co-occurrence count under independence.
        span_mean: Mean (end - start) across instances.
        span_std: Standard deviation of (end - start) across instances.
        order_entropy: Shannon entropy (bits) of the distribution over
            observed order tuples. 0.0 means every instance has the same
            internal order — a strict template. Higher values mean the
            members' relative order varies across instances.
        dominant_orders: Up to 3 most common order tuples with their
            relative frequency, sorted by frequency descending.
    """

    def __init__(self, phrase_set: FrozenSet[str], instances: List[MotifInstance], lift: float):
        self.phrase_set = phrase_set
        self.instances = sorted(instances, key=lambda inst: inst.start)
        self.support = len(self.instances)
        self.lift = lift

        spans = [inst.end - inst.start for inst in self.instances]
        self.span_mean = sum(spans) / len(spans) if spans else 0.0
        if len(spans) > 1:
            mean = self.span_mean
            variance = sum((s - mean) ** 2 for s in spans) / len(spans)
            self.span_std = math.sqrt(variance)
        else:
            self.span_std = 0.0

        order_counts = Counter(inst.order for inst in self.instances)
        total = sum(order_counts.values()) or 1
        self.order_entropy = -sum(
            (c / total) * math.log2(c / total) for c in order_counts.values()
        )
        self.dominant_orders: List[Tuple[Tuple[str, ...], float]] = [
            (order, count / total) for order, count in order_counts.most_common(3)
        ]

    def __repr__(self) -> str:
        return (f"Motif(phrases={sorted(self.phrase_set)}, support={self.support}, "
                f"lift={self.lift:.2f}, span_mean={self.span_mean:.1f}, "
                f"order_entropy={self.order_entropy:.2f})")

    def __len__(self) -> int:
        return len(self.phrase_set)


class PhraseMotifIndex:
    """Mines statistically significant ordered phrase motifs from positions.

    Builds anchor-forward windows over the merged, sorted occurrence stream
    of all candidate phrases: for each phrase occurrence (the "anchor"), it
    looks forward up to *window* tokens and records which other candidate
    phrases occur in that span. Combinations that include the anchor are
    counted as itemset occurrences — by construction the anchor is always
    the earliest member of any combination drawn from its own window, so
    each real-world cluster of co-occurring phrases is credited to exactly
    one combination per itemset, avoiding double counting.

    Args:
        phrase_positions: Dict mapping phrase string to a list of token
            offsets, as produced by ``PhraseCooccurrenceIndex.phrase_positions``.
        window: Maximum forward distance (in tokens) between the earliest
            and latest member of a motif occurrence. Either:

            - a number — a single global window, manually chosen, used
              for every anchor phrase;
            - ``'auto'`` (recommended) — a single global window derived
              from ``WindowEstimator.cluster_based_window``: the merged
              stream of all candidate phrases is clustered at its natural
              burst/gap threshold, and a high percentile of the resulting
              cluster spans is used. Robust even when several
              document/element types are interleaved in the stream;
            - ``'auto-phrase'`` — a genuinely phrase-dependent window: each
              anchor phrase gets its own window from
              ``WindowEstimator.recommended_window``, based on that
              phrase's own successive-occurrence gaps. Cheaper to compute
              per phrase, but biased upward (and flagged ``multimodal``)
              when a phrase's "next occurrence" distance is dominated by
              how far away the next document of the *same type* happens
              to be rather than the within-unit span — prefer ``'auto'``
              unless you have a specific reason to want per-phrase windows.

            For document/element-level motifs, manually-chosen windows are
            typically in the 100–2000 token range; ``'auto'`` finds the
            corresponding scale directly from the data.
        min_set_size: Minimum number of phrases in a motif (default 2).
        max_set_size: Maximum number of phrases in a motif (default 4).
            Kept small because the candidate phrase vocabulary is itself
            small; mining larger sets is rarely useful for formulaic
            document structure and grows combinatorially.
        min_support: Minimum number of instances for a motif to be kept.
        min_lift: Minimum ratio of observed to expected co-occurrence count
            (under an independence assumption on the single-phrase window
            frequencies) for a motif to be kept.
        window_estimator_kwargs: Extra keyword arguments forwarded to the
            ``WindowEstimator`` constructor when *window* is ``'auto'`` or
            ``'auto-phrase'`` (e.g. ``scale_factor``/``min_occurrences``,
            which affect ``'auto-phrase'``'s per-phrase self-gap windows).
        auto_window_percentile: Percentile (0–1) of the cluster-span
            distribution used by ``'auto'`` (passed to
            ``WindowEstimator.cluster_based_window``). Higher values
            produce a more generous window that covers more instances at
            the cost of being less tight; ignored for ``'auto-phrase'``
            and manual numeric windows.
    """

    def __init__(self, phrase_positions: Dict[str, List[int]],
                 window: Union[int, float, str] = 500,
                 min_set_size: int = 2,
                 max_set_size: int = 4,
                 min_support: int = 5,
                 min_lift: float = 2.0,
                 max_window_items: int = 12,
                 window_estimator_kwargs: Optional[dict] = None,
                 auto_window_percentile: float = 0.9):
        self.phrase_positions = phrase_positions
        self.window = window
        self.min_set_size = min_set_size
        self.max_set_size = max_set_size
        self.min_support = min_support
        self.min_lift = min_lift
        self.max_window_items = max_window_items
        self.window_estimator_kwargs = window_estimator_kwargs or {}
        self.auto_window_percentile = auto_window_percentile
        self.window_estimator: Optional[WindowEstimator] = None
        self._global_auto_window: Optional[float] = None

        if isinstance(window, str) and window not in ('auto', 'auto-phrase'):
            raise ValueError(f"window must be a number, 'auto', or 'auto-phrase', not {window!r}")

    def _window_for(self, phrase: str) -> float:
        """Resolve the forward window to use when anchoring on *phrase*."""
        if isinstance(self.window, (int, float)):
            return self.window
        if self.window_estimator is None:
            self.window_estimator = WindowEstimator(self.phrase_positions,
                                                     **self.window_estimator_kwargs)
        if self._global_auto_window is None:
            self._global_auto_window = self.window_estimator.cluster_based_window(
                phrases=list(self.phrase_positions.keys()),
                percentile=self.auto_window_percentile,
            )
        if self.window == 'auto':
            return self._global_auto_window
        # 'auto-phrase': fall back to the cluster-based global window for
        # any phrase whose own self-gap knee can't be estimated.
        return self.window_estimator.recommended_window(
            phrase, default=self._global_auto_window
        )

    def mine_motifs(self) -> List[Motif]:
        """Mine and return all motifs that pass the support/lift thresholds.

        Runs in two passes over the anchor-forward windows:

        1. Significance testing — counts, for every window, every itemset
           (size ``min_set_size``..``max_set_size``) that is a subset of
           that window's distinct phrases. Marginal (single-phrase) window
           counts are accumulated the same way, over the same set of
           windows, so the resulting lift ratio is computed from a single,
           consistent counting convention (each real-world cluster of
           co-occurring phrases is credited from multiple overlapping
           anchor windows in *both* numerator and denominator, which keeps
           the ratio valid even though it inflates raw counts).
        2. Instance extraction — for itemsets that pass the significance
           test only, re-scans to extract deduplicated, clean instances
           (one per real-world cluster) for span/order statistics. Here an
           instance is only recorded for the window anchored at the
           itemset's own earliest member, which removes the redundancy
           from step 1 for the surviving itemsets.

        Returns:
            List of ``Motif`` objects, unsorted.
        """
        events: List[Tuple[int, str]] = sorted(
            (pos, phrase)
            for phrase, positions in self.phrase_positions.items()
            for pos in positions
        )
        n_events = len(events)
        if n_events == 0:
            return []
        event_offsets = [pos for pos, _ in events]

        # --- Pass 1: consistent counting for the significance test ---
        itemset_counts: Counter = Counter()
        phrase_window_count: Counter = Counter()
        transactions: List[Dict[str, int]] = []
        n_transactions = n_events

        for i, (pos_i, phrase_i) in enumerate(events):
            # bisect rather than a monotonic two-pointer, since the window
            # can differ per anchor phrase (window='auto-phrase') and a
            # two-pointer that only ever advances would be wrong whenever
            # a later anchor's window is smaller than an earlier one's.
            j = bisect_right(event_offsets, pos_i + self._window_for(phrase_i), lo=i)
            window_events = events[i:j]

            item_first_pos: Dict[str, int] = {}
            for pos, phrase in window_events:
                if phrase not in item_first_pos:
                    item_first_pos[phrase] = pos
            transactions.append(item_first_pos)

            items = list(item_first_pos.keys())
            for phrase in items:
                phrase_window_count[phrase] += 1
            if len(items) > self.max_window_items:
                items = items[:self.max_window_items]
            for size in range(self.min_set_size, min(self.max_set_size, len(items)) + 1):
                for combo in combinations(sorted(items), size):
                    itemset_counts[frozenset(combo)] += 1

        candidate_itemsets: List[Tuple[FrozenSet[str], float]] = []
        for itemset, count in itemset_counts.items():
            if count < self.min_support:
                continue
            expected = n_transactions
            for phrase in itemset:
                expected *= phrase_window_count[phrase] / n_transactions
            lift = count / expected if expected > 0 else float('inf')
            if lift >= self.min_lift:
                candidate_itemsets.append((itemset, lift))

        if not candidate_itemsets:
            return []

        # --- Pass 2: deduplicated instance extraction for survivors only ---
        candidate_set = {itemset for itemset, _ in candidate_itemsets}
        itemset_instances: Dict[FrozenSet[str], set] = defaultdict(set)

        for i, (pos_i, phrase_i) in enumerate(events):
            item_first_pos = transactions[i]
            other_phrases = sorted(p for p in item_first_pos if p != phrase_i)
            for size in range(self.min_set_size, self.max_set_size + 1):
                k = size - 1
                if k < 0 or k > len(other_phrases):
                    continue
                for combo_rest in combinations(other_phrases, k):
                    combo = tuple(sorted((phrase_i,) + combo_rest))
                    itemset = frozenset(combo)
                    if itemset not in candidate_set:
                        continue
                    positions = {p: item_first_pos[p] for p in combo}
                    start = positions[phrase_i]  # guaranteed minimum by construction
                    end = max(positions.values())
                    order = tuple(sorted(combo, key=lambda p: positions[p]))
                    itemset_instances[itemset].add(MotifInstance(start, end, order))

        motifs: List[Motif] = []
        for itemset, lift in candidate_itemsets:
            instances = list(itemset_instances.get(itemset, ()))
            if len(instances) < self.min_support:
                continue
            motifs.append(Motif(itemset, instances, lift))
        return motifs


class MotifRelation:
    """Summarises how two motifs relate positionally across the stream.

    Attributes:
        phrases_a: Phrase set of the first motif.
        phrases_b: Phrase set of the second motif.
        shared_phrases: Phrases common to both motifs (a "hinge" if non-empty).
        relation_counts: Counter over relation labels (see classify_relation)
            across all considered instance pairs.
        dominant_relation: The most frequent relation label.
        dominant_fraction: Fraction of considered pairs with the dominant label.
        n_pairs: Total number of instance pairs considered.
    """

    def __init__(self, phrases_a: FrozenSet[str], phrases_b: FrozenSet[str],
                 relation_counts: Counter):
        self.phrases_a = phrases_a
        self.phrases_b = phrases_b
        self.shared_phrases = phrases_a & phrases_b
        self.relation_counts = relation_counts
        self.n_pairs = sum(relation_counts.values())
        if self.n_pairs > 0:
            self.dominant_relation, dominant_count = relation_counts.most_common(1)[0]
            self.dominant_fraction = dominant_count / self.n_pairs
        else:
            self.dominant_relation = None
            self.dominant_fraction = 0.0

    def __repr__(self) -> str:
        return (f"MotifRelation(shared={sorted(self.shared_phrases)}, "
                f"dominant={self.dominant_relation!r} "
                f"({self.dominant_fraction:.0%} of {self.n_pairs}))")


def _nearest_index(sorted_values: List[int], value: int) -> Optional[int]:
    """Return the index in sorted_values closest to value, or None if empty."""
    if not sorted_values:
        return None
    idx = bisect_left(sorted_values, value)
    if idx == 0:
        return 0
    if idx == len(sorted_values):
        return len(sorted_values) - 1
    before, after = sorted_values[idx - 1], sorted_values[idx]
    return idx - 1 if (value - before) <= (after - value) else idx


def classify_relation(motif_a: Motif, motif_b: Motif, max_gap: int = 200) -> MotifRelation:
    """Classify the positional relationship between two motifs.

    For each instance of *motif_a*, finds the nearest instance of *motif_b*
    by start offset and classifies the pair into one of:

    - ``'a_before_b'`` — a ends, then b starts within max_gap (sequential;
      strong boundary evidence between two adjacent document/element instances).
    - ``'b_before_a'`` — symmetric case.
    - ``'a_nested_in_b'`` / ``'b_nested_in_a'`` — one span fully contains
      the other (hierarchical / sub-element structure).
    - ``'overlapping_partial'`` — spans overlap but neither contains the other.
    - ``'disjoint_far'`` — nearest instance is further than max_gap away in
      both directions (the two motifs are not positionally related here).

    The dominant relation across all instance pairs (see ``MotifRelation``)
    indicates the structural relationship between the two motifs: mostly
    ``a_before_b``/``b_before_a`` suggests sequential document/element
    boundaries; mostly nested suggests one is a sub-element of the other;
    mostly overlapping (especially combined with shared phrases) suggests
    they are two partial views of the same structural unit rather than
    genuinely distinct elements.

    Args:
        motif_a: First motif.
        motif_b: Second motif.
        max_gap: Maximum token distance between the end of one instance and
            the start of another for them to be considered adjacent rather
            than disjoint.

    Returns:
        A populated ``MotifRelation``.
    """
    b_starts = [inst.start for inst in motif_b.instances]
    relation_counts: Counter = Counter()

    for a_inst in motif_a.instances:
        idx = _nearest_index(b_starts, a_inst.start)
        if idx is None:
            continue
        b_inst = motif_b.instances[idx]

        if b_inst.start >= a_inst.start and b_inst.end <= a_inst.end:
            relation = 'b_nested_in_a'
        elif a_inst.start >= b_inst.start and a_inst.end <= b_inst.end:
            relation = 'a_nested_in_b'
        elif b_inst.start < a_inst.end and b_inst.end > a_inst.start:
            relation = 'overlapping_partial'
        elif b_inst.start >= a_inst.end and (b_inst.start - a_inst.end) <= max_gap:
            relation = 'a_before_b'
        elif a_inst.start >= b_inst.end and (a_inst.start - b_inst.end) <= max_gap:
            relation = 'b_before_a'
        else:
            relation = 'disjoint_far'
        relation_counts[relation] += 1

    return MotifRelation(motif_a.phrase_set, motif_b.phrase_set, relation_counts)


def merge_overlapping_phrases(phrase_positions: Dict[str, List[int]],
                              window: int = 10,
                              min_support: int = 5,
                              min_lift: float = 3.0) -> Dict[str, str]:
    """Group candidate phrases that are slices of the same formula, and
    collapse each group onto one canonical representative.

    Overlapping sliding-window n-gram candidates of the same underlying
    formula (e.g. "waar op gedelibereert zynde is" and "op gedelibereert
    zynde is goedgevonden") co-occur at very short, near-constant distance
    almost every time they occur at all. This causes two problems if such
    phrases are fed directly into ``PhraseMotifIndex`` at a document/
    element-level window:

    1. Redundant motifs — the same real formula gets rediscovered many
       times over as different combinations of its own overlapping
       slices, inflating the motif count without adding information
       (this is what drove the 0 -> 1472 -> hangs-on-NMS progression seen
       when mining motifs directly over undeduplicated n-gram candidates).
    2. Cross-occurrence aliasing — if a formula recurs within the window
       used for document/element-level motif mining, slice A2 from one
       occurrence and slice A1 from a *different* occurrence can also
       fall within that window of each other, producing a spurious motif
       that looks like A1 and A2 co-occur as distinct entities, when
       really they are just two slices of one formula seen twice.

    This function detects same-formula slices via small-window pairwise
    co-occurrence (computed by running ``PhraseMotifIndex`` itself with
    ``max_set_size=2``, so no separate co-occurrence machinery is needed),
    then groups them with the same Louvain community detection used for
    document-type signatures, just applied at a much smaller scale: at
    this window, two overlapping n-gram slices of one formula should
    co-occur almost every time either occurs, while genuinely different
    formulas should not.

    Args:
        phrase_positions: Dict mapping phrase string to sorted token offsets.
        window: Distance scale for "these are slices of the same formula".
            Should be small — comparable to how much adjacent sliding-
            window n-grams of one sentence overlap — not the larger
            document/element-level window used for the actual motif
            mining pass that follows this merge step.
        min_support: Minimum number of co-occurrences for a phrase pair to
            be linked in the merge graph.
        min_lift: Minimum lift for a phrase pair to be linked. Kept high
            by default since this pass should only catch very reliable,
            near-always co-occurring pairs, not genuine but looser
            cross-formula associations.

    Returns:
        Dict mapping every phrase in *phrase_positions* to its canonical
        representative (the most frequent phrase in its community; a
        phrase with no qualifying co-occurrences maps to itself).
    """
    from formula_detection.phrase_cooccurrence import _llr, detect_communities
    import networkx as nx

    # Pairwise co-occurrence is computed directly here (a simple anchor-
    # forward sweep over the merged event stream, with LLR scoring exactly
    # like PhraseCooccurrenceIndex) rather than via PhraseMotifIndex: the
    # merge step only ever needs pairs, and keeping it self-contained avoids
    # coupling to PhraseMotifIndex's own (separately evolving) windowing
    # behaviour for multi-phrase motifs.
    events: List[Tuple[int, str]] = sorted(
        (pos, phrase)
        for phrase, positions in phrase_positions.items()
        for pos in positions
    )
    n_events = len(events)
    event_offsets = [pos for pos, _ in events]
    pair_counts: Counter = Counter()
    phrase_counts: Counter = Counter()

    for i, (pos_i, phrase_i) in enumerate(events):
        phrase_counts[phrase_i] += 1
        j = bisect_right(event_offsets, pos_i + window, lo=i + 1)
        seen_in_window: set = set()
        for _pos, phrase in events[i + 1:j]:
            if phrase == phrase_i or phrase in seen_in_window:
                continue
            seen_in_window.add(phrase)
            pair_counts[tuple(sorted((phrase_i, phrase)))] += 1

    graph = nx.Graph()
    graph.add_nodes_from(phrase_positions.keys())
    for (a, b), k11 in pair_counts.items():
        if k11 < min_support or n_events == 0:
            continue
        fa, fb = phrase_counts[a], phrase_counts[b]
        expected = n_events * (fa / n_events) * (fb / n_events)
        lift = k11 / expected if expected > 0 else float('inf')
        if lift < min_lift:
            continue
        k12 = max(fa - k11, 0)
        k21 = max(fb - k11, 0)
        k22 = max(n_events - k11 - k12 - k21, 0)
        graph.add_edge(a, b, weight=_llr(k11, k12, k21, k22))

    community_map = detect_communities(graph, weight='weight')

    phrase_freq = {p: len(positions) for p, positions in phrase_positions.items()}
    community_members: Dict[int, List[str]] = defaultdict(list)
    for phrase, comm in community_map.items():
        community_members[comm].append(phrase)

    phrase_to_canonical: Dict[str, str] = {}
    for members in community_members.values():
        canonical = max(members, key=lambda p: phrase_freq.get(p, 0))
        for member in members:
            phrase_to_canonical[member] = canonical
    for phrase in phrase_positions:
        phrase_to_canonical.setdefault(phrase, phrase)
    return phrase_to_canonical


def merge_phrase_positions(phrase_positions: Dict[str, List[int]],
                           phrase_to_canonical: Dict[str, str],
                           dedup_distance: int = 3) -> Dict[str, List[int]]:
    """Collapse phrase_positions onto canonical phrases and deduplicate.

    Pairs naturally with ``merge_overlapping_phrases``: occurrences from
    every phrase in a community are pooled under that community's
    canonical phrase, then near-duplicate occurrences (within
    *dedup_distance* tokens of each other — typically the same real
    occurrence, seen through different overlapping n-gram slices) are
    collapsed to a single position.

    Args:
        phrase_positions: Dict mapping phrase string to sorted token offsets.
        phrase_to_canonical: Mapping from ``merge_overlapping_phrases``.
        dedup_distance: Maximum gap between two pooled occurrences for
            them to be treated as the same real occurrence.

    Returns:
        Dict mapping each canonical phrase to a deduplicated, sorted list
        of merged occurrence offsets.
    """
    grouped: Dict[str, List[int]] = defaultdict(list)
    for phrase, positions in phrase_positions.items():
        canonical = phrase_to_canonical.get(phrase, phrase)
        grouped[canonical].extend(positions)

    merged: Dict[str, List[int]] = {}
    for canonical, positions in grouped.items():
        sorted_positions = sorted(set(positions))
        deduped: List[int] = []
        for pos in sorted_positions:
            if deduped and pos - deduped[-1] <= dedup_distance:
                continue
            deduped.append(pos)
        merged[canonical] = deduped
    return merged
