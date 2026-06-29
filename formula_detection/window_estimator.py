"""
window_estimator.py — Data-driven window-size recommendation for phrase
co-occurrence and motif mining.

Rather than hand-tuning a single global ``window`` parameter, this module
estimates a natural window directly from the corpus: the gap between
successive occurrences of a phrase, or between nearest occurrences of two
phrases, has a "knee" in its cumulative distribution at the scale where
genuine within-unit co-occurrence gives way to rare, coincidental
long-range matches. Finding that knee gives a principled, automatic
window recommendation — per phrase, per phrase pair, or aggregated into
one global default.

Operates entirely on phrase_positions (as produced by
PhraseCooccurrenceIndex.phrase_positions) — no stream rescan needed.
"""
import math
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple


def _find_knee(sorted_values: List[float]) -> Optional[float]:
    """Find the knee of the empirical CDF of *sorted_values*.

    Uses the "maximum distance from the chord" (kneedle) heuristic:
    normalise the cumulative curve to [0, 1] on both axes and pick the
    point furthest from the straight line joining its endpoints.

    Args:
        sorted_values: Non-negative numbers, sorted ascending (e.g. gaps).

    Returns:
        The value at the knee, or None if fewer than 3 distinct values
        are available (no meaningful knee can be identified).
    """
    n = len(sorted_values)
    if n < 3:
        return None
    x0, x1 = sorted_values[0], sorted_values[-1]
    if x1 == x0:
        return None  # all gaps identical; no curve to find a knee in

    dx = x1 - x0
    norm = (dx ** 2 + 1.0) ** 0.5
    best_index, best_distance = None, -1.0
    for i, value in enumerate(sorted_values):
        x = (value - x0) / dx
        y = (i + 1) / n
        distance = abs(x - y) * dx / norm if norm > 0 else 0.0
        if distance > best_distance:
            best_distance = distance
            best_index = i
    return sorted_values[best_index] if best_index is not None else None


def _gaps(positions: List[int]) -> List[int]:
    return [b - a for a, b in zip(positions[:-1], positions[1:]) if b > a]


def _is_multimodal(gaps: List[int], n_bins: int = 20) -> bool:
    """Rough multimodality check via local maxima in a coarse histogram.

    Two or more local maxima suggest the phrase (or pair) participates in
    more than one recurrence cycle — e.g. a phrase shared between document
    types of different typical lengths, or a phrase whose "next occurrence
    distance" is dominated by how far away the next document of the *same
    type* happens to be, rather than a single consistent within-document span.
    """
    if len(gaps) < 10:
        return False
    lo, hi = min(gaps), max(gaps)
    if hi <= lo:
        return False
    bin_width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for g in gaps:
        idx = min(int((g - lo) / bin_width), n_bins - 1)
        counts[idx] += 1
    peaks = 0
    for i in range(n_bins):
        if counts[i] == 0:
            continue
        left = counts[i - 1] if i > 0 else -1
        right = counts[i + 1] if i < n_bins - 1 else -1
        if counts[i] > left and counts[i] > right:
            peaks += 1
    return peaks >= 2


class PhraseWindowStats:
    """Diagnostic statistics for a single phrase's self-gap distribution.

    Attributes:
        phrase: The phrase string.
        n_occurrences: Number of times the phrase occurs in the corpus.
        n_gaps: Number of successive-occurrence gaps (``n_occurrences - 1``).
        knee: Knee of the self-gap distribution, or None if there were too
            few occurrences to estimate one reliably.
        multimodal: True if the gap histogram shows 2+ local maxima,
            suggesting the phrase participates in more than one
            recurrence cycle (e.g. a hinge phrase, or one whose own
            "next occurrence" distance is dominated by how far away the
            next document of the same type happens to be).
        gaps: Sorted list of successive-occurrence gap values, for
            inspection or plotting (e.g. a box/strip plot of gap
            distributions across several candidate phrases).
        median_gap: Median of ``gaps``, or None if empty. The characteristic
            recurrence scale of this phrase — used by
            ``WindowEstimator.group_by_scale`` to separate phrases that
            operate at different levels of document structure (e.g. a
            phrase that recurs roughly once per day vs. one that recurs
            roughly once per resolution vs. one that recurs multiple times
            within a single resolution).
        mean_gap: Mean of ``gaps``, or None if empty.
    """

    def __init__(self, phrase: str, n_occurrences: int, n_gaps: int,
                 knee: Optional[float], multimodal: bool,
                 gaps: Optional[List[int]] = None):
        self.phrase = phrase
        self.n_occurrences = n_occurrences
        self.n_gaps = n_gaps
        self.knee = knee
        self.multimodal = multimodal
        self.gaps = gaps or []
        if self.gaps:
            sorted_gaps = sorted(self.gaps)
            mid = len(sorted_gaps) // 2
            self.median_gap = (sorted_gaps[mid] if len(sorted_gaps) % 2 == 1
                               else (sorted_gaps[mid - 1] + sorted_gaps[mid]) / 2)
            self.mean_gap = sum(sorted_gaps) / len(sorted_gaps)
        else:
            self.median_gap = None
            self.mean_gap = None

    def __repr__(self) -> str:
        return (f"PhraseWindowStats(phrase={self.phrase!r}, "
                f"n_occurrences={self.n_occurrences}, knee={self.knee}, "
                f"median_gap={self.median_gap}, multimodal={self.multimodal})")


class PairWindowStats:
    """Diagnostic statistics for the nearest-neighbour gap distribution
    between two phrases.

    Attributes:
        phrase_a: First phrase string.
        phrase_b: Second phrase string.
        n_pairs: Number of nearest-neighbour gap measurements.
        knee: Knee of the nearest-neighbour gap distribution, or None if
            too few measurements were available.
    """

    def __init__(self, phrase_a: str, phrase_b: str, n_pairs: int,
                 knee: Optional[float]):
        self.phrase_a = phrase_a
        self.phrase_b = phrase_b
        self.n_pairs = n_pairs
        self.knee = knee

    def __repr__(self) -> str:
        return (f"PairWindowStats({self.phrase_a!r}, {self.phrase_b!r}, "
                f"n_pairs={self.n_pairs}, knee={self.knee})")


class StreamClusterStats:
    """Diagnostics from single-link clustering of the combined candidate-
    phrase stream (see ``WindowEstimator.stream_cluster_stats``).

    Attributes:
        linkage_threshold: Gap size used to decide whether two consecutive
            candidate-phrase occurrences (of *any* phrase, merged into one
            stream) belong to the same cluster.
        n_clusters: Number of resulting clusters.
        spans: Sorted list of cluster spans (last position - first
            position within each cluster). Each cluster is a candidate
            document/element instance; its span is the distance covered
            by all the candidate phrases observed in that instance.
    """

    def __init__(self, linkage_threshold: float, spans: List[int]):
        self.linkage_threshold = linkage_threshold
        self.spans = sorted(spans)
        self.n_clusters = len(spans)

    def percentile(self, p: float) -> Optional[float]:
        """Return the *p*-th percentile (0–1) of cluster spans, or None if empty."""
        if not self.spans:
            return None
        idx = min(int(p * len(self.spans)), len(self.spans) - 1)
        return self.spans[idx]

    def __repr__(self) -> str:
        return (f"StreamClusterStats(linkage_threshold={self.linkage_threshold}, "
                f"n_clusters={self.n_clusters}, "
                f"span_median={self.percentile(0.5)}, span_p90={self.percentile(0.9)})")


class WindowEstimator:
    """Estimates natural window sizes from a corpus's phrase positions.

    Two complementary estimates are available:

    - ``phrase_stats`` / ``recommended_window`` — a phrase's own
      successive-occurrence gaps. Cheap, always available, and gives a
      phrase-dependent default: a phrase that recurs at short intervals
      gets a small recommended window, one that recurs rarely gets a
      large one. Caveat: this measures the gap to this phrase's *own*
      next occurrence, which in a stream of several interleaved
      document/element types is the distance to the next occurrence of
      the *same* type — not necessarily the within-unit span you actually
      want to bound. Use ``recommended_window``'s ``scale_factor`` to stay
      safely below it, and check ``multimodal`` before trusting it.
    - ``pair_stats`` — the nearest-neighbour gap distribution between two
      specific phrases. More expensive (one computation per pair) but
      directly measures the within-unit distance between two phrases that
      might co-occur, and serves as a good post-hoc audit of whatever
      window was used to mine a given motif.

    Args:
        phrase_positions: Dict mapping phrase string to a sorted list of
            token offsets, as produced by
            ``PhraseCooccurrenceIndex.phrase_positions``.
        min_occurrences: Minimum number of occurrences (hence
            ``min_occurrences - 1`` gaps) required before a phrase's
            self-gap knee is trusted. Phrases below this report ``knee=None``.
        scale_factor: Multiplier applied to a phrase's self-gap knee
            before it is returned by ``recommended_window``, so the result
            stays comfortably inside one recurrence cycle. Has no effect
            on the raw ``knee`` values in ``phrase_stats``/``pair_stats``.
    """

    def __init__(self, phrase_positions: Dict[str, List[int]],
                 min_occurrences: int = 6,
                 scale_factor: float = 0.7):
        self.phrase_positions = phrase_positions
        self.min_occurrences = min_occurrences
        self.scale_factor = scale_factor
        self._phrase_stats_cache: Dict[str, PhraseWindowStats] = {}

    def phrase_stats(self, phrase: str) -> PhraseWindowStats:
        """Self-gap knee and multimodality diagnostic for *phrase*."""
        if phrase in self._phrase_stats_cache:
            return self._phrase_stats_cache[phrase]
        positions = sorted(self.phrase_positions.get(phrase, []))
        gaps = sorted(_gaps(positions))
        knee = _find_knee(gaps) if len(positions) >= self.min_occurrences else None
        stats = PhraseWindowStats(
            phrase=phrase,
            n_occurrences=len(positions),
            n_gaps=len(gaps),
            knee=knee,
            multimodal=_is_multimodal(gaps),
            gaps=gaps,
        )
        self._phrase_stats_cache[phrase] = stats
        return stats

    def recommended_window(self, phrase: str,
                           default: Optional[float] = None) -> Optional[float]:
        """Recommended mining window for *phrase*, or *default* if not estimable."""
        stats = self.phrase_stats(phrase)
        if stats.knee is None:
            return default
        return stats.knee * self.scale_factor

    def pair_stats(self, phrase_a: str, phrase_b: str) -> PairWindowStats:
        """Nearest-neighbour gap knee between *phrase_a* and *phrase_b*.

        For every occurrence of *phrase_a*, finds the distance to the
        nearest occurrence of *phrase_b* (whichever side is closer), and
        finds the knee of the resulting gap distribution.
        """
        positions_a = sorted(self.phrase_positions.get(phrase_a, []))
        positions_b = sorted(self.phrase_positions.get(phrase_b, []))
        if not positions_a or not positions_b:
            return PairWindowStats(phrase_a, phrase_b, 0, None)

        gaps: List[int] = []
        for pos in positions_a:
            idx = bisect_right(positions_b, pos)
            candidates = []
            if idx < len(positions_b):
                candidates.append(abs(positions_b[idx] - pos))
            if idx > 0:
                candidates.append(abs(pos - positions_b[idx - 1]))
            if candidates:
                gaps.append(min(candidates))
        gaps.sort()
        knee = _find_knee(gaps) if len(gaps) >= 3 else None
        return PairWindowStats(phrase_a, phrase_b, len(gaps), knee)

    def recommended_global_window(self, phrases: Optional[List[str]] = None,
                                  default: float = 200.0) -> float:
        """Aggregate recommended window across multiple phrases' self-gaps.

        Takes the median of each phrase's ``recommended_window`` (skipping
        phrases with no estimable knee). Falls back to *default* if none
        are estimable.

        Note: in a corpus where several different document/element types
        are interleaved, a phrase's own self-gap measures the distance to
        the *next occurrence of the same phrase* — typically the next
        document of the *same type* — which can be much larger than the
        within-unit span you actually want to bound (and is exactly why
        ``phrase_stats`` flags these as ``multimodal``). Prefer
        ``cluster_based_window`` for a global estimate; this method is
        kept mainly to support genuinely phrase-dependent windowing.

        Args:
            phrases: Phrases to aggregate over. Defaults to all phrases in
                ``phrase_positions``.
            default: Fallback value when no phrase yields an estimate.
        """
        if phrases is None:
            phrases = list(self.phrase_positions.keys())
        estimates = sorted(
            w for w in (self.recommended_window(p) for p in phrases) if w is not None
        )
        if not estimates:
            return default
        mid = len(estimates) // 2
        if len(estimates) % 2 == 1:
            return estimates[mid]
        return (estimates[mid - 1] + estimates[mid]) / 2

    def stream_cluster_stats(self, phrases: Optional[List[str]] = None) -> StreamClusterStats:
        """Cluster the combined occurrence stream of *phrases* and report span stats.

        Merges every occurrence of every phrase in *phrases* into one
        sorted point process, finds the knee of its successive-gap
        distribution (the natural break between "still inside one
        document/element instance" and "crossed into the next one"), and
        uses that as a single-link clustering threshold: consecutive
        occurrences closer together than the threshold are grouped into
        the same cluster. Each resulting cluster approximates one
        document/element instance; its span is the distance from its
        first to its last occurrence.

        This sidesteps the same-phrase confound in ``recommended_window``:
        it doesn't matter which specific phrases happen to be close
        together, only that *some* candidate phrases cluster densely
        before a gap opens up to the next cluster.

        Args:
            phrases: Phrases to merge. Defaults to all phrases in
                ``phrase_positions``.

        Returns:
            A populated ``StreamClusterStats``.
        """
        if phrases is None:
            phrases = list(self.phrase_positions.keys())
        all_positions = sorted(
            pos for phrase in phrases for pos in self.phrase_positions.get(phrase, [])
        )
        if len(all_positions) < 3:
            return StreamClusterStats(linkage_threshold=0, spans=[])

        combined_gaps = sorted(_gaps(all_positions))
        linkage_threshold = _find_knee(combined_gaps)
        if linkage_threshold is None:
            return StreamClusterStats(linkage_threshold=0, spans=[])

        spans: List[int] = []
        cluster_start = cluster_end = all_positions[0]
        for prev, curr in zip(all_positions[:-1], all_positions[1:]):
            if curr - prev <= linkage_threshold:
                cluster_end = curr
            else:
                spans.append(cluster_end - cluster_start)
                cluster_start = cluster_end = curr
        spans.append(cluster_end - cluster_start)
        return StreamClusterStats(linkage_threshold=linkage_threshold, spans=spans)

    def cluster_based_window(self, phrases: Optional[List[str]] = None,
                             percentile: float = 0.9,
                             default: float = 200.0) -> float:
        """Recommended global window from combined-stream clustering.

        This is the recommended way to get an automatic *global* window:
        empirically, it is far more robust than aggregating per-phrase
        self-gaps (``recommended_global_window``) when several
        document/element types are interleaved in the same stream, since
        it measures actual instance spans directly rather than
        same-phrase recurrence distance.

        Args:
            phrases: Phrases to use. Defaults to all phrases in
                ``phrase_positions``.
            percentile: Which percentile (0–1) of the cluster span
                distribution to use. The default, 0.9, deliberately sits
                above the median so the window comfortably covers most
                instances rather than only the typical (shorter) ones.
            default: Fallback value when clustering yields no spans.
        """
        stats = self.stream_cluster_stats(phrases)
        value = stats.percentile(percentile)
        return value if value is not None else default

    def group_by_scale(self, phrases: Optional[List[str]] = None,
                       min_occurrences: int = 3,
                       log_gap_threshold: float = 0.5) -> "ScaleGroups":
        """Group phrases by the characteristic scale of their own recurrence.

        Many corpora are structured at several nested levels — e.g.
        resolutions grouped per day, each day preceded by a recurring
        "resumption" formula; each resolution typically containing one
        decision formula, but sometimes several, each followed by a
        repeated phrase. A phrase that operates at a higher level (e.g.
        the daily resumption) recurs at a much larger, fairly uniform
        token interval than a phrase that can repeat several times within
        a single resolution. This method uses each phrase's own median
        successive-occurrence gap (``PhraseWindowStats.median_gap``) as a
        1-D "scale" feature and groups phrases whose scales are close
        together on a *log* axis — log, because structural levels in a
        nested hierarchy typically differ by an order of magnitude or
        more in recurrence interval, not by a fixed additive amount.

        Grouping itself is simple single-linkage clustering on the sorted
        log10(median_gap) values: phrases are sorted by scale, and a new
        group starts wherever the gap between consecutive phrases' log
        scales exceeds *log_gap_threshold*. This means the number of
        groups is discovered automatically — there is no need to know in
        advance how many structural levels the corpus has.

        Args:
            phrases: Phrases to consider. Defaults to all phrases in
                ``phrase_positions``.
            min_occurrences: Minimum number of occurrences for a phrase to
                be included; phrases below this are reported in
                ``ScaleGroups.excluded`` rather than grouped, since a
                median over very few gaps is unreliable.
            log_gap_threshold: Minimum gap (in log10 units) between two
                phrases' scales for them to be split into separate groups.
                ``0.5`` (the default) means phrases more than a factor of
                ~3.2 apart in median recurrence interval end up in
                different groups; raise it towards 1.0 for coarser
                splitting (only separating levels an order of magnitude
                or more apart), or lower it for finer-grained splitting.

        Returns:
            A populated ``ScaleGroups``.
        """
        if phrases is None:
            phrases = list(self.phrase_positions.keys())

        scale_items: List[Tuple[str, float]] = []
        excluded: List[str] = []
        for phrase in phrases:
            stats = self.phrase_stats(phrase)
            if stats.n_occurrences < min_occurrences or not stats.median_gap:
                excluded.append(phrase)
                continue
            scale_items.append((phrase, math.log10(stats.median_gap)))
        scale_items.sort(key=lambda item: item[1])

        raw_groups: List[List[Tuple[str, float]]] = []
        for phrase, log_scale in scale_items:
            if raw_groups and log_scale - raw_groups[-1][-1][1] <= log_gap_threshold:
                raw_groups[-1].append((phrase, log_scale))
            else:
                raw_groups.append([(phrase, log_scale)])

        groups: List[ScaleGroup] = []
        for group_id, members in enumerate(raw_groups):
            member_phrases = [phrase for phrase, _ in members]
            log_scales = [log_scale for _, log_scale in members]
            mid = len(log_scales) // 2
            median_log = (log_scales[mid] if len(log_scales) % 2 == 1
                         else (log_scales[mid - 1] + log_scales[mid]) / 2)
            groups.append(ScaleGroup(group_id, member_phrases, 10 ** median_log))

        return ScaleGroups(groups, excluded)


class ScaleGroup:
    """A set of phrases that share a similar characteristic recurrence scale.

    Attributes:
        group_id: Integer ID (groups are ordered by increasing scale, so
            group 0 has the smallest typical recurrence interval).
        phrases: Phrase strings in this group.
        median_scale: Median of the member phrases' median gaps (in tokens).
    """

    def __init__(self, group_id: int, phrases: List[str], median_scale: float):
        self.group_id = group_id
        self.phrases = phrases
        self.median_scale = median_scale

    def __repr__(self) -> str:
        return (f"ScaleGroup(group_id={self.group_id}, "
                f"median_scale={self.median_scale:.1f}, "
                f"n_phrases={len(self.phrases)})")

    def __len__(self) -> int:
        return len(self.phrases)


class ScaleGroups:
    """Result of ``WindowEstimator.group_by_scale``.

    Attributes:
        groups: List of ``ScaleGroup``, ordered by increasing median_scale
            (group 0 = phrases with the shortest recurrence interval).
        excluded: Phrases that had too few occurrences to assign reliably.
    """

    def __init__(self, groups: List["ScaleGroup"], excluded: List[str]):
        self.groups = groups
        self.excluded = excluded

    def __repr__(self) -> str:
        scales = ", ".join(f"{g.median_scale:.0f}" for g in self.groups)
        return (f"ScaleGroups(n_groups={len(self.groups)}, "
                f"median_scales=[{scales}], n_excluded={len(self.excluded)})")

    def phrase_to_group(self) -> Dict[str, int]:
        """Return a flat Dict mapping each grouped phrase to its group_id."""
        return {phrase: group.group_id for group in self.groups for phrase in group.phrases}
