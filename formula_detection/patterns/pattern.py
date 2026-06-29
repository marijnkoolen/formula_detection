"""
pattern.py — Core data structures for representing and matching sequence
and set-based patterns of term labels (e.g. tokens or term IDs).

This module provides two families of pattern representation:

- Pattern / PatternIndex: ordered sequences of term labels matched
  positionally against the start and end of a window in a document.
- PhrasePattern / PhrasePatternCounter: order-insensitive (multiset)
  patterns of term IDs, used to detect and count phrase occurrences
  regardless of word order, including support for a `<VAR>` wildcard
  term for unrecognised vocabulary items.

PhraseClusters and FormulaSignature (defined later in this module)
build on top of these to represent clustering and community-detection
results over formulaic phrases.
"""
from __future__ import annotations
from collections import defaultdict
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Union

from fuzzy_search.tokenization.token import Doc
from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.vocabulary import Vocabulary


class Pattern:
    """An ordered sequence of term labels representing a fixed pattern.

    A Pattern is a tuple of labels (e.g. normalised token strings) that
    is expected to be matched against a contiguous window of tokens in a
    document, comparing the first and last labels against the window's
    start and end.

    Attributes:
        labels: Tuple of term labels making up the pattern, in order.
        start: The first label in the pattern.
        end: The last label in the pattern.
    """

    def __init__(self, labels: List[str]):
        """Initialise a Pattern from an ordered list of labels.

        Args:
            labels: Ordered list of term labels making up the pattern.
                Must contain at least one element.
        """
        self.labels = tuple(labels)
        self.start = labels[0]
        self.end = labels[-1]

    def __contains__(self, item):
        return item in self.labels

    def __len__(self):
        return len(self.labels)

    @property
    def length(self):
        return len(self)


class PatternIndex:
    """An index of Pattern objects keyed by their start and end labels.

    Maintains a set of all indexed patterns plus two lookup indexes
    (by start label and by end label) to support efficient lookup of
    candidate patterns when scanning a document.

    Attributes:
        patterns: Set of all Pattern objects currently indexed.
        start_index: Dict mapping a start label to the set of patterns
            that begin with that label.
        end_index: Dict mapping an end label to the set of patterns
            that end with that label.
    """

    def __init__(self, patterns: Union[Pattern, List[Pattern]]):
        """Initialise the index with one or more patterns.

        Args:
            patterns: A single Pattern or a list of Pattern objects to
                index. A single Pattern is wrapped into a one-item list.
        """
        if isinstance(patterns, Pattern):
            patterns = [patterns]
        self.patterns = set()
        self.start_index = defaultdict(set)
        self.end_index = defaultdict(set)
        self.index_patterns(patterns)

    def __contains__(self, item: Pattern):
        return item in self.patterns

    def __len__(self):
        return len(self.patterns)

    def index_patterns(self, patterns: List[Pattern]):
        """Add new patterns to the index.

        Patterns already present in self.patterns are skipped. Newly
        added patterns are registered in both start_index and end_index.

        Args:
            patterns: List of Pattern objects to add to the index.
        """
        for pattern in patterns:
            if pattern not in self.patterns:
                self.patterns.add(pattern)
                self.start_index[pattern.start].add(pattern)
                self.end_index[pattern.end].add(pattern)

    def find_pattern_in_doc(self, doc: Doc) -> bool:
        """Search doc for occurrences of any indexed pattern.

        For each token in doc whose normalised form is a registered
        start label, this checks whether the token located length(pattern)
        - 1 positions later in doc equals the pattern's end label, and if
        so records the intervening span as a match.

        Args:
            doc: A fuzzy_search Doc (sequence of tokens) to search.

        Returns:
            True if at least one indexed pattern matched in doc, False
            otherwise.
        """
        matches = []
        for token in doc:
            if token.n in self.start_index:
                for pattern in self.start_index[token.n]:
                    end_token = pattern.end
                    end_index = token.i + len(pattern) - 1
                    if end_index < len(doc) and doc[end_index].n == end_token:
                        match = doc[token.i:end_index]
                        matches.append(match)
        return bool(matches)


class PhrasePattern:
    """An order-insensitive (multiset) pattern of term IDs.

    Represents the set of term IDs making up a phrase, while retaining
    enough information (the original ordered tuple and per-term counts)
    to support multiset-aware overlap computations between two phrases
    that share the same vocabulary of term IDs.

    Attributes:
        term_ids: Tuple of term IDs in their original phrase order
            (duplicates retained).
        id_set: Set of the unique term IDs in term_ids.
        sorted: Tuple of the unique term IDs in id_set, sorted ascending.
            Used as a canonical key for grouping equivalent term-ID sets.
    """

    def __init__(self, term_ids: Iterable[int]):
        """Initialise a PhrasePattern from an iterable of term IDs.

        Args:
            term_ids: Iterable of term IDs, in phrase order. Duplicate
                term IDs are allowed and preserved in term_ids.
        """
        self.term_ids = tuple(term_ids)
        self.id_set = set(term_ids)
        self.sorted = tuple(sorted(self.id_set))

    def __repr__(self):
        return f"{self.__class__.__name__}(term_ids={self.term_ids}, sorted={self.sorted})"

    def __eq__(self, other: PhrasePattern):
        return self.id_set == other.id_set

    def __contains__(self, item):
        if isinstance(item, PhrasePattern):
            return all(term_id in self.id_set for term_id in item.id_set)
        elif isinstance(item, int):
            return item in self.id_set

    def set_overlap(self, other: PhrasePattern):
        """Return the set of term IDs shared between this and other.

        This is a set intersection: each shared term ID appears at most
        once in the result, regardless of how many times it occurs in
        either phrase's term_ids.

        Args:
            other: The other PhrasePattern to compare against.

        Returns:
            A set of term IDs present in both self.id_set and
            other.id_set.
        """
        return self.id_set.intersection(other.id_set)

    def term_overlap(self, other: PhrasePattern):
        """Return the multiset overlap of term IDs between this and other.

        Unlike set_overlap, this accounts for repeated term IDs: for each
        term ID present in both phrases, the number of times it can
        contribute to the overlap is the minimum of its occurrence count
        in self.term_ids and in other.term_ids. The result preserves the
        relative order of self.term_ids, including up to that per-term
        count of repeats for each overlapping term ID.

        Args:
            other: The other PhrasePattern to compare against.

        Returns:
            A tuple of term IDs (with repeats, in self's original order)
            representing the multiset intersection of self.term_ids and
            other.term_ids.
        """
        set_overlap = self.id_set.intersection(other.id_set)
        term_overlap = []
        term_overlap_freq = {}
        for term_id in set_overlap:
            count = min(self.term_ids.count(term_id), other.term_ids.count(term_id))
            # term_overlap.extend([term_id] * count)
            term_overlap_freq[term_id] = count
        for term_id in self.term_ids:
            if term_id in term_overlap_freq and term_overlap_freq[term_id] > 0:
                term_overlap.append(term_id)
                term_overlap_freq[term_id] -= 1
        return tuple(term_overlap)


def pattern_to_id_set_tuple(pattern: Union[PhrasePattern, Iterable[int]]):
    """Normalise a pattern-like value into a sorted tuple of unique term IDs.

    Args:
        pattern: Either a PhrasePattern (whose .sorted attribute is used
            directly) or any iterable of term IDs.

    Returns:
        A tuple of the unique term IDs, sorted ascending.
    """
    if isinstance(pattern, PhrasePattern):
        return pattern.sorted
    return tuple(sorted(pattern))


class PhrasePatternCounter:
    """Tracks observed PhrasePattern occurrences grouped by term-ID set.

    Phrases are grouped by their sorted set of unique term IDs
    (pattern.sorted), and within each group, a Counter tracks how many
    times each distinct ordered term_ids tuple (i.e. distinct word order
    / multiset of repeats) has been observed.

    Attributes:
        vocabulary: The Vocabulary used to convert phrases (lists of
            strings/Tokens) into term IDs.
        set2tuple: Dict mapping a sorted term-ID-set tuple to a Counter of
            observed term_ids tuples (with their occurrence counts).
    """

    def __init__(self, vocabulary: Vocabulary, phrase_patterns: List[PhrasePattern] = None):
        """Initialise the counter, optionally seeding it with patterns.

        Args:
            vocabulary: The Vocabulary used to map phrase tokens to term
                IDs (via add_phrase).
            phrase_patterns: Optional list of PhrasePattern objects to add
                to the counter immediately.
        """
        self.vocabulary = vocabulary
        self.set2tuple = defaultdict(Counter)
        if phrase_patterns is not None:
            for pp in phrase_patterns:
                self.add_pattern(pp)

    def __contains__(self, item):
        if isinstance(item, PhrasePattern):
            return item.sorted in self.set2tuple
        else:
            sorted_set_tuple = pattern_to_id_set_tuple(item)
            return sorted_set_tuple in self.set2tuple

    def add_pattern(self, pattern: PhrasePattern):
        """Record an occurrence of a PhrasePattern.

        Increments the count for pattern.term_ids within the group keyed
        by pattern.sorted.

        Args:
            pattern: The PhrasePattern occurrence to record.
        """
        self.set2tuple[pattern.sorted].update([pattern.term_ids])

    def add_phrase(self, phrase: List[Union[str, Token]]):
        """Convert a phrase to a PhrasePattern and record its occurrence.

        Args:
            phrase: List of strings or Tokens making up the phrase, in
                order. Converted to term IDs via self.vocabulary using
                phrase_to_term_ids.
        """
        term_ids = phrase_to_term_ids(self.vocabulary, phrase)
        self.add_pattern(PhrasePattern(term_ids))

    def _has_pattern_set(self, id_set: Set[int]):
        sorted_set_tuple = pattern_to_id_set_tuple(id_set)
        return sorted_set_tuple in self.set2tuple

    def has_pattern(self, phrase_pattern: Iterable[int]):
        """Check whether an exact ordered term-ID pattern has been recorded.

        Args:
            phrase_pattern: An iterable of term IDs (in a specific order)
                to look up.

        Returns:
            True if the exact tuple(phrase_pattern) has been recorded
            (i.e. is present, with a positive count, in the Counter for
            its term-ID-set group); False if the term-ID set itself was
            never seen, or was seen but not with this exact ordered tuple.
        """
        sorted_set_tuple = pattern_to_id_set_tuple(phrase_pattern)
        if sorted_set_tuple not in self.set2tuple:
            return False
        pattern_tuple = tuple(phrase_pattern)
        return pattern_tuple in self.set2tuple[sorted_set_tuple]

    def get_id_set_patterns(self, id_set: Set[int]):
        """Return all distinct ordered patterns recorded for a term-ID set.

        Args:
            id_set: A set (or other iterable) of term IDs identifying the
                group to look up.

        Returns:
            A list of Pattern objects, one per distinct ordered term_ids
            tuple recorded for this term-ID set, or an empty list if the
            term-ID set has never been recorded.
        """
        sorted_set_tuple = pattern_to_id_set_tuple(id_set)
        if sorted_set_tuple not in self.set2tuple:
            return []
        return [Pattern(term_ids) for term_ids in self.set2tuple[sorted_set_tuple]]


def tokens_match_pattern(tokens: List[Token], pattern: Pattern):
    """Check whether a list of tokens matches a Pattern position-by-position.

    Args:
        tokens: List of Token objects to compare against pattern.labels.
        pattern: The Pattern whose labels are compared against tokens.

    Returns:
        True if tokens and pattern have equal length and each token's
        normalised form (token.n) equals the corresponding pattern label;
        False otherwise (including when lengths differ, in which case a
        diagnostic message and the mismatched tokens/pattern are printed).
    """
    if len(tokens) != len(pattern):
        print('tokens_match_pattern - unequal length')
        print(tokens)
        print(pattern.labels, len(pattern))
        return False
    return all([token.n == label for token, label in zip(tokens, pattern.labels)])


def find_pattern_in_doc(doc: Doc, pattern: Pattern) -> List[List[Token]]:
    """Find all contiguous windows in doc that match pattern.

    Scans doc for tokens whose normalised form matches pattern.start, then
    checks whether the following len(pattern) tokens (including that
    token) match pattern exactly via tokens_match_pattern. Diagnostic
    information is printed for each candidate start token found.

    Args:
        doc: A fuzzy_search Doc (sequence of tokens) to search.
        pattern: The Pattern to match against windows of doc.

    Returns:
        A list of token-list matches, one per window in doc that matches
        pattern.
    """
    matches = []
    for token in doc:
        if token.n == pattern.start:
            print(f"{token.n} matches start of pattern {pattern.labels}")
            tokens = doc[token.i:token.i+len(pattern)]
            print('tokens:', tokens)
            if tokens_match_pattern(tokens, pattern):
                matches.append(tokens)
    return matches


def pattern_in_doc(doc: Doc, pattern: Pattern) -> bool:
    """Check whether pattern occurs anywhere in doc.

    Args:
        doc: A fuzzy_search Doc (sequence of tokens) to search.
        pattern: The Pattern to look for.

    Returns:
        True if find_pattern_in_doc(doc, pattern) returns at least one
        match, False otherwise.
    """
    matches = find_pattern_in_doc(doc, pattern)
    return len(matches) > 0


def phrase_to_term_ids(vocabulary: Vocabulary, tokens: List[Union[str, Token]]):
    """Convert a list of tokens/strings to a list of vocabulary term IDs.

    Tokens not found in vocabulary are mapped to the vocabulary's `<VAR>`
    term ID if one is registered, or to -1 otherwise.

    Args:
        vocabulary: The Vocabulary used to look up term IDs.
        tokens: List of strings or Token objects making up the phrase.

    Returns:
        A list of term IDs (ints), one per element of tokens, in the same
        order.
    """
    term_ids = []
    for token in tokens:
        term_id = vocabulary.term2id(token)
        if term_id is None:
            term_id = vocabulary.term_id['<VAR>'] if '<VAR>' in vocabulary.term_id else -1
        term_ids.append(term_id)
    return term_ids


def phrase_to_phrase_pattern(vocabulary: Vocabulary, tokens: List[Union[str, Token]]):
    """Convert a list of tokens/strings directly into a PhrasePattern.

    Args:
        vocabulary: The Vocabulary used to look up term IDs (see
            phrase_to_term_ids).
        tokens: List of strings or Token objects making up the phrase.

    Returns:
        A PhrasePattern built from the term IDs of tokens.
    """
    term_ids = phrase_to_term_ids(vocabulary=vocabulary, tokens=tokens)
    return PhrasePattern(term_ids)


class PhraseClusters:
    """Stores the result of clustering candidate phrases.

    Each phrase is assigned to a cluster ID. The mapping is bidirectional.
    Cluster IDs are arbitrary integers assigned by the clustering algorithm.
    """

    def __init__(self, phrase_to_cluster: Dict[str, int]):
        self.phrase_to_cluster: Dict[str, int] = phrase_to_cluster
        self.cluster_to_phrases: Dict[int, List[str]] = defaultdict(list)
        for phrase, cluster_id in phrase_to_cluster.items():
            self.cluster_to_phrases[cluster_id].append(phrase)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"n_clusters={len(self.cluster_to_phrases)}, "
                f"n_phrases={len(self.phrase_to_cluster)})")

    def __len__(self) -> int:
        return len(self.cluster_to_phrases)

    def cluster_of(self, phrase: str) -> Optional[int]:
        """Return the cluster ID for a phrase, or None if not found."""
        return self.phrase_to_cluster.get(phrase)

    def phrases_in(self, cluster_id: int) -> List[str]:
        """Return all phrases assigned to cluster_id."""
        return list(self.cluster_to_phrases.get(cluster_id, []))


class FormulaSignature:
    """Represents a community of formulaic phrases that tend to co-occur.

    A FormulaSignature is produced by community detection on the phrase
    co-occurrence graph and captures a recurring document pattern or
    document type — an empirically derived set of phrases that are
    typically found together in the text stream.

    Attributes:
        community_id: Integer ID assigned by the community detection algorithm.
        phrases: All phrases that belong to this community.
        centroid_phrase: The phrase closest to the community centroid (most
            representative member), or None if not yet computed.
        position_tendency: Mean relative position (0.0 = stream-beginning,
            1.0 = stream-end) of occurrences across the whole corpus.
            Used to infer opening vs. closing role without supervision.
    """

    def __init__(self, community_id: int, phrases: List[str],
                 centroid_phrase: Optional[str] = None,
                 position_tendency: Optional[float] = None):
        self.community_id = community_id
        self.phrases: List[str] = list(phrases)
        self.centroid_phrase = centroid_phrase
        self.position_tendency = position_tendency

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"community_id={self.community_id}, "
                f"n_phrases={len(self.phrases)}, "
                f"centroid={self.centroid_phrase!r})")

    def __len__(self) -> int:
        return len(self.phrases)
