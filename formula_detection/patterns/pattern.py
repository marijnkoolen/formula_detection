from __future__ import annotations
from collections import defaultdict
from collections import Counter
from typing import Iterable, List, Set, Union

from fuzzy_search.tokenization.token import Doc
from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.vocabulary import Vocabulary


class Pattern:

    def __init__(self, labels: List[str]):
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

    def __init__(self, patterns: Union[Pattern, List[Pattern]]):
        if isinstance(patterns, Pattern):
            patterns = [patterns]
        self.patterns = set(patterns)
        self.start_index = defaultdict(set)
        self.end_index = defaultdict(set)

    def __contains__(self, item: Pattern):
        return item in self.patterns

    def __len__(self):
        return len(self.patterns)

    def index_patterns(self, patterns: List[Pattern]):
        for pattern in patterns:
            if pattern not in self.patterns:
                self.patterns.add(pattern)
                self.start_index[pattern.start].add(pattern)
                self.end_index[pattern.end].add(pattern)

    def find_pattern_in_doc(self, doc: Doc) -> bool:
        matches = []
        for token in doc:
            if token.n in self.start_index:
                for pattern in self.start_index[token.n]:
                    end_token = pattern.end
                    end_index = token.i + len(pattern) - 1
                    if doc[end_index] == end_token:
                        match = doc[token.i:end_index]
                        matches.append(match)
        return False


class PhrasePattern:

    def __init__(self, term_ids: Iterable[int]):
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
        return self.id_set.intersection(other.id_set)

    def term_overlap(self, other: PhrasePattern):
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
    if isinstance(pattern, PhrasePattern):
        return pattern.sorted
    return tuple(sorted(pattern))


class PhrasePatternCounter:

    def __init__(self, vocabulary: Vocabulary, phrase_patterns: List[PhrasePattern] = None):
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
        self.set2tuple[pattern.sorted].update([pattern.term_ids])

    def add_phrase(self, phrase: List[Union[str, Token]]):
        term_ids = phrase_to_term_ids(self.vocabulary, phrase)
        self.add_pattern(PhrasePattern(term_ids))

    def _has_pattern_set(self, id_set: Set[int]):
        sorted_set_tuple = pattern_to_id_set_tuple(id_set)
        return sorted_set_tuple in self.set2tuple

    def has_pattern(self, phrase_pattern: Iterable[int]):
        sorted_set_tuple = pattern_to_id_set_tuple(phrase_pattern)
        if sorted_set_tuple not in self.set2tuple:
            return False
        pattern_tuple = tuple(phrase_pattern)
        return pattern_tuple in self.set2tuple[sorted_set_tuple]

    def get_id_set_patterns(self, id_set: Set[int]):
        sorted_set_tuple = pattern_to_id_set_tuple(id_set)
        if sorted_set_tuple not in self.set2tuple:
            return []
        return [Pattern(term_ids) for term_ids in self.set2tuple[sorted_set_tuple]]


def tokens_match_pattern(tokens: List[Token], pattern: Pattern):
    if len(tokens) != len(pattern):
        print('tokens_match_pattern - unequal length')
        print(tokens)
        print(pattern.labels, len(pattern))
        return False
    return all([token.n == label for token, label in zip(tokens, pattern.labels)])


def find_pattern_in_doc(doc: Doc, pattern: Pattern) -> List[List[Token]]:
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
    matches = find_pattern_in_doc(doc, pattern)
    return len(matches) > 0


def phrase_to_term_ids(vocabulary: Vocabulary, tokens: List[Union[str, Token]]):
    term_ids = []
    for token in tokens:
        term_id = vocabulary.term2id(token)
        if term_id is None:
            term_id = vocabulary.term_id['<VAR>'] if '<VAR>' in vocabulary.term_id else -1
        term_ids.append(term_id)
    return term_ids


def phrase_to_phrase_pattern(vocabulary: Vocabulary, tokens: List[Union[str, Token]]):
    term_ids = phrase_to_term_ids(vocabulary=vocabulary, tokens=tokens)
    return PhrasePattern(term_ids)
