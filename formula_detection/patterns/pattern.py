from collections import defaultdict
from typing import List, Union

from fuzzy_search.tokenization.token import Doc
from fuzzy_search.tokenization.token import Token


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
