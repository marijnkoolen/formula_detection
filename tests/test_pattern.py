from unittest import TestCase

from fuzzy_search.tokenization.token import Tokenizer

from formula_detection.patterns.pattern import Pattern
from formula_detection.patterns.pattern import PatternIndex
from formula_detection.patterns.pattern import pattern_in_doc
from formula_detection.patterns.pattern import find_pattern_in_doc
from formula_detection.patterns.pattern import tokens_match_pattern


class TestPattern(TestCase):

    def test_pattern_is_tuple(self):
        labels = ['A', 'set', 'of', 'labels']
        pattern = Pattern(labels)
        self.assertEqual(True, isinstance(pattern.labels, tuple))

    def test_pattern_can_check_label_inclusion(self):
        labels = ['A', 'set', 'of', 'labels']
        pattern = Pattern(labels)
        self.assertEqual(True, 'set' in pattern)

    def test_pattern_length_is_number_of_labels(self):
        labels = ['A', 'set', 'of', 'labels']
        pattern = Pattern(labels)
        self.assertEqual(len(labels), len(pattern))


class TestPatternIndex(TestCase):

    def setUp(self) -> None:
        self.labels = ['A', 'set', 'of', 'labels']
        self.pattern = Pattern(self.labels)

    def test_pattern_index_can_index_single_pattern(self):
        pattern_index = PatternIndex(self.pattern)
        self.assertEqual(True, self.pattern in pattern_index)

    def test_pattern_index_can_index_pattern_list(self):
        pattern_index = PatternIndex([self.pattern])
        self.assertEqual(True, self.pattern in pattern_index)

    def test_pattern_index_has_length(self):
        pattern_index = PatternIndex(self.pattern)
        self.assertEqual(1, len(pattern_index))


class TestPatternInDoc(TestCase):

    def setUp(self) -> None:
        self.text = 'This is a sentence'
        self.tokenizer = Tokenizer()
        self.doc = self.tokenizer.tokenize_doc(self.text)

    def test_tokens_match_pattern_return_true_when_match(self):
        text = 'There is a repetition is a repetition'
        doc = self.tokenizer.tokenize_doc(text)
        pattern = Pattern(['is', 'a', 'repetition'])
        tokens = doc.tokens[1:4]
        self.assertEqual(True, tokens_match_pattern(tokens, pattern))

    def test_pattern_in_doc_returns_false_when_no_match(self):
        pattern = Pattern(['is', 'a', 'doc'])
        self.assertEqual(False, pattern_in_doc(self.doc, pattern))

    def test_find_pattern_in_doc_returns_empty_when_no_match(self):
        pattern = Pattern(['is', 'a', 'doc'])
        matches = find_pattern_in_doc(self.doc, pattern)
        self.assertEqual(0, len(matches))

    def test_find_pattern_in_doc_returns_match_when_match(self):
        pattern = Pattern(['is', 'a', 'sentence'])
        matches = find_pattern_in_doc(self.doc, pattern)
        self.assertEqual(1, len(matches))

    def test_find_pattern_in_doc_returns_multiple_matches(self):
        text = 'There is a repetition is a repetition'
        doc = self.tokenizer.tokenize_doc(text)
        pattern = Pattern(['is', 'a', 'repetition'])
        matches = find_pattern_in_doc(doc, pattern)
        self.assertEqual(2, len(matches))

    def test_pattern_in_doc_returns_true_when_match(self):
        pattern = Pattern(['is', 'a'])
        self.assertEqual(True, pattern_in_doc(self.doc, pattern))
