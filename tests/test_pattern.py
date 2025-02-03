from unittest import TestCase

from fuzzy_search.tokenization.token import Tokenizer
from fuzzy_search.tokenization.vocabulary import Vocabulary

from formula_detection.patterns.pattern import Pattern
from formula_detection.patterns.pattern import PatternIndex
from formula_detection.patterns.pattern import PhrasePatternCounter
from formula_detection.patterns.pattern import pattern_in_doc
from formula_detection.patterns.pattern import phrase_to_term_ids
from formula_detection.patterns.pattern import phrase_to_phrase_pattern
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


class TestPhrasePattern(TestCase):

    def setUp(self) -> None:
        self.text = "There is a repetition is a repetition"
        self.tokenizer = Tokenizer()
        self.doc = self.tokenizer.tokenize_doc(self.text)
        self.vocab = Vocabulary(self.doc.tokens)

    def test_phrase_to_ids(self):
        phrase = ['is', 'a', 'repetition']
        term_ids = phrase_to_term_ids(self.vocab, phrase)
        self.assertEqual(False, any([term_id is None for term_id in term_ids]))

    def test_phrase_to_ids_uses_minus_one_when_term_not_in_vocab(self):
        phrase = ['is', 'a', 'sentence']
        term_ids = phrase_to_term_ids(self.vocab, phrase)
        self.assertEqual(True, -1 in term_ids)

    def test_phrase_to_ids_uses_var_id_when_term_not_in_vocab_and_var_in_vocab(self):
        self.vocab.add_terms('<VAR>')
        var_id = self.vocab.term_id['<VAR>']
        phrase = ['is', 'a', 'sentence']
        term_ids = phrase_to_term_ids(self.vocab, phrase)
        self.assertEqual(True, var_id in term_ids)

    def test_phrase_pattern_sorts_term_ids(self):
        phrase = ['a', 'repetition', 'is']
        pattern = phrase_to_phrase_pattern(self.vocab, phrase)
        term_ids = phrase_to_term_ids(self.vocab, phrase)
        self.assertEqual(tuple(sorted(set(term_ids))), pattern.sorted)

    def test_phrase_patterns_matches_token_anagram_to_ids(self):
        phrase1 = ['is', 'a', 'repetition']
        phrase2 = ['a', 'repetition', 'is']
        pattern1 = phrase_to_phrase_pattern(self.vocab, phrase1)
        pattern2 = phrase_to_phrase_pattern(self.vocab, phrase2)
        self.assertEqual(pattern1.id_set, pattern2.id_set)

    def test_phrase_patterns_matches_multi_token_anagram_to_ids(self):
        phrase1 = ['is', 'a', 'repetition']
        phrase2 = ['is', 'a', 'repetition', 'is', 'repetition', 'a']
        pattern1 = phrase_to_phrase_pattern(self.vocab, phrase1)
        pattern2 = phrase_to_phrase_pattern(self.vocab, phrase2)
        self.assertEqual(pattern1.id_set, pattern2.id_set)

    def test_phrase_pattern_equality_is_set_based(self):
        phrase1 = ['is', 'a', 'repetition']
        phrase2 = ['a', 'repetition', 'is']
        pattern1 = phrase_to_phrase_pattern(self.vocab, phrase1)
        pattern2 = phrase_to_phrase_pattern(self.vocab, phrase2)
        self.assertEqual(pattern1, pattern2)

    def test_phrase_pattern_can_check_contains_of_term_ids(self):
        phrase1 = ['is', 'a', 'repetition']
        phrase2 = ['a', 'repetition', 'is']
        pattern1 = phrase_to_phrase_pattern(self.vocab, phrase1)
        pattern2 = phrase_to_phrase_pattern(self.vocab, phrase2)
        self.assertEqual(True, pattern1.term_ids[0] in pattern2)

    def test_phrase_pattern_can_check_contains_of_phrase_patterns(self):
        phrase1 = ['is', 'a', 'repetition']
        phrase2 = ['a', 'repetition', 'is']
        pattern1 = phrase_to_phrase_pattern(self.vocab, phrase1)
        pattern2 = phrase_to_phrase_pattern(self.vocab, phrase2)
        self.assertEqual(True, pattern1 in pattern2)

    def test_phrase_pattern_can_check_set_overlap(self):
        phrase1 = ['is', 'a', 'repetition']
        phrase2 = ['a', 'repetition', 'is']
        pattern1 = phrase_to_phrase_pattern(self.vocab, phrase1)
        pattern2 = phrase_to_phrase_pattern(self.vocab, phrase2)
        pattern_set_overlap = pattern1.set_overlap(pattern2)
        term_set_overlap = set(phrase1).intersection(set(phrase2))
        term_id_set_overlap = set([self.vocab.term_id[term] for term in term_set_overlap])
        self.assertEqual(term_id_set_overlap, pattern_set_overlap)

    def test_phrase_pattern_can_check_term_overlap(self):
        phrase1 = ['is', 'a', 'repetition', 'is', 'a']
        phrase2 = ['a', 'repetition', 'is', 'a', 'repetition']
        pattern1 = phrase_to_phrase_pattern(self.vocab, phrase1)
        pattern2 = phrase_to_phrase_pattern(self.vocab, phrase2)
        pattern_term_overlap = pattern1.term_overlap(pattern2)
        term_overlap = ['is', 'a', 'repetition', 'a']
        term_id_overlap = tuple([self.vocab.term_id[term] for term in term_overlap])
        self.assertEqual(term_id_overlap, pattern_term_overlap)


class TestPhrasePatternCounter(TestCase):

    def setUp(self) -> None:
        self.text = "There is a repetition is a repetition"
        self.tokenizer = Tokenizer()
        self.doc = self.tokenizer.tokenize_doc(self.text)
        self.vocab = Vocabulary(self.doc.tokens)
        phrase1 = ['is', 'a', 'repetition']
        phrase2 = ['a', 'repetition', 'is']
        self.pattern1 = phrase_to_phrase_pattern(self.vocab, phrase1)
        self.pattern2 = phrase_to_phrase_pattern(self.vocab, phrase2)

    def test_phrase_pattern_counter_can_check_contains_set(self):
        pp_count = PhrasePatternCounter(self.vocab, [self.pattern1, self.pattern2])
        self.assertEqual(True, self.pattern1 in pp_count)

    def test_phrase_pattern_counter_maps_patterns_to_same_set(self):
        pp_count = PhrasePatternCounter(self.vocab, [self.pattern1, self.pattern2])
        p_set = self.pattern1.sorted
        freq1 = pp_count.set2tuple[p_set][self.pattern1.term_ids]
        freq2 = pp_count.set2tuple[p_set][self.pattern2.term_ids]
        self.assertEqual(True, all([freq == 1 for freq in [freq1, freq2]]))
