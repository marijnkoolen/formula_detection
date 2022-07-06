from unittest import TestCase

from hist_text_template.vocabulary import Vocabulary
from hist_text_template.vocabulary import make_selected_vocab, calculate_term_freq


class TestVocabulary(TestCase):

    def setUp(self) -> None:
        self.sent = ['this', 'is', 'a', 'sentence']

    def test_empty_vocabulary_can_be_made(self):
        vocab = Vocabulary()
        self.assertEqual(isinstance(vocab, Vocabulary), True)

    def test_vocabulary_can_add_term(self):
        vocab = Vocabulary()
        term = 'nonsense'
        vocab.add_term(term)
        self.assertEqual(term in vocab.term_id, True)

    def test_vocabulary_can_add_multipl_terms(self):
        vocab = Vocabulary()
        vocab.index_terms(self.sent)
        self.assertEqual(self.sent[2] in vocab.term_id, True)

    def test_vocabulary_has_correct_size(self):
        vocab = Vocabulary()
        vocab.index_terms(self.sent)
        self.assertEqual(len(vocab), 4)

    def test_vocabulary_can_access_term_by_id(self):
        vocab = Vocabulary(terms=self.sent)
        term = 'a'
        term_id = vocab.term2id(term)
        self.assertEqual(vocab.id2term(term_id), term)

    def test_can_make_selected_vocabulary(self):
        vocab = Vocabulary(terms=self.sent)
        terms = ['a', 'sentence']
        selected_vocab = make_selected_vocab(vocab, selected_terms=terms)
        term_id = vocab.term2id(terms[0])
        self.assertEqual(vocab.id2term(term_id), selected_vocab.id2term(term_id))

    def test_selected_vocabulary_has_correct_size(self):
        vocab = Vocabulary(terms=self.sent)
        terms = ['a', 'sentence']
        selected_vocab = make_selected_vocab(vocab, selected_terms=terms)
        self.assertEqual(len(selected_vocab), len(terms))

    def test_can_make_term_freq(self):
        vocab = Vocabulary(terms=self.sent)
        term_freq = calculate_term_freq([self.sent], vocab)
        self.assertEqual(len(term_freq), len(vocab))

