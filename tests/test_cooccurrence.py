from unittest import TestCase

from formula_detection.cooccurrence import get_word_ngrams
from formula_detection.cooccurrence import make_cooc_freq
from fuzzy_search.tokenization.token import Tokenizer
from formula_detection.vocabulary import Vocabulary, make_selected_vocab, calculate_term_freq


class TestNgrams(TestCase):

    def test_make_ngrams_yield_right_size(self):
        sent = ['this', 'is', 'a', 'sentence']
        ngram_size = 3
        ngrams = get_word_ngrams(sent, ngram_size=ngram_size)
        sizes = [len(n) for n in ngrams]
        self.assertEqual(sizes.count(ngram_size), len(sizes))

    def test_make_ngrams_yield_right_number(self):
        sent = ['this', 'is', 'a', 'slightly', 'longer', 'sentence']
        ngram_size = 3
        ngrams = [n for n in get_word_ngrams(sent, ngram_size=ngram_size)]
        self.assertEqual(len(ngrams), len(sent) - ngram_size + 1)


class TestMakeCooc(TestCase):

    def setUp(self) -> None:
        self.tokenizer = Tokenizer(ignorecase=True)
        sent = 'this is a sentence'
        self.doc = self.tokenizer.tokenize(sent)
        boring_sent = 'this is a bit of a repetitive a sentence with a bit of a repetitive message'
        self.boring_doc = self.tokenizer.tokenize(boring_sent)
        self.vocab = Vocabulary(terms=self.doc)

    def test_make_cooc_freq_calculates_freq(self):
        freq = make_cooc_freq([self.doc], self.vocab)
        id1, id2 = self.vocab.term2id('this'), self.vocab.term2id('is')
        self.assertEqual(freq[(id1, id2)], 1)

    def test_make_cooc_freq_uses_term_ids(self):
        freq = make_cooc_freq([self.doc], self.vocab)
        self.assertEqual(freq[('this', 'is')], 0)

    def test_make_cooc_freq_can_use_skips(self):
        freq = make_cooc_freq([self.doc], self.vocab, skip_size=1)
        id1, id2 = self.vocab.term2id('this'), self.vocab.term2id('a')
        self.assertEqual(freq[(id1, id2)], 1)

    def test_make_cooc_freq_does_not_skip_too_far(self):
        freq = make_cooc_freq([self.doc], self.vocab, skip_size=1)
        id1, id2 = self.vocab.term2id('this'), self.vocab.term2id('sentence')
        self.assertEqual(freq[(id1, id2)], 0)

    def test_make_cooc_freq_has_no_none(self):
        vocab = Vocabulary(terms=self.boring_doc)
        term_freq = calculate_term_freq([self.boring_doc], vocab)
        min_freq_vocab = make_selected_vocab(vocab, term_freq=term_freq, min_term_freq=2)
        freq = make_cooc_freq([self.boring_doc], min_freq_vocab, skip_size=4)
        none_cooc = [cooc for cooc in freq if None in cooc]
        self.assertEqual(len(none_cooc), 0)
