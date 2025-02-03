from unittest import TestCase

from fuzzy_search.tokenization.token import Tokenizer
from formula_detection.search import FormulaSearch


class TestNgrams(TestCase):

    def setUp(self) -> None:
        self.tokenizer = Tokenizer(ignorecase=True)
        sent = 'this is a sentence'
        self.doc = self.tokenizer.tokenize_doc(sent)
        boring_sent = 'this is a bit of a repetitive a sentence with a bit of a repetitive message'
        self.boring_doc = self.tokenizer.tokenize_doc(boring_sent)

    def test_cannot_make_empty_formula_searcher(self):
        error = None
        try:
            FormulaSearch()
        except TypeError as err:
            error = err
        self.assertNotEqual(error, None)

    def test_can_make_basic_formula_searcher(self):
        formula_search = FormulaSearch(doc_iterator=[self.doc])
        self.assertEqual(type(formula_search), FormulaSearch)

    def test_formula_searcher_set_coll_size(self):
        formula_search = FormulaSearch(doc_iterator=[self.doc])
        self.assertEqual(len(self.doc), formula_search.coll_size)

    def test_formula_searcher_makes_no_cooc(self):
        formula_search = FormulaSearch(doc_iterator=[self.doc])
        self.assertEqual(0, len(formula_search.cooc_freq))

    def test_formula_searcher_makes_vocab(self):
        formula_search = FormulaSearch(doc_iterator=[self.doc])
        self.assertEqual(len(self.doc), len(formula_search.full_vocab))

    def test_formula_searcher_makes_min_freq_vocab(self):
        formula_search = FormulaSearch(doc_iterator=[self.doc])
        self.assertEqual(len(set([token.n for token in self.doc])), len(formula_search.min_freq_vocab))

    def test_formula_searcher_min_freq_vocab_has_same_ids(self):
        formula_search = FormulaSearch(doc_iterator=[self.doc])
        self.assertEqual(formula_search.min_freq_vocab.term2id('a'), formula_search.full_vocab.term2id('a'))

    def test_formula_searcher_filters_min_freq_vocab(self):
        formula_search = FormulaSearch(doc_iterator=[self.doc], min_term_freq=2)
        self.assertEqual(len(formula_search.min_freq_vocab), 0)

    def test_formula_searcher_makes_min_freq_cooc(self):
        formula_search = FormulaSearch(doc_iterator=[self.doc], skip_size=0, min_cooc_freq=1)
        self.assertEqual(len(formula_search.cooc_freq), 3)

    def test_formula_searcher_filters_min_freq_cooc(self):
        formula_search = FormulaSearch(doc_iterator=[self.boring_doc], min_term_freq=2, min_cooc_freq=2)
        self.assertEqual(len(formula_search.min_freq_vocab), 4)

    def test_formula_searcher_can_extract_phrases(self):
        formula_search = FormulaSearch(doc_iterator=[self.boring_doc], min_term_freq=2)
        formula_search.calculate_co_occurrence_frequencies()
        phrases = [phrase for phrase in formula_search.extract_phrases(phrase_type='long_phrases',
                                                                       min_cooc_freq=2)]
        self.assertEqual(len(phrases), 2)
