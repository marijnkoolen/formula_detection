from unittest import TestCase

from hist_text_template.cooccurrence import get_word_ngrams
from hist_text_template.cooccurrence import make_cooc_freq
from hist_text_template.vocabulary import Vocabulary
from hist_text_template.template import TextTemplateSearch


class TestNgrams(TestCase):

    def setUp(self) -> None:
        self.sent = ['this', 'is', 'a', 'sentence']
        self.boring_sent = [
            'this', 'is', 'a', 'bit', 'of', 'a', 'repetitive', 'sentence',
            'with', 'a', 'bit', 'of', 'a', 'repetitive', 'message'
        ]

    def test_cannot_make_empty_template_searcher(self):
        error = None
        try:
            TextTemplateSearch()
        except TypeError as err:
            error = err
        self.assertNotEqual(error, None)

    def test_can_make_basic_template_searcher(self):
        template = TextTemplateSearch(sent_iterator=[self.sent])
        self.assertEqual(type(template), TextTemplateSearch)

    def test_template_searcher_set_coll_size(self):
        template = TextTemplateSearch(sent_iterator=[self.sent])
        self.assertEqual(template.coll_size, 4)

    def test_template_searcher_makes_no_cooc(self):
        template = TextTemplateSearch(sent_iterator=[self.sent])
        self.assertEqual(len(template.cooc_freq), 0)

    def test_template_searcher_makes_vocab(self):
        template = TextTemplateSearch(sent_iterator=[self.sent])
        self.assertEqual(len(template.full_vocab), 4)

    def test_template_searcher_makes_min_freq_vocab(self):
        template = TextTemplateSearch(sent_iterator=[self.sent])
        self.assertEqual(len(template.min_freq_vocab), 4)

    def test_template_searcher_min_freq_vocab_has_same_ids(self):
        template = TextTemplateSearch(sent_iterator=[self.sent])
        self.assertEqual(template.min_freq_vocab.term2id('a'), template.full_vocab.term2id('a'))

    def test_template_searcher_filters_min_freq_vocab(self):
        template = TextTemplateSearch(sent_iterator=[self.sent], min_term_freq=2)
        self.assertEqual(len(template.min_freq_vocab), 0)

    def test_template_searcher_makes_min_freq_cooc(self):
        template = TextTemplateSearch(sent_iterator=[self.sent], skip_size=0, min_cooc_freq=1)
        self.assertEqual(len(template.cooc_freq), 3)

    def test_template_searcher_filters_min_freq_cooc(self):
        template = TextTemplateSearch(sent_iterator=[self.boring_sent], min_term_freq=2, min_cooc_freq=2)
        self.assertEqual(len(template.min_freq_vocab), 4)

    def test_template_searcher_can_extract_phrases(self):
        template = TextTemplateSearch(sent_iterator=[self.boring_sent], min_term_freq=2)
        print(template.term_freq)
        template.calculate_co_occurrence_frequencies()
        print('cooc_freq:', template.cooc_freq)
        phrases = [phrase for phrase in template.extract_phrases(min_cooc_freq=2)]
        print('phrases:', phrases)
        self.assertEqual(len(phrases), 2)
