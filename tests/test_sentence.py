from unittest import TestCase

from formula_detection.sentence import get_sent_terms


class TestSentence(TestCase):

    def test_get_sent_terms_raises_error_on_string_input(self):
        sent = 'this is a sentence'
        error = None
        try:
            get_sent_terms(sent)
        except TypeError as err:
            error = err
        self.assertEqual(error is None, False)

    def test_get_sent_terms_raises_error_on_invalid_dict(self):
        sent = {'text': 'this is a sentence'}
        error = None
        try:
            get_sent_terms(sent)
        except KeyError as err:
            error = err
        self.assertEqual(error is None, False)

    def test_get_sent_term_return_list_of_strings(self):
        sent = ['this', 'is', 'a', 'sentence']
        terms = get_sent_terms(sent)
        self.assertEqual(sent[0], terms[0])
