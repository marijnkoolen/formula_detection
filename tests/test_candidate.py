from unittest import TestCase

from fuzzy_search.tokenization.token import Tokenizer

from formula_detection.candidate import CandidatePhrase
from formula_detection.candidate import CandidatePhraseMatch
from formula_detection.candidate import transform_candidate_to_string
from formula_detection.candidate import transform_candidate_to_list


class TestTransformToString(TestCase):

    def setUp(self) -> None:
        self.phrase = "that's just like, uh, your opinion, man."
        tokenizer = Tokenizer()
        self.tokens = tokenizer.tokenize(self.phrase)

    def test_can_transform_string_to_string(self):
        string = transform_candidate_to_string(self.phrase)
        self.assertEqual(self.phrase, string)

    def test_can_transform_token_strings_to_string(self):
        string_tokens = self.phrase.split(' ')
        string = transform_candidate_to_string(string_tokens)
        self.assertEqual(self.phrase, string)

    def test_can_transform_token_instances_to_string(self):
        string = transform_candidate_to_string(self.tokens)
        self.assertEqual(self.phrase, string)


class TestCandidatePhrase(TestCase):

    def setUp(self) -> None:
        self.phrase = "that's just like, uh, your opinion, man."

    def test_candidate_has_string_representation(self):
        can_phrase = CandidatePhrase(self.phrase)
        self.assertEqual(self.phrase, can_phrase.phrase_string)

    def test_candidate_has_list_representation(self):
        can_phrase = CandidatePhrase(self.phrase)
        self.assertTrue(self.phrase.startswith(can_phrase.phrase_list[0]))

    def test_candidate_can_be_made_from_list_of_token_instances(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(self.phrase)
        error = None
        try:
            can_phrase = CandidatePhrase(tokens)
        except BaseException as err:
            error = err
        self.assertEqual(None, error)


class TestCandidatePhraseMatch(TestCase):

    def setUp(self) -> None:
        self.text = "Yeah, well, you know, that's just like, uh, your opinion, man."
        self.phrase = "that's just like, uh, your opinion, man."
        tokenizer = Tokenizer()
        self.doc = tokenizer.tokenize_doc(self.text)

    def test_candidate_match_has_candidate_phrase(self):
        can_phrase = CandidatePhrase(self.phrase)
        char_start = self.text.index(self.phrase)
        match = CandidatePhraseMatch(can_phrase, char_start=char_start, doc=self.doc)
        self.assertEqual(can_phrase, match.candidate_phrase)

    def test_candidate_match_has_direct_access_to_phrase_string(self):
        can_phrase = CandidatePhrase(self.phrase)
        char_start = self.text.index(self.phrase)
        match = CandidatePhraseMatch(can_phrase, char_start=char_start, doc=self.doc)
        self.assertEqual(can_phrase.phrase_string, match.phrase)
