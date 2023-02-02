from unittest import TestCase

from hist_text_template.merge_bigrams import IndexedWords
from hist_text_template.merge_bigrams import BigramVariantSet
import hist_text_template.merge_bigrams as merge_bigrams


class TestBigramVariantSet(TestCase):

    def test_non_tuple_throws_error(self):
        error = None
        try:
            merge_bigrams.validate_bigram(1)
        except TypeError as err:
            error = err
        self.assertEqual(True, error is not None)

    def test_non_string_throws_error(self):
        variant_map = {
            ('variant_word1', 'variant_word2'): ('word1', 1)
        }
        error = None
        try:
            BigramVariantSet(variant_map)
        except TypeError as err:
            error = err
        self.assertEqual(True, error is not None)

    def test_set_accepts_dict(self):
        variant_map = {
            ('variant_word1', 'variant_word2'): ('word1', 'word2')
        }
        bigram_variant_set = BigramVariantSet(variant_map)
        self.assertEqual(2, len(bigram_variant_set))


class TestIndexedWords(TestCase):

    def setUp(self) -> None:
        self.words = ['is', 'goedgevonden', 'en', 'verstaan', 'mits', 'desen']

    def test_words_are_indexed(self):
        indexed_words = IndexedWords(self.words)
        word = self.words[2]
        word_index = indexed_words.word_indexes[2]
        indexed_word = indexed_words.indexed_words[word_index[0]]
        self.assertEqual(True, word == indexed_word)

    def test_apply_merge_reduces_index_length(self):
        indexed_words = IndexedWords(self.words)
        indexed_words.apply_merge(1, 3)
        self.assertEqual(True, len(indexed_words.word_indexes) == len(indexed_words.words) - 1)

    def test_apply_merge_correctly_merges_index(self):
        indexed_words = IndexedWords(self.words)
        indexes = (1, 3)
        indexed_words.apply_merge(indexes[0], indexes[1])
        merged_index = indexed_words.word_indexes[1]
        self.assertEqual(True, indexes == merged_index)

    def test_apply_merge_correctly_orders_indexes(self):
        indexed_words = IndexedWords(self.words)
        indexed_words.apply_merge(1, 3)
        next_index = indexed_words.word_indexes[2]
        self.assertEqual(True, next_index == tuple([2]))

    def test_apply_merge_correctly_generates_word_strings(self):
        indexed_words = IndexedWords(self.words)
        indexes = (1, 3)
        indexed_words.apply_merge(indexes[0], indexes[1])
        word_string = indexed_words._get_index_words_as_string(indexes)
        bigram = f"{self.words[1]}____{self.words[3]}"
        self.assertEqual(word_string, bigram)

    def test_apply_merge_correctly_generates_bigrams(self):
        indexed_words = IndexedWords(self.words)
        indexed_words.apply_merge(1, 3)
        indexed_words.apply_merge(3, 4)
        for word1, word2 in indexed_words.get_bigrams():
            self.assertEqual(True, word1.word_index_list[0] < word2.word_index_list[0])
            self.assertEqual(True, word1.word_index_list[-1] < word2.word_index_list[0] + indexed_words.window_size)
