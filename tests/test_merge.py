from unittest import TestCase

from formula_detection.merge_bigrams import IndexedTokens
from formula_detection.merge_bigrams import IndexBigram
from formula_detection.merge_bigrams import IndexToken
from formula_detection.merge_bigrams import BigramVariantSet
import formula_detection.merge_bigrams as merge_bigrams


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


class TestIndexedTokens(TestCase):

    def setUp(self) -> None:
        self.words = ['is', 'goedgevonden', 'en', 'verstaan', 'mits', 'desen']

    def test_words_are_indexed(self):
        indexed_tokens = IndexedTokens(self.words)
        word = self.words[2]
        word_index = indexed_tokens.token_indexes[2]
        indexed_token = indexed_tokens.indexed_tokens[word_index[0]]
        self.assertEqual(True, word == indexed_token)

    def test_generates_bigrams(self):
        indexed_tokens = IndexedTokens(self.words)
        for bi, bigram in enumerate(indexed_tokens.get_bigrams(window_size=2)):
            self.assertEqual(self.words[bi], bigram.index_token1.token_string)


class TestIndexedTokensMerge(TestCase):

    def setUp(self) -> None:
        self.words = ['is', 'goedgevonden', 'en', 'verstaan', 'mits', 'desen']
        self.indexed_tokens = IndexedTokens(self.words)
        self.bigrams = [bigram for bigram in self.indexed_tokens.get_bigrams(window_size=3)]
        bigrams = [bigram for bigram in self.bigrams if bigram.index_token1.index == 1 and bigram.index_token2.index == 3]
        self.bigram = bigrams[0]
        self.indexes = (self.bigram.index_token1.index, self.bigram.index_token2.index)

    def test_apply_merge_reduces_index_length(self):
        indexed_tokens = IndexedTokens(self.words)
        indexed_tokens.apply_merge(self.bigram)
        self.assertEqual(True, len(indexed_tokens.token_indexes) == len(indexed_tokens.tokens) - 1)

    def test_apply_merge_correctly_merges_index(self):
        indexed_tokens = IndexedTokens(self.words)
        indexed_tokens.apply_merge(self.bigram)
        merged_index = indexed_tokens.token_indexes[self.bigram.index_token1.index]
        self.assertEqual(self.indexes, merged_index)

    def test_apply_merge_correctly_orders_indexes(self):
        indexed_tokens = IndexedTokens(self.words)
        indexed_tokens.apply_merge(self.bigram)
        next_index = indexed_tokens.token_indexes[self.bigram.index_token1.index + 1]
        self.assertEqual(True, next_index == tuple([self.bigram.index_token1.index + 1]))

    def test_apply_merge_correctly_generates_word_strings(self):
        indexed_tokens = IndexedTokens(self.words)
        indexed_tokens.apply_merge(self.bigram)
        word_string = indexed_tokens._get_index_tokens_as_string(self.indexes)
        bigram = f"{self.words[1]}____{self.words[3]}"
        self.assertEqual(word_string, bigram)

    def test_apply_merge_correctly_generates_bigrams(self):
        indexed_tokens = IndexedTokens(self.words)
        indexed_tokens.apply_merge(self.bigram)
        merge_bigram = None
        for bigram in indexed_tokens.get_bigrams():

            token1 = bigram.index_token1
            token2 = bigram.index_token2
            if bigram.index_token1.token_index_list != (1, 3):
                continue
            if bigram.index_token2.token_index_list != (4,):
                continue
            merge_bigram = bigram
        indexed_tokens.apply_merge(merge_bigram)
        for bigram in indexed_tokens.get_bigrams():
            token1 = bigram.index_token1
            token2 = bigram.index_token2
            self.assertEqual(True, token1.token_index_list[0] < token2.token_index_list[0])
            self.assertEqual(True, token1.token_index_list[-1] < token2.token_index_list[0] + indexed_tokens.window_size)
