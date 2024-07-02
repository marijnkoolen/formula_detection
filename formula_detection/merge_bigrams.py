import math
import pickle
from typing import Dict, Iterable, List, Tuple, Union
from functools import reduce

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk import FreqDist
from Levenshtein import distance as levenshtein


_SMALL = 1e-20

_ln = math.log


# Indices to marginals arguments:

NGRAM = 0
"""Marginals index for the ngram count"""

UNIGRAMS = -2
"""Marginals index for a tuple of each unigram count"""

TOTAL = -1
"""Marginals index for the number of words in the data"""


def product(s):
    return reduce(lambda x, y: x * y, s)


class BigramMerger:

    def __init__(self, ft_models):
        self.ft_models = ft_models

    def get_bigram_variants(self, selected_bigrams: List[str], bfd: FreqDist, max_levenshtein_dist: int = 4):
        """Find similar bigrams for a selected set of bigrams, based on FastText embeddings."""
        bigram_variant_set = {}
        for w1, w2 in selected_bigrams:
            if (w1, w2) in bigram_variant_set:
                continue
            bigram_variant_set[(w1, w2)] = (w1, w2)
            w1_sims = self.get_variant_words(w1)
            w2_sims = self.get_variant_words(w2)
            for sim1 in w1_sims:
                for sim2 in w2_sims:
                    if (sim1, sim2) in bfd:
                        ld = levenshtein(f"{w1} {w2}", f"{sim1} {sim2}")
                        if ld > max_levenshtein_dist:
                            continue
                        bigram_variant_set[(sim1, sim2)] = (w1, w2)
        return BigramVariantSet(bigram_variant_set)

    def get_variant_words(self, word: str, max_length_diff: int = 2,
                          max_lev_diff: int = 3) -> Dict[str, Dict[str, float]]:
        """Return a list of similar words for a given word based on FastText embeddings."""
        sim = {}
        for model_type in self.ft_models:
            if self.ft_models[model_type] is None:
                continue
            if len(word) < 4:
                continue
            elif len(word) <= 6:
                topn = 100
            else:
                topn = 1000
            for sim_word, sim_score in self.ft_models[model_type].wv.most_similar(word, topn=topn):
                if abs(len(word) - len(sim_word)) > max_length_diff:
                    continue
                if levenshtein(word, sim_word) > max_lev_diff:
                    continue
                if sim_word not in sim:
                    sim[sim_word] = {model_type: sim_score}
                else:
                    sim[sim_word][model_type] = sim_score
        return sim


def validate_bigram(bigram: any):
    """Checks whether a given token bigram is a valid bigram of two token strings."""
    if isinstance(bigram, tuple) is False:
        raise TypeError(f"bigram must be a tuple of two strings, not {type(bigram)}")
    if len(bigram) != 2:
        raise IndexError(f"bigrams must be a tuple of two strings, received tuple of {len(bigram)} elements")
    for word in bigram:
        if isinstance(word, str) is False:
            tuple_types = [type(word) for word in bigram]
            raise TypeError(f"bigram must be a tuple of two strings, received tuple of {tuple_types}")


def validate_bigram_variant_set(bigram_variant_set: Union[Dict[Tuple[str, str],
                                                               Tuple[str, str]],
                                                          List[Tuple[str, str]]]):
    """Checks whether a set of bigram variants consists of valid bigrams."""
    if isinstance(bigram_variant_set, list):
        for bigram in bigram_variant_set:
            validate_bigram(bigram)
    elif isinstance(bigram_variant_set, dict):
        for bigram in bigram_variant_set:
            validate_bigram(bigram)
            validate_bigram(bigram_variant_set[bigram])
    else:
        raise TypeError('bigram_variant_set must be a list of bigram tuples or a dictionary of '
                        'bigram to bigram mappings')


class BigramVariantSet:

    def __init__(self, bigram_variant_set: Union[Dict[Tuple[str, str], Tuple[str, str]], List[Tuple[str, str]]]):
        """A mapping for a list of bigrams to their preferred spelling. preferred spellings map to themselves.

        :param bigram_variant_set: either a dictionary with mappings from a bigram with tokens
        that are spelling variants of the preferred tokens that they map to, or a list of bigrams that
        have no spelling variants. In the latter case, the bigrams are turned into a dictionary in which they
        map to themselves.
        """
        validate_bigram_variant_set(bigram_variant_set)
        if isinstance(bigram_variant_set, list):
            bigram_map = {}
            for bigram in bigram_variant_set:
                bigram_map[bigram] = bigram
            bigram_variant_set = bigram_map
        self.bigram_variant_set = bigram_variant_set
        missing = []
        for bigram in list(bigram_variant_set.values()):
            if bigram not in self.bigram_variant_set:
                self.bigram_variant_set[bigram] = bigram
                missing.append(bigram)
        if len(missing) > 0:
            print(f'added {len(missing)} missing bigrams that are only in the values:', missing)

    def __iter__(self):
        for bigram in self.bigram_variant_set:
            yield bigram

    def __contains__(self, bigram):
        return bigram in self.bigram_variant_set

    def __len__(self):
        return len(self.bigram_variant_set)

    def map(self, bigram):
        return self.bigram_variant_set[bigram]


class IndexToken:

    def __init__(self, index: int, token_string: str, token_index_list: Tuple[int, ...]):
        """A convenience representation of a token string from a list of tokens. This contains the token string
        representation and a list of the token indexes in the original list of text tokens.

        :param index: the index of the token string in an indexed token list
        :type index: int
        :param token_string: the string representation of the token or the merge list of tokens
        :type token_string: str
        :param token_index_list: the list of indexes of the merged tokens in the original list of text tokens.
        :type token_index_list: Tuple[int, ...]
        """
        self.index = index
        self.token_string = token_string
        self.token_index_list = tuple(token_index_list)

    def __repr__(self):
        return f'IndexToken(index={self.index}, token_string="{self.token_string}", ' \
               f'token_index_list={self.token_index_list})'


class IndexBigram:

    def __init__(self, index_token1: IndexToken, index_token2: IndexToken):
        """An IndexBigram combines two IndexTokens and the bigram of their string representation.

        :param index_token1: an IndexToken representation of the first token
        :type index_token1: IndexToken
        :param index_token2: an IndexToken representation of the second token
        :type index_token2: IndexToken
        """
        self.index_token1 = index_token1
        self.index_token2 = index_token2
        self.bigram = (index_token1.token_string, index_token2.token_string)


class IndexedTokens:

    def __init__(self, tokens: List[str], window_size: int = 3):
        """Create an IndexedTokens instance based on a list of string tokens and a window size.
        The IndexedTokens keeps track of the original token indexes in the list, so that bigrams
        with skips can be iteratively and hierarchically merged while retaining the original
        token order.

        :param tokens: a list of string tokens
        :type tokens: List[str]
        :param window_size: the size of the window from which to generate token bigrams
        :type window_size: int
        """
        self.tokens = tokens
        self.indexed_tokens = tokens
        self.token_indexes: List[Tuple[int]] = [tuple([i]) for i in range(0, len(tokens))]
        self.window_size = window_size

    def __len__(self):
        return len(self.indexed_tokens)

    def __repr__(self):
        indexed_tokens_string = ',\n\t'.join([str(iw) for iw in self])
        if len(self) > 0:
            indexed_tokens_string = f'\n\t{indexed_tokens_string}\n'
        return f"IndexedTokens(indexed_tokens=[{indexed_tokens_string}])"

    def __iter__(self):
        for wi, token_index_list in enumerate(self.token_indexes):
            token_string = self._get_index_tokens_as_string(token_index_list)
            yield IndexToken(wi, token_string, token_index_list)

    def _get_index_tokens_as_string(self, token_index_list: Tuple[int, ...], default_infix: str = '__'):
        token_string = ''
        prev_token_index = None
        for curr_token_index in token_index_list:
            if curr_token_index != token_index_list[0]:
                token_string += f"{default_infix * (curr_token_index - prev_token_index)}"
            token_string += self.tokens[curr_token_index]
            prev_token_index = curr_token_index
        return token_string

    def get_bigrams(self, window_size: int = 3):
        """Returns a generator that yields IndexedToken bigrams. Each bigram consists of
        a IndexedToken pair and their bigram representation. """
        for ci, curr_index_list in enumerate(self.token_indexes):
            num_higher = 0
            # print('curr:', ci, curr_index_list)
            for ni, next_index_list in enumerate(self.token_indexes[ci+1:]):
                if next_index_list[-1] > curr_index_list[-1]:
                    num_higher += 1
                if num_higher >= window_size:
                    break
                if next_index_list[0] >= curr_index_list[-1] + window_size:
                    break
                # print('\tnext:', ni, next_index_list)
                token_string1 = self._get_index_tokens_as_string(curr_index_list)
                token_string2 = self._get_index_tokens_as_string(next_index_list)
                index_token1 = IndexToken(ci, token_string1, curr_index_list)
                index_token2 = IndexToken(ci+ni+1, token_string2, next_index_list)
                yield IndexBigram(index_token1, index_token2)

    def apply_merge(self, bigram: IndexBigram) -> None:
        """Apply the merge of two tokens that are part of the given bigram.

        :param bigram: a bigram consisting of to IndexedToken instances
        :type bigram: IndexBigram
        """
        # print('START MERGING')
        # print('\ttoken_indexes:', self.token_indexes)
        index1 = bigram.index_token1.index
        index2 = bigram.index_token2.index
        # print('\tindex1:', index1)
        # print('\tindex2:', index2)
        token_index1 = self.token_indexes[index1]
        token_index2 = self.token_indexes[index2]
        # print('\ttoken_index1:', token_index1)
        # print('\ttoken_index2:', token_index2)
        merged_token_index = tuple(sorted(list(token_index1) + list(token_index2)))
        # print('\tmerged_token_index:', merged_token_index)
        pre_index = self.token_indexes[:index1]
        # print('\tpre_index:', pre_index)
        between_index = self.token_indexes[index1+1:index2]
        # print('\tbetween_index:', between_index)
        post_index = self.token_indexes[index2+1:] if index2 < len(self.token_indexes) - 1 else []
        # print('\tpost_index:', post_index)
        self.token_indexes = pre_index + [merged_token_index] + between_index + post_index
        # print('\ttoken_indexes:', self.token_indexes)
        # print('FINISHED MERGING')

    def apply_merge_set(self, bigram_variant_set: BigramVariantSet, window_size: int = 3,
                        debug: bool = False) -> None:
        """Apply a set of bigram variants.

        :param bigram_variant_set: a set of bigram variants mapped to their preferred spelling
        :type bigram_variant_set: BigramVariantSet
        :param window_size: the size of the window from which to generate token bigrams.
        :type window_size: int
        :param debug: a debugging flag
        :type debug: bool
        Validate """
        to_apply = []
        for bigram in self.get_bigrams(window_size=window_size):
            token1 = bigram.index_token1
            token2 = bigram.index_token2
            if bigram.bigram in bigram_variant_set:
                if debug:
                    print(f'\tMerging {token1.token_string} {token2.token_string} '
                          f'({token1.token_index_list}, {token2.token_index_list})')
                to_apply.append(bigram)
        for bigram in to_apply[::-1]:
            # Apply bigrams in reverse token index order, so that indexes don't get mixed up.
            self.apply_merge(bigram)

    def apply_merge_sets(self, bigram_variant_sets: List[BigramVariantSet], window_size: int = 3,
                         debug: bool = False):
        """Apply a list of bigram variant sets that are apply in order. The lists are applied one by one
        because later sets might merge tokens that are the results of merges in previous sets.

        :param bigram_variant_sets: a list of bigram variants sets that map bigram variants to their preferred spelling
        :type bigram_variant_sets: BigramVariantSet
        :param window_size: the size of the window from which to generate token bigrams.
        :type window_size: int
        :param debug: a debugging flag
        :type debug: bool
        """
        for bigram_variant_set in bigram_variant_sets:
            self.apply_merge_set(bigram_variant_set, window_size=window_size, debug=debug)


def write_bigrams(bcf: BigramCollocationFinder, best_bigrams: List[str], bigram_file: int) -> None:
    with open(bigram_file, 'wb') as fh:
        pickle.dump({'bcf': bcf, 'best_bigrams': best_bigrams}, fh)


def read_bigrams(bigram_file: str) -> Tuple[BigramCollocationFinder, List[str]]:
    with open(bigram_file, 'rb') as fh:
        data = pickle.load(fh)
    return data['bcf'], data['best_bigrams']


class BigramFinder:

    _n = 0

    def __init__(self, ufd1, ufd2, bfd, window_size: int = 2):
        self.ufd1 = ufd1
        self.ufd2 = ufd2
        self.bfd = bfd
        self.window_size = window_size
        self.N = sum(ufd1.values())

    @staticmethod
    def _contingency(n_ii, n_ix_xi_tuple, n_xx):
        """Calculates values of a bigram contingency table from marginal values."""
        (n_ix, n_xi) = n_ix_xi_tuple
        n_oi = n_xi - n_ii
        n_io = n_ix - n_ii
        return n_ii, n_oi, n_io, n_xx - n_ii - n_oi - n_io

    @staticmethod
    def _marginals(n_ii, n_oi, n_io, n_oo):
        """Calculates values of contingency table marginals from its values."""
        return n_ii, (n_oi + n_ii, n_io + n_ii), n_oo + n_oi + n_io + n_ii

    @staticmethod
    def _expected_values(cont):
        """Calculates expected values for a contingency table."""
        n_xx = sum(cont)
        # For each contingency table cell
        for i in range(4):
            yield (cont[i] + cont[i ^ 1]) * (cont[i] + cont[i ^ 2]) / n_xx

    @classmethod
    def pmi(cls, *marginals):
        """Scores ngrams by pointwise mutual information, as in Manning and
        Schutze 5.4.
        """
        return math.log2(marginals[NGRAM] * marginals[TOTAL] ** (cls._n - 1)) - math.log2(
            product(marginals[UNIGRAMS])
        )

    @classmethod
    def _likelihood_ratio(cls, *marginals):
        """Scores ngrams using likelihood ratios as in Manning and Schutze 5.3.4."""
        cont = cls._contingency(*marginals)
        return 2 * sum(
            obs * _ln(obs / (exp + _SMALL) + _SMALL)
            for obs, exp in zip(cont, cls._expected_values(cont))
        )

    def score_ngram(self, w1, w2, score_fn=None):
        """Returns the score for a given bigram using the given scoring
        function.  Following Church and Hanks (1990), counts are scaled by
        a factor of 1/(window_size - 1).
        """
        if score_fn is None:
            score_fn = self._likelihood_ratio
        n_all = self.N
        # n_ii = self.bfd[(w1, w2)] / (self.window_size - 1.0)
        n_ii = self.bfd[(w1, w2)]
        if not n_ii:
            return
        n_ix = self.ufd1[w1]
        n_xi = self.ufd2[w2]
        return score_fn(n_ii, (n_ix, n_xi), n_all)

    def _score_ngrams(self, score_fn):
        """Generates of (ngram, score) pairs as determined by the scoring
        function provided.
        """
        for tup in self.bfd:
            score = self.score_ngram(score_fn, *tup)
            if score is not None:
                yield tup, score

    def score_ngrams(self, score_fn):
        """Returns a sequence of (ngram, score) pairs ordered from highest to
        lowest score, as determined by the scoring function provided.
        """
        return sorted(self._score_ngrams(score_fn), key=lambda t: (-t[1], t[0]))

    def nbest(self, score_fn, n):
        """Returns the top n ngrams when scored by the given function."""
        return [p for p, s in self.score_ngrams(score_fn)[:n]]

    def above_score(self, score_fn, min_score):
        """Returns a sequence of ngrams, ordered by decreasing score, whose
        scores each exceed the given minimum score.
        """
        for ngram, score in self.score_ngrams(score_fn):
            if score > min_score:
                yield ngram
            else:
                break


def make_bigram_collocation_finder(texts: Iterable, bigram_variant_sets: List[BigramVariantSet] = None,
                                   window_size: int = 3) -> BigramFinder:
    """Create an NLTK BigramCollocationFinder from bigrams generated after applying
    a list of bigram variant sets ot a list of texts.

    :param texts: an Iterable of texts, with each text represented as a tokenised list of strings
    :type texts: Iterable[str]
    :param bigram_variant_sets: a list of bigram variant sets, to be applied in list order
    :type bigram_variant_sets: List[BigramVariantSet]
    :param window_size: the size of the window from which to generate bigram token pairs
    :type window_size: int
    :return: an NLTK BigramCollocationFinder
    :rtype: BigramCollocationFinder
    """
    # ufd = nltk.FreqDist()
    ufd1 = nltk.FreqDist()
    ufd2 = nltk.FreqDist()
    bfd = nltk.FreqDist()

    for pi, text in enumerate(texts):
        indexed_tokens = IndexedTokens(text['words'])
        if bigram_variant_sets:
            indexed_tokens.apply_merge_sets(bigram_variant_sets)
        # unigram count of tokens
        # ufd.update([indexed_token.token_string for indexed_token in indexed_tokens])
        for bigram in indexed_tokens.get_bigrams(window_size=window_size):
            bfd[bigram.bigram] += 1
            ufd1[bigram.index_token1.token_string] += 1
            ufd2[bigram.index_token2.token_string] += 1
        if (pi+1) >= 1000000:
            break

    return BigramFinder(ufd1, ufd2, bfd, window_size=window_size)
    # return nltk.collocations.BigramCollocationFinder(ufd, bfd, window_size=window_size)


def select_non_overlapping_bigrams(best_bigrams: List[str]) -> List[str]:
    """Filter a list of bigrams that is ordered by likelihood ratio to retain only bigrams
    that have unique words. That is, if there is term overlap between two bigrams,
    select only the bigram with the highest likelihood ratio."""

    selected_words = set()
    selected_bigrams = []
    for bigram in best_bigrams:
        w1, w2 = bigram
        if len(w1) + len(w2) <= 8:
            continue
        if w1 in selected_words or w2 in selected_words:
            continue
        selected_bigrams.append(bigram)
        selected_words.add(w1)
        selected_words.add(w2)
    return selected_bigrams


