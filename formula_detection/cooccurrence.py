from collections import defaultdict
from collections import Counter
from typing import Generator, Iterable, List, Tuple, Union

from fuzzy_search.tokenization.token import Token
from formula_detection.vocabulary import Vocabulary


def get_skip_coocs(seq_ids: List[str], skip_size: int = 0,
                   skip_none: bool = True) -> Generator[Tuple[int, int], None, None]:
    """Return a list of skipgram co-occurrences for a sequence of term ids.
    :param seq_ids: a sequence of term ids (based on a Vocabulary instance)
    :type seq_ids: List[str]
    :param skip_size: the maximum number of skips
    :type skip_size: int
    :param skip_none: whether to skip co-occurrence that includes a None
    :return a generator of co-occurring term ids"""
    for ci, curr_id in enumerate(seq_ids):
        for next_offset in range(ci+1, ci + 2 + skip_size):
            if next_offset >= len(seq_ids):
                break
            next_id = seq_ids[next_offset]
            if skip_none and None in (curr_id, next_id):
                continue
            yield curr_id, next_id


class SkipCooccurrence:

    def __init__(self, vocabulary: Vocabulary, skip_size: int = 0):
        """A skip co-occurrence frequency counter."""
        self.cooc_freq = defaultdict(int)
        self.vocabulary = vocabulary
        self.skip_size: int = skip_size

    def calculate_skip_cooccurrences(self, docs: Iterable, skip_size: int = None):
        """Calculate the skip co-occurrences of terms from a doc iterable.
        The iterable should return docs as list of term ids
        :param docs: an iterable of docs
        :type docs: Iterable
        :param skip_size: the maximum number of skips between co-occurring term
        :type skip_size: int
        """
        for doc in docs:
            seq_ids = [self.vocabulary.term2id(t) for t in doc]
            self.cooc_freq.update(get_skip_coocs(seq_ids, skip_size=skip_size))

    def _cooc_ids2terms(self, cooc_ids: Tuple[int, int]) -> Tuple[str, str]:
        id1, id2 = cooc_ids
        return self.vocabulary.id2term(id1), self.vocabulary.id2term(id2)

    def get_term_coocs(self, term: Union[str, Token]) -> Union[None, Generator[Tuple[str, str], None, None]]:
        """Return a generator of term co-occurrences for a given term.

        :param term: a term string for which to lookup co-occurring terms
        :type term: str
        :return: a generator yield tuples of two co-occurring terms
        :rtype: Generator[Tuple[str, str]]
        """
        term_id = self.vocabulary.term2id(term)
        if term_id is None:
            return None
        for cooc_ids in self.cooc_freq:
            if term_id in cooc_ids:
                yield self._cooc_ids2terms(cooc_ids), self.cooc_freq[cooc_ids]


def get_word_ngrams(sent: List[str], ngram_size: int = 2) -> Generator[List[str], None, None]:
    for i in range(len(sent) - ngram_size + 1):
        yield tuple(sent[i:i + ngram_size])


def get_context(term_index: int, seq: List[str],
                seq_ids: List[int]) -> Tuple[int, List[str], List[int]]:
    start = term_index - 4 if term_index - 4 >= 0 else 0
    own_index = term_index - start
    end = term_index + 5
    context_terms = seq[start:end]
    context_ids = seq_ids[start:end]
    return own_index, context_terms, context_ids


def cooc_ids2terms(vocab, id1, id2):
    return vocab.id2term(id1), vocab.id2term(id2)


def make_cooc_freq(doc_iterator: Iterable, vocab: Vocabulary, skip_size: int = 0,
                   report: bool = False, report_per: int = 1e4) -> Counter:
    """Returns co-occurrence frequencies of terms in a doc with at
    most skip_size words in between.

    :param doc_iterator: an iterable that yield doc objects
    :type doc_iterator: Iterable
    :param vocab: the vocabulary of accepted terms
    :type vocab: Vocabulary
    :param skip_size: the maximum number of skips to use for co-occurring term pairs
    :type skip_size: int
    :param report: whether to report progress of processing the doc objects
    :type report: bool
    :param report_per: the number of lines after which to report
    :type report_per: int
    :return: the co-occurrence frequencies of pairs of words that are both in the vocabulary
    :rtype: Counter
    ."""
    cooc_freq = Counter()
    num_words = 0
    si = 0
    for si, doc in enumerate(doc_iterator):
        num_words += len(doc)
        seq_ids = [vocab.term2id(t) for t in doc]
        cooc_freq.update(get_skip_coocs(seq_ids, skip_size=skip_size))
        if report and (si + 1) % report_per == 0:
            cooc_string = f'num_coocs: {sum(cooc_freq.values())}\tnum distinct coocs: {len(cooc_freq)}'
            print(f'docs: {si + 1}\tnum_words: {num_words}\t{cooc_string}')
    cooc_string = f'num_coocs: {sum(cooc_freq.values())}\tnum distinct coocs: {len(cooc_freq)}'
    print(f'docs: {si + 1}\tnum_words: {num_words}\t{cooc_string}')
    return cooc_freq
