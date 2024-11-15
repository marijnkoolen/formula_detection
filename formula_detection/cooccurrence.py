from collections import defaultdict
from collections import Counter
from typing import Generator, Iterable, List, Tuple, Union

from fuzzy_search.tokenization.token import Token
from formula_detection.vocabulary import Vocabulary


def get_skip_coocs(seq_ids: List[int], skip_size: int = 0,
                   skip_none: bool = True) -> Generator[Tuple[int, int], None, None]:
    """
    Generate skipgram co-occurrences for a sequence of term IDs.

    A skipgram co-occurrence occurs when two terms are within a certain
    distance (skip_size) of each other in a sequence of term IDs, allowing
    for a specified number of skipped terms between them.

    :param seq_ids: A list of term IDs (based on a Vocabulary instance).
    :type seq_ids: List[int]
    :param skip_size: The maximum number of skipped terms between co-occurring terms.
    :type skip_size: int
    :param skip_none: Whether to skip co-occurrences that include a `None` value.
    :type skip_none: bool
    :return: A generator of tuples, each containing a pair of co-occurring term IDs.
    :rtype: Generator[Tuple[int, int], None, None]
    """
    for ci, curr_id in enumerate(seq_ids):
        for next_offset in range(ci + 1, ci + 2 + skip_size):
            if next_offset >= len(seq_ids):
                break
            next_id = seq_ids[next_offset]
            if skip_none and None in (curr_id, next_id):
                continue
            yield curr_id, next_id


class SkipCooccurrence:
    """
    A class for calculating and storing skip co-occurrence frequencies of terms.

    Skip co-occurrences are pairs of terms that occur within a specified distance
    (skip_size) of each other in documents, allowing for skipped terms between them.
    """

    def __init__(self, vocabulary: Vocabulary, skip_size: int = 0):
        """
        Initialize the SkipCooccurrence instance with a vocabulary and skip size.

        :param vocabulary: The vocabulary mapping terms to IDs.
        :type vocabulary: Vocabulary
        :param skip_size: The maximum number of skipped terms between co-occurring terms.
        :type skip_size: int
        """
        self.cooc_freq = defaultdict(int)  # A dictionary for co-occurrence frequencies
        self.vocabulary = vocabulary  # The vocabulary instance
        self.skip_size: int = skip_size  # Maximum skip size for co-occurrence calculation

    def calculate_skip_cooccurrences(self, docs: Iterable, skip_size: int = None):
        """
        Calculate the skip co-occurrence frequencies from a sequence of documents.

        Each document is assumed to be a list of term IDs. The co-occurrence frequencies
        are updated for term pairs that appear within the specified skip size.

        :param docs: An iterable of documents, where each document is a list of term IDs.
        :type docs: Iterable
        :param skip_size: The maximum number of skips between co-occurring terms (optional).
        :type skip_size: int, optional
        """
        for doc in docs:
            seq_ids = [self.vocabulary.term2id(t) for t in doc]
            self.cooc_freq.update(get_skip_coocs(seq_ids, skip_size=skip_size))

    def _cooc_ids2terms(self, cooc_ids: Tuple[int, int]) -> Tuple[str, str]:
        """
        Convert a pair of co-occurrence IDs to their corresponding terms.

        :param cooc_ids: A tuple of two term IDs.
        :type cooc_ids: Tuple[int, int]
        :return: A tuple of the corresponding term strings.
        :rtype: Tuple[str, str]
        """
        id1, id2 = cooc_ids
        return self.vocabulary.id2term(id1), self.vocabulary.id2term(id2)

    def get_term_coocs(self, term: Union[str, Token]) -> Union[None, Generator[Tuple[str, str], None, None]]:
        """
        Get the co-occurring terms for a given term.

        The function will yield co-occurring term pairs for the specified term
        from the stored co-occurrence frequencies.

        :param term: The term (as a string or Token) for which to find co-occurring terms.
        :type term: str or Token
        :return: A generator that yields tuples of co-occurring term pairs, or None if
                 the term is not in the vocabulary.
        :rtype: Generator[Tuple[str, str]], or None
        """
        term_id = self.vocabulary.term2id(term)
        if term_id is None:
            return None
        for cooc_ids in self.cooc_freq:
            if term_id in cooc_ids:
                yield self._cooc_ids2terms(cooc_ids), self.cooc_freq[cooc_ids]


def get_word_ngrams(sent: List[str], ngram_size: int = 2) -> Generator[List[str], None, None]:
    """
    Generate n-grams from a list of words (tokens).

    This function returns all possible n-grams from the given sentence, where `n` is
    determined by the `ngram_size`. Each n-gram is a list of words.

    :param sent: The sentence (list of words) from which to generate n-grams.
    :type sent: List[str]
    :param ngram_size: The size of the n-grams to generate.
    :type ngram_size: int
    :return: A generator yielding n-grams (lists of words).
    :rtype: Generator[List[str], None, None]
    """
    for i in range(len(sent) - ngram_size + 1):
        yield tuple(sent[i:i + ngram_size])


def get_context(term_index: int, seq: List[str],
                seq_ids: List[int]) -> Tuple[int, List[str], List[int]]:
    """
    Get the context of a term at a specific index in a sequence.

    This function returns the context terms around a specific term, including its
    position within the context and the corresponding term IDs.

    :param term_index: The index of the target term in the sequence.
    :type term_index: int
    :param seq: The sequence of terms (strings).
    :type seq: List[str]
    :param seq_ids: The sequence of term IDs corresponding to the terms in `seq`.
    :type seq_ids: List[int]
    :return: A tuple containing the position of the term within the context,
             a list of context terms, and a list of context term IDs.
    :rtype: Tuple[int, List[str], List[int]]
    """
    start = term_index - 4 if term_index - 4 >= 0 else 0
    own_index = term_index - start
    end = term_index + 5
    context_terms = seq[start:end]
    context_ids = seq_ids[start:end]
    return own_index, context_terms, context_ids


def cooc_ids2terms(vocab, id1, id2):
    """
    Convert a pair of term IDs to their corresponding terms using the provided vocabulary.

    :param vocab: The vocabulary instance used to convert IDs to terms.
    :type vocab: Vocabulary
    :param id1: The first term ID.
    :type id1: int
    :param id2: The second term ID.
    :type id2: int
    :return: A tuple of the two corresponding terms.
    :rtype: Tuple[str, str]
    """
    return vocab.id2term(id1), vocab.id2term(id2)


def make_cooc_freq(doc_iterator: Iterable, vocab: Vocabulary, skip_size: int = 0,
                   report: bool = False, report_per: int = 1e4) -> Counter:
    """
    Calculate and return the co-occurrence frequencies of term pairs in a sequence of documents.

    Co-occurrences are pairs of terms that appear within a certain distance (skip_size)
    of each other in the documents.

    :param doc_iterator: An iterable of documents, where each document is a list of term IDs.
    :type doc_iterator: Iterable
    :param vocab: The vocabulary instance used to map terms to IDs.
    :type vocab: Vocabulary
    :param skip_size: The maximum number of skipped terms allowed between co-occurring terms.
    :type skip_size: int
    :param report: Whether to print progress updates during processing.
    :type report: bool
    :param report_per: The number of documents after which to print progress updates.
    :type report_per: int
    :return: A `Counter` object containing the co-occurrence frequencies of term pairs.
    :rtype: Counter
    """
    cooc_freq = Counter()
    num_words = 0
    si = 0
    for si, doc in enumerate(doc_iterator):
        num_words += len(doc)
        seq_ids = [vocab.term2id(t) for t in doc]
        cooc_freq.update(get_skip_coocs(seq_ids, skip_size=skip_size))
        if report and (si + 1) % report_per == 0:
            cooc_string = f'num_coocs: {sum(cooc_freq.values()):,}\tnum distinct coocs: {len(cooc_freq):,}'
            print(f'docs: {si + 1}\tnum_words: {num_words:,}\t{cooc_string}')
    cooc_string = f'num_coocs: {sum(cooc_freq.values()):,}\tnum distinct coocs: {len(cooc_freq):,}'
    print(f'docs: {si + 1:,}\tnum_words: {num_words:,}\t{cooc_string}')
    return cooc_freq
