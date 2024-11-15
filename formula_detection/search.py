from collections import Counter
from collections import defaultdict
from itertools import combinations
from typing import Dict, Generator, Iterable, List, Tuple, Union

from fuzzy_search.tokenization.token import Doc
from fuzzy_search.tokenization.token import Token
from formula_detection.vocabulary import Vocabulary, make_selected_vocab
from formula_detection.cooccurrence import make_cooc_freq, get_context
from formula_detection.candidate import CandidatePhraseMatch
from formula_detection.candidate import make_candidate_phrase
from formula_detection.candidate import make_candidate_phrase_match
from formula_detection.candidate import transform_candidates_to_strings


def extract_sub_phrases_with_skips(phrase: List[str], ngram_size: int, max_skips: int):
    """
    Extracts sub-phrases from the given phrase, allowing for a certain number of skipped terms
    between n-gram components.

    Args:
        phrase (List[str]): The phrase from which to extract sub-phrases.
        ngram_size (int): The desired size of each n-gram.
        max_skips (int): The maximum number of terms that can be skipped in each n-gram.

    Yields:
        List[str]: A sub-phrase, which is a list of strings representing the terms in the n-gram.
    """
    for ti, token in enumerate(phrase[:-(ngram_size - 1)]):
        full_tail = phrase[ti + 1:ti + ngram_size + max_skips]
        if len(full_tail) < ngram_size - 1:
            break
        for tail_phrase in combinations(full_tail, ngram_size-1):
            sub_phrase = [token] + [tail_token for tail_token in tail_phrase]
            assert len(sub_phrase) == ngram_size
            yield sub_phrase
    return None


def extract_sub_phrases(phrase: List[str], max_length: int = 5) -> List[List[str]]:
    """
    Extracts sub-phrases from the given phrase with a maximum length.

    Args:
        phrase (List[str]): The phrase from which to extract sub-phrases.
        max_length (int, optional): The maximum length of each sub-phrase. Defaults to 5.

    Returns:
        List[List[str]]: A list of sub-phrases, each a list of strings representing a
         sub-sequence of the original phrase.
    """
    sub_phrases = []
    for i in range(0, len(phrase) - max_length + 1):
        sub_phrase = phrase[i:i + max_length]
        sub_phrases.append(sub_phrase)
    return sub_phrases


class FormulaSearch:
    """
    A class for searching formulaic language structures in a text corpus.

    This class processes a collection of documents and identifies terms and their co-occurrence patterns.
    It provides methods to calculate term frequencies, co-occurrence frequencies, and filter terms
    based on frequency thresholds. It is useful for formulaic language research, keyword extraction,
    and corpus analysis tasks.

    Attributes:
        full_vocab (Vocabulary): Full vocabulary of terms in the corpus.
        min_freq_vocab (Vocabulary): Vocabulary with terms above the minimum frequency threshold.
        term_freq (Counter): Term frequencies for the entire corpus.
        doc_iterator (Iterable): An iterable that yields documents, which must have a 'words' attribute.
        min_term_freq (int): Minimum frequency for a term to be included in the vocabulary.
        min_cooc_freq (int): Minimum frequency for a co-occurrence to be considered.
        min_neighbour_cooc (int): Minimum number of neighboring co-occurrences required for a term.
        skip_size (int): Number of words to skip when calculating co-occurrences.
        max_min_term_frac (float): Threshold for maximum term frequency fraction.
        cooc_freq (Counter): Co-occurrence frequencies of terms.
        coll_size (int): Size of the corpus in terms of total tokens.
        report (bool): If True, report progress during calculations.
        report_per (int): Report frequency during progress (e.g., every 'report_per' documents).
    """

    def __init__(self, doc_iterator: Iterable,
                 min_term_freq: int = 1,
                 skip_size: int = 4,
                 min_cooc_freq: int = None,
                 min_neighbour_cooc: int = None,
                 max_min_term_frac: float = 0.01,
                 report: bool = False, report_per: int = 1e4):
        """
        Initializes the FormulaSearch object with the given parameters.

        :param doc_iterator: An iterable that yields documents with a 'words' property.
        :type doc_iterator: Iterable
        :param min_term_freq: The frequency threshold for including a term in the vocabulary.
        :type min_term_freq: int
        :param skip_size: The size of the skip window for co-occurrence calculation.
        :type skip_size: int
        :param min_cooc_freq: The frequency threshold for including a co-occurrence in the candidates.
        :type min_cooc_freq: int or None
        :param min_neighbour_cooc: The minimum number of neighboring terms for co-occurrence.
        :type min_neighbour_cooc: int or None
        :param max_min_term_frac: The fraction threshold above which co-occurrences are too common.
        :type max_min_term_frac: float
        :param report: Whether to print progress updates during computation.
        :type report: bool
        :param report_per: Report progress after processing this number of documents.
        :type report_per: int
        """
        self.full_vocab = Vocabulary()
        self.min_freq_vocab = Vocabulary()
        self.term_freq = Counter()
        self.doc_iterator = doc_iterator
        self.min_term_freq = min_term_freq
        self.min_cooc_freq = min_cooc_freq
        self.min_neighbour_cooc = min_neighbour_cooc
        self.skip_size = skip_size
        self.max_min_term_frac = max_min_term_frac
        self.cooc_freq = Counter()
        self.coll_size = 0
        self.cooc_freq = Counter()
        self.report = report
        self.report_per = report_per
        self.calculate_term_frequencies()
        self.make_min_freq_vocabulary()
        if min_cooc_freq is not None:
            self.calculate_co_occurrence_frequencies()
        else:
            print('WARNING: No value passed for min_cooc_freq, skipping co-occurrence calculations.')

    def term2id(self, term: Union[str, Token]):
        """Map a term to its identifier in the vocabulary (or None if it's not in there).

        :param term: The term to map to its identifier in the vocabulary.
        :type term: Union[str, Token]
        :return: The identifier of the given term
        :rtype: int
        """
        return self.full_vocab.term2id(term)

    def id2term(self, term_id: int):
        """Map an identifier to its term in the vocabulary (or None if it's not in there).

        :param term: The identifier of a vocabuluary term
        :type term: int
        :return: The term of a given identifier.
        :rtype: Union[str, Token]
        """
        return self.full_vocab.id2term(term_id)

    def tf(self, term: Union[str, Token]) -> int:
        """
        Returns the frequency of a term in the corpus.

        :param term: The term whose frequency to look up. Can be a string or a Token object.
        :type term: Union[str, Token]
        :return: The frequency of the term in the corpus, or 0 if the term does not occur.
        :rtype: int
        """
        term_id = self.full_vocab.term2id(term)
        return self.term_freq[term_id] if term_id in self.term_freq else 0

    def calculate_term_frequencies(self):
        """
        Iterates over documents in the corpus to calculate term frequencies.

        This method updates the `term_freq` counter with the frequency of terms found
        in the corpus. If the `report` flag is set, progress is printed after every
        `report_per` documents processed.
        """
        print('1. Iterating over sentences to calculate term frequencies')
        self.term_freq = Counter()
        di = 0
        for di, doc in enumerate(self.doc_iterator):
            term_ids = [self.full_vocab.index_term(token) for token in doc]
            # print(f"doc: {doc}")
            # print(f"term_ids: {term_ids}")
            if self.report is True and self.report_per:
                if (di + 1) % self.report_per == 0:
                    print(f"{di + 1} docs processed\tvocab size: {len(self.full_vocab)}"
                          f"\tterms: {len(self.term_freq)}")
            self.term_freq.update(term_ids)
        if self.report is True and self.report_per:
            print(f"{di + 1} docs processed\tvocab size: {len(self.full_vocab)}"
                  f"\tterms: {len(self.term_freq)}")

    def calculate_co_occurrence_frequencies(self, skip_size: int = None, report: bool = None,
                                            report_per: int = None):
        """
        Iterates over the documents in the corpus to calculate co-occurrence frequencies.

        This method computes the co-occurrence frequencies of terms and stores them in
        the `cooc_freq` counter. If the `report` flag is set, it prints the size of
        the co-occurrence index.
        """
        print('2. Iterating over sentences to calculate the co-occurrence frequencies')
        if skip_size is None:
            skip_size = self.skip_size
        if report is None:
            report = self.report
        if report_per is None:
            report_per = self.report_per
        self.cooc_freq = make_cooc_freq(self.doc_iterator, self.min_freq_vocab,
                                        skip_size=skip_size, report=report,
                                        report_per=report_per)
        print(f'    co-occurrence index size: {len(self.cooc_freq):,}')

    def make_min_freq_vocabulary(self, min_term_freq: int = None) -> None:
        """
        Creates a vocabulary with terms that meet the minimum frequency threshold.

        This method filters out terms that occur less frequently than the specified
        `min_term_freq` threshold and stores the remaining terms in `min_freq_vocab`.

        :param min_term_freq: The minimum frequency threshold for including terms.
                               If not provided, uses the value set during initialization.
        :type min_term_freq: int
        """
        if min_term_freq is None:
            min_term_freq = self.min_term_freq
        # f'{value:,}'
        print(f'    full collection size (tokens): {sum(self.term_freq.values()):,}')
        print(f'    full lexicon size (types): {len(self.term_freq):,}')
        print(f'    minimum term frequency: {min_term_freq:,}')
        min_freq_term_ids = [term_id for term_id in self.term_freq if self.term_freq[term_id] >= min_term_freq]
        self.min_freq_vocab = make_selected_vocab(self.full_vocab, selected_ids=min_freq_term_ids)
        print(f'    minimum frequency lexicon size: {len(self.min_freq_vocab):,}')
        self.coll_size = sum(self.term_freq.values())

    def _get_selected_terms(self, doc: Union[Doc, List[str], List[Token]],
                            min_cooc_freq: int = None,
                            min_neighbour_cooc: int = None) -> List[Union[str, None]]:
        """
        Selects terms from a document that meet the co-occurrence frequency criteria.

        This method filters out terms that don't meet the co-occurrence frequency
        thresholds or the required number of neighboring co-occurrences.

        :param doc: The document whose terms are to be filtered. Document can be a list of string,
            fuzzy-search `Token` objects or a fuzzy-search `Doc`.
        :type doc: Union[Doc, List[str], List[Token]]
        :param min_cooc_freq: The minimum frequency of co-occurrences required for a term to be considered.
        :type min_cooc_freq: int or None
        :param min_neighbour_cooc: The minimum number of neighboring co-occurrences required for a term to be kept.
        :type min_neighbour_cooc: int or None
        :return: A list of terms that meet the co-occurrence and neighbor criteria, with `None` for terms that do not.
        :rtype: List[Union[str, None]]
        """
        # print(f"_get_selected_terms received - ")
        # print("    token types:", [type(t) for t in doc])
        # print("    token terms:", [t for t in doc])
        seq_ids = [self.min_freq_vocab.term2id(t) for t in doc]
        if isinstance(doc, Doc):
            doc = [t for t in doc.normalized]
        seq = [t if t in self.min_freq_vocab else None for t in doc]
        # print(f"seq_ids: {seq_ids}")
        # print(f"seq: {seq}")
        selected = []
        for ti, term1 in enumerate(seq):
            id1 = seq_ids[ti]
            if self.term_freq[id1] < min_cooc_freq:
                selected.append(None)
                continue
            terms = []
            own_index, context_terms, context_ids = get_context(ti, seq, seq_ids)
            for i in range(len(context_terms)):
                if i == own_index:
                    continue
                term2 = context_terms[i]
                id2 = context_ids[i]
                if term2 is None:
                    continue
                if i < own_index:
                    if self.cooc_freq[(id2, id1)] < min_cooc_freq:
                        continue
                else:
                    if self.cooc_freq[(id1, id2)] < min_cooc_freq:
                        continue
                terms.append(term2)
            selected.append(term1 if len(terms) >= min_neighbour_cooc else None)
        # print(f"selected: {selected}")
        return selected

    def _iter_get_doc_and_selected_terms(self, min_cooc_freq: int = None,
                                         min_neighbour_cooc: int = None,
                                         max_docs: int = None) -> Generator[dict, None, None]:
        """
        Iterates over documents, yielding the document and its selected terms based on co-occurrence criteria.

        This method iterates through the document iterator, selects terms from each document
        that meet the specified co-occurrence frequency and neighbor co-occurrence criteria,
        and yields a dictionary with the document and the corresponding selected terms.

        :param min_cooc_freq: The minimum frequency of co-occurrences required for a term to be considered.
        :type min_cooc_freq: int or None
        :param min_neighbour_cooc: The minimum number of neighboring co-occurrences required for a term to be kept.
        :type min_neighbour_cooc: int or None
        :param max_docs: The maximum number of documents to process. If None, processes all documents.
        :type max_docs: int or None
        :return: A generator that yields dictionaries containing the document and its selected terms.
        :rtype: Generator[dict, None, None]
        :raises ValueError: If `min_cooc_freq` or `min_neighbour_cooc` is not provided.
        """
        if min_cooc_freq is None:
            min_cooc_freq = self.min_cooc_freq
        if min_cooc_freq is None:
            raise ValueError('No min_cooc_freq set')
        if min_neighbour_cooc is None:
            min_neighbour_cooc = self.min_neighbour_cooc
        if min_neighbour_cooc is None:
            raise ValueError('No min_neighbour_cooc set')
        print('Minimum co-occurrence frequency:', min_cooc_freq)
        for si, doc in enumerate(self.doc_iterator):
            if (si + 1) % 100000 == 0:
                if self.report:
                    print(si + 1, 'sentences processed')
            if max_docs is not None and si >= max_docs:
                break
            # if isinstance(doc, list):
            #     doc = Doc(text=None, doc_id=f"doc_{si+1}", tokens=doc)
            yield {
                'doc': doc,
                'selected': self._get_selected_terms(doc, min_cooc_freq=min_cooc_freq,
                                                     min_neighbour_cooc=min_neighbour_cooc)
            }

    def _get_extract_function(self, phrase_type: str):
        """
        Returns the appropriate extraction function based on the phrase type.

        This helper method maps a given `phrase_type` to an extraction function that handles
        the extraction of the corresponding type of phrases from selected terms.

        :param phrase_type: The type of phrase to extract. Can be 'sub_phrases' or 'long_phrases'.
        :type phrase_type: str
        :return: The corresponding function to extract the specified type of phrase.
        :rtype: function
        :raises ValueError: If an invalid `phrase_type` is provided.
        """
        type_extract_func = {
            'sub_phrases': self._extract_sub_phrases_from_selected,
            'long_phrases': self._extract_long_phrases_from_selected
        }
        if phrase_type not in type_extract_func:
            accepted_types = "\'sub_phrases\', \'long_phrases\'"
            raise ValueError(f'invalid phrase_type "{phrase_type}", must be in {accepted_types}')
        else:
            return type_extract_func[phrase_type]

    def extract_phrases(self, phrase_type: str, min_cooc_freq: int = None,
                        min_neighbour_cooc: int = 1, max_docs: int = None,
                        *args, **kwargs) -> Generator[CandidatePhraseMatch, None, None]:
        """
        Extracts candidate phrases of a specified type from a document corpus.

        This method extracts candidate phrases (either sub-phrases or long-phrases) from the selected terms
        of each document in the corpus. The selected terms are those that meet the co-occurrence frequency
        and neighbor co-occurrence thresholds.

        :param phrase_type: The type of phrases to extract. Can be either 'sub_phrases' or 'long_phrases'.
        :type phrase_type: str
        :param min_cooc_freq: The minimum frequency of co-occurrences required for a term to be considered.
        :type min_cooc_freq: int or None
        :param min_neighbour_cooc: The minimum number of neighboring co-occurrences required for a term to be kept.
        :type min_neighbour_cooc: int
        :param max_docs: The maximum number of documents to process. If None, processes all documents.
        :type max_docs: int or None
        :param args: Additional arguments passed to the phrase extraction function.
        :param kwargs: Additional keyword arguments passed to the phrase extraction function.
        :return: A generator that yields `CandidatePhraseMatch` objects for each extracted phrase.
        :rtype: Generator[CandidatePhraseMatch, None, None]
        """
        if min_cooc_freq is None:
            min_cooc_freq = self.min_cooc_freq
        extract_func = self._get_extract_function(phrase_type)
        for doc_selected in self._iter_get_doc_and_selected_terms(min_cooc_freq=min_cooc_freq,
                                                                  min_neighbour_cooc=min_neighbour_cooc,
                                                                  max_docs=max_docs):
            for candidate_phrase_match in extract_func(doc=doc_selected['doc'],
                                                       selected=doc_selected['selected'],
                                                       *args, **kwargs):
                yield candidate_phrase_match

    def extract_phrases_from_docs(self, *args, **kwargs):
        """Left for Backward-compatibility."""
        print("WARNING - 'extract_phrases_from_docs' has been renamed to 'extract_phrases."
              " The old function name maps to 'extract_phrases' for backward compatibility.")
        return self.extract_phrases(*args, **kwargs)

    def _extract_sub_phrases_from_selected(self, doc: Doc, selected: List[Union[str, None]],
                                           min_phrase_length: int = 3,
                                           max_phrase_length: int = 5,
                                           max_variables: int = 0,
                                           max_skips: int = 0) -> Generator[CandidatePhraseMatch, None, None]:
        """
        Extracts sub-phrases from the selected terms in a document.

        This method extracts sub-phrases of varying lengths (within the specified range)
        from the selected terms, considering the possibility of skipped terms.
        Sub-phrases are generated only if they meet the minimum phrase length and
        frequency criteria.

        :param doc: The document from which to extract sub-phrases.
        :type doc: Doc
        :param selected: The list of selected terms from the document.
        :type selected: List[Union[str, None]]
        :param min_phrase_length: The minimum length of a sub-phrase to be considered.
        :type min_phrase_length: int
        :param max_phrase_length: The maximum length of a sub-phrase to be considered.
        :type max_phrase_length: int
        :param max_variables: The maximum number of variables (None) allowed in a sub-phrase.
        :type max_variables: int
        :param max_skips: The maximum number of skips allowed between terms in a sub-phrase.
        :type max_skips: int
        :return: A generator yielding candidate sub-phrases that meet the criteria.
        :rtype: Generator[CandidatePhraseMatch, None, None]
        """
        phrase = []
        word_start = 0
        for ti, term in enumerate(selected):
            if term is None and phrase.count(None) == max_variables:
                if len(phrase) > min_phrase_length:
                    for candidate_phrase_match in self.make_sub_phrase_matches(phrase, word_start,
                                                                               max_phrase_length=max_phrase_length,
                                                                               max_skips=max_skips,
                                                                               doc=doc):
                        yield candidate_phrase_match
                phrase = []
                continue
            if term is None and len(phrase) == 0:
                continue
            elif len(phrase) == 0:
                word_start = ti
            phrase.append(term)
        if len(phrase) > min_phrase_length:
            candidate_gen = self.make_sub_phrase_matches(phrase, word_start, doc=doc,
                                                         max_phrase_length=max_phrase_length,
                                                         max_skips=max_skips)
            for candidate_phrase_match in candidate_gen:
                yield candidate_phrase_match

    def make_sub_phrase_matches(self, phrase, word_start: int,
                                max_phrase_length: int,
                                doc: Doc, max_skips: int = 0):
        """
        Creates candidate sub-phrase matches from a phrase, considering the term frequency
        and co-occurrence thresholds.

        This method generates sub-phrases from the given phrase, ensuring that the frequency
        of the terms in the sub-phrase meets the minimum criteria, and that the phrase
        respects the maximum phrase length and skip constraints.

        :param phrase: The phrase to generate sub-phrases from.
        :type phrase: list
        :param word_start: The starting position of the phrase in the document.
        :type word_start: int
        :param max_phrase_length: The maximum length of sub-phrases.
        :type max_phrase_length: int
        :param doc: The document that contains the phrase.
        :type doc: Doc
        :param max_skips: The maximum number of skips allowed in sub-phrases.
        :type max_skips: int
        :return: A generator yielding candidate phrase matches for the sub-phrases.
        :rtype: Generator[CandidatePhraseMatch, None, None]
        """
        min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase])
        if min_term_frac < self.max_min_term_frac:
            sub_phrases = extract_sub_phrases_with_skips(phrase, ngram_size=max_phrase_length,
                                                         max_skips=max_skips)
            for si, sub_phrase in enumerate(sub_phrases):
                sub_start = word_start + si
                sub_phrase = make_candidate_phrase(sub_phrase)
                if isinstance(doc, Doc):
                    variable_match = doc.normalized[sub_start: sub_start + len(phrase)]
                else:
                    variable_match = doc[sub_start: sub_start + len(phrase)]
                yield CandidatePhraseMatch(sub_phrase, word_start=sub_start,
                                           variable_match=variable_match, doc=doc)
        else:
            print(f'minimum term fraction {min_term_frac} is higher than '
                  f'max_min_term_frac {self.max_min_term_frac} for phrase {phrase}')

    def _passes_freq_thresholds(self, phrase: list, min_phrase_length: int) -> bool:
        """
        Checks whether a phrase passes the frequency thresholds.

        This method checks if the phrase meets the required length and frequency criteria
        based on the term frequency in the corpus.

        :param phrase: The phrase to check.
        :type phrase: list
        :param min_phrase_length: The minimum number of terms required for a valid phrase.
        :type min_phrase_length: int
        :return: True if the phrase passes the frequency threshold, False otherwise.
        :rtype: bool
        """
        if len(phrase) - phrase.count(None) >= min_phrase_length:
            min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase])
            return min_term_frac < self.max_min_term_frac
        else:
            return False

    def _extract_long_phrases_from_selected(self, doc: Doc, selected: List[Union[str, None]],
                                            min_phrase_length: int = 3,
                                            max_variables: int = 0) -> Generator[CandidatePhraseMatch, None, None]:
        """
        Extracts long phrases from the selected terms in a document.

        This method extracts long phrases of terms from the selected terms. It allows for
        a specified number of variables (None) in the phrase and checks that the phrase
        meets the frequency threshold before yielding it as a candidate phrase.

        :param doc: The document from which to extract long phrases.
        :type doc: Doc
        :param selected: The list of selected terms from the document.
        :type selected: List[Union[str, None]]
        :param min_phrase_length: The minimum length of a phrase to be considered.
        :type min_phrase_length: int
        :param max_variables: The maximum number of variables (None) allowed in a phrase.
        :type max_variables: int
        :return: A generator yielding candidate long phrases that meet the criteria.
        :rtype: Generator[CandidatePhraseMatch, None, None]
        """
        phrase = []
        phrase_start = 0
        for ti, term in enumerate(selected):
            if term is None and phrase.count(None) == max_variables:
                if self._passes_freq_thresholds(phrase, min_phrase_length):
                    yield make_candidate_phrase_match(phrase, phrase_start, doc)
                phrase = []
            if term is None and len(phrase) == 0:
                continue
            elif len(phrase) == 0:
                phrase_start = ti
            phrase.append(term)
        if self._passes_freq_thresholds(phrase, min_phrase_length):
            yield make_candidate_phrase_match(phrase, phrase_start, doc)

    def _extract_candidate_phrases(self, min_length: int = 3, max_length: int = 5,
                                   min_cooc_freq: int = None,
                                   max_docs: int = None) -> Tuple[int, CandidatePhraseMatch]:
        """
        Extracts candidate phrases of various lengths from the selected terms in a document.

        This method generates candidate phrases by considering selected terms that meet
        the frequency and co-occurrence criteria. The length of the candidate phrases is
        bounded by `min_length` and `max_length`.

        :param min_length: The minimum length of a candidate phrase.
        :type min_length: int
        :param max_length: The maximum length of a candidate phrase.
        :type max_length: int
        :param min_cooc_freq: The minimum co-occurrence frequency for terms to be considered.
        :type min_cooc_freq: int or None
        :param max_docs: The maximum number of documents to process. If None, processes all documents.
        :type max_docs: int or None
        :return: A tuple containing the document ID and the candidate phrase match.
        :rtype: Tuple[int, CandidatePhraseMatch]
        """
        for doc_selected in self._iter_get_doc_and_selected_terms(min_cooc_freq=min_cooc_freq, max_docs=max_docs):
            selected = doc_selected['selected']
            doc = doc_selected['doc']
            for ti, term in enumerate(selected[:-max_length + 1]):
                if term is None:
                    continue
                phrase = []
                phrase_start = ti
                for i in range(ti, ti + max_length):
                    phrase.append(selected[i])
                if phrase.count(None) >= min_length:
                    continue
                min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase if t is not None])
                if min_term_frac < self.max_min_term_frac:
                    candidate_phrase = make_candidate_phrase(phrase)
                    if isinstance(doc, Doc):
                        variable_match = doc.normalized[phrase_start: phrase_start+len(phrase)]
                    else:
                        variable_match = doc[phrase_start: phrase_start+len(phrase)]
                    yield CandidatePhraseMatch(candidate_phrase, word_start=phrase_start,
                                               variable_match=variable_match)

    def index_candidate_docs(self, candidate_phrases: List[Union[str, List[str]]],
                             min_cooc_freq: int = None, **kwargs) -> Dict[str, List[str]]:
        """
        Indexes documents by candidate phrases, mapping each candidate phrase to the documents
        in which it appears.

        This method iterates through documents and extracts long phrases from the selected terms.
        If a phrase from the `candidate_phrases` list is found in the document, the document ID
        is added to the index for that phrase. It returns a dictionary mapping each phrase to the
        list of document IDs where the phrase appears.

        :param candidate_phrases: A list of candidate phrases or a list of phrases represented as strings
                                   that need to be indexed.
        :type candidate_phrases: List[Union[str, List[str]]]
        :param min_cooc_freq: The minimum co-occurrence frequency for terms to be selected. If not provided,
                               the default value from the instance is used.
        :type min_cooc_freq: int, optional
        :param kwargs: Additional arguments passed to the phrase extraction functions.
        :return: A dictionary where keys are candidate phrases, and values are lists of document IDs
                 containing those phrases.
        :rtype: Dict[str, List[str]]
        """
        if min_cooc_freq is None:
            min_cooc_freq = self.min_cooc_freq
        candidate_doc_index = defaultdict(list)
        candidate_phrases = transform_candidates_to_strings(candidate_phrases)
        for doc in self.doc_iterator:
            selected = self._get_selected_terms(doc, min_cooc_freq=min_cooc_freq)
            for phrase in self._extract_long_phrases_from_selected(doc, selected, **kwargs):
                if phrase not in candidate_phrases:
                    continue
                candidate_doc_index[phrase].append(doc['id'])
        return candidate_doc_index

    def extract_candidate_variables(self, phrase_type: str, candidates: List[Union[str, List[str]]],
                                    min_cooc_freq: int = None, max_docs: int = None, *args, **kwargs):
        """
        Extracts candidate variables from documents based on the given candidate phrases.

        This method iterates through documents, extracting candidate phrases of the specified
        `phrase_type` (either long or sub-phrases). It checks whether the phrase matches any of the
        provided candidate phrases. If a match is found, the corresponding variable (the portion
        of the document text corresponding to the phrase) is yielded along with the associated
        `CandidatePhraseMatch`.

        :param phrase_type: The type of phrase to extract, either 'long_phrases' or 'sub_phrases'.
        :type phrase_type: str
        :param candidates: A list of candidate phrases or strings to search for in the documents.
        :type candidates: List[Union[str, List[str]]]
        :param min_cooc_freq: The minimum co-occurrence frequency for terms to be selected.
                              If not provided,
                              the default value from the instance is used.
        :type min_cooc_freq: int, optional
        :param max_docs: The maximum number of documents to process. If not provided, all
                         documents will be processed.
        :type max_docs: int, optional
        :param args: Additional positional arguments passed to the extraction function.
        :param kwargs: Additional keyword arguments passed to the extraction function.
        :return: A generator that yields tuples of (variable_match, candidate_phrase_match)
                 for each matching candidate phrase.
        :rtype: Generator[Tuple[Union[str, List[str]], CandidatePhraseMatch], None, None]
        :raises ValueError: If `min_cooc_freq` is not provided and it is not set in the instance.
        """
        if min_cooc_freq is None:
            if self.min_cooc_freq is None:
                raise ValueError(f'no min_cooc_freq passed, nor set in {self.__class__.__name__} instance')
            min_cooc_freq = self.min_cooc_freq
        candidate_set = {t for t in transform_candidates_to_strings(candidates)}
        extract_func = self._get_extract_function(phrase_type)
        for di, doc in enumerate(self.doc_iterator):
            if (di+1) >= max_docs:
                break
            selected = self._get_selected_terms(doc, min_cooc_freq=min_cooc_freq)
            for candidate_pm in extract_func(doc=doc, selected=selected, *args, **kwargs):
                if candidate_pm.candidate_phrase.phrase_string in candidate_set:
                    if isinstance(doc, Doc):
                        variable_match = doc.normalized[candidate_pm.word_start: candidate_pm.word_end]
                    else:
                        variable_match = doc[candidate_pm.word_start: candidate_pm.word_end]
                    yield variable_match, candidate_pm
