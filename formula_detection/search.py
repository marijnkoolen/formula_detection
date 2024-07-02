from collections import Counter
from collections import defaultdict
from itertools import combinations
from typing import Dict, Generator, Iterable, List, Tuple, Union

from fuzzy_search.tokenization.token import Doc
from formula_detection.vocabulary import Vocabulary, make_selected_vocab
from formula_detection.cooccurrence import make_cooc_freq, get_context


class CandidatePhrase:

    def __init__(self, phrase: Union[str, List[str]]):
        self.phrase_string = transform_candidate_to_string(phrase)
        self.phrase_list = transform_candidate_to_list(phrase)

    def __len__(self):
        return len(self.phrase_list)

    def __repr__(self):
        return f'({self.__class__.__name__}={self.phrase_string})'


def get_variable_terms_from_match(candidate_phrase: CandidatePhrase,
                                  variable_match: List[str]) -> List[str]:
    variable_terms = []
    for ti, term in enumerate(candidate_phrase.phrase_list):
        if term == '<VAR>':
            variable_terms.append(variable_match[ti])
    return variable_terms


class CandidatePhraseMatch:

    def __init__(self, candidate_phrase: CandidatePhrase, char_start: int = None,
                 word_start: int = None, variable_match: List[str] = None,
                 doc: Doc = None):
        self.candidate_phrase = candidate_phrase
        self.char_start = None if char_start is None else char_start
        self.char_end = None if char_start is None else char_start + len(candidate_phrase.phrase_string)
        self.word_start = None if word_start is None else word_start
        self.word_end = None if word_start is None else word_start + len(candidate_phrase.phrase_list)
        self.variable_match = None if variable_match is None else variable_match
        self.variable_terms = []
        self.doc_id = doc.id if doc is not None else None
        if self.variable_match:
            self.variable_terms = get_variable_terms_from_match(candidate_phrase, variable_match)

    def __len__(self):
        return len(self.candidate_phrase.phrase_string)

    def __repr__(self):
        return f'({self.__class__.__name__}, char_start={self.char_start}, ' \
               f'word_start={self.word_start}, phrase={self.candidate_phrase.phrase_string})'


def make_candidate_phrase(phrase: Union[str, List[str]]) -> CandidatePhrase:
    phrase = transform_candidate_to_list(phrase)
    phrase = ' '.join([t if t else '<VAR>' for t in phrase])
    return CandidatePhrase(phrase)


def extract_sub_phrases_with_skips(phrase, ngram_size, max_skips):
    for ti, token in enumerate(phrase[:-(ngram_size - 1)]):
        # print('token:', token)
        full_tail = phrase[ti + 1:ti + ngram_size + max_skips]
        # print('full_tail:', full_tail)
        if len(full_tail) < ngram_size - 1:
            break
        for tail_phrase in combinations(full_tail, ngram_size-1):
            # print('tail_phrase:', tail_phrase)
            sub_phrase = [token] + [tail_token for tail_token in tail_phrase]
            assert len(sub_phrase) == ngram_size
            yield sub_phrase
    return None


def extract_sub_phrases(phrase: List[str],
                        max_length: int = 5) -> List[List[str]]:
    sub_phrases = []
    for i in range(0, len(phrase) - max_length + 1):
        sub_phrase = phrase[i:i + max_length]
        sub_phrases.append(sub_phrase)
    return sub_phrases


def make_candidate_phrase_match(phrase, phrase_start, doc: Doc):
    candidate_phrase = make_candidate_phrase(phrase)
    variable_match = doc.normalized[phrase_start: phrase_start + len(phrase)]
    return CandidatePhraseMatch(candidate_phrase, word_start=phrase_start,
                                variable_match=variable_match, doc=doc)


class FormulaSearch:

    def __init__(self, doc_iterator: Iterable,
                 min_term_freq: int = 1,
                 skip_size: int = 4,
                 min_cooc_freq: int = None,
                 max_min_term_frac: float = 0.01,
                 report: bool = False, report_per: int = 1e4):
        """Template Language Use detector class.

        :param doc_iterator: an iterable that yields sent objects with a 'words' property
        :type doc_iterator: Iterable
        :param min_term_freq: the frequency threshold for including a term in the vocabulary
        :type min_term_freq: int
        :param min_cooc_freq: the frequency threshold for including a cooccurrence in the candidate
        :type min_cooc_freq: int
        :param max_min_term_frac: the fraction threshold above which co-occurrence are considered
        too common to be of interest."""
        self.full_vocab = Vocabulary()
        self.min_freq_vocab = Vocabulary()
        self.term_freq = Counter()
        self.doc_iterator = doc_iterator
        self.min_term_freq = min_term_freq
        self.min_cooc_freq = min_cooc_freq
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
            print('No value passed for min_cooc_freq, skipping co-occurrence calculations.')

    def tf(self, term: str):
        term_id = self.full_vocab.term_id[term]
        return self.term_freq[term_id] if term_id in self.term_freq else 0

    def calculate_co_occurrence_frequencies(self):
        print('Iterating over sentences to calculate the co-occurrence frequencies')
        self.cooc_freq = make_cooc_freq(self.doc_iterator, self.min_freq_vocab,
                                        skip_size=self.skip_size, report=self.report,
                                        report_per=self.report_per)
        print(f'co-occurence index size: {len(self.cooc_freq)}')

    def calculate_term_frequencies(self):
        print('Iterating over sentences to calculate term frequencies')
        self.term_freq = Counter()
        di = 0
        for di, doc in enumerate(self.doc_iterator):
            term_ids = [self.full_vocab.index_term(token) for token in doc]
            if self.report is True and self.report_per:
                if (di+1) % self.report_per == 0:
                    print(f"{di+1} docs processed\tvocab size: {len(self.full_vocab.term_id)}"
                          f"\tterms: {len(self.term_freq)}")
            self.term_freq.update(term_ids)
        # print(di+1, doc.id)
        if self.report is True and self.report_per:
            print(f"{di+1} docs processed\tvocab size: {len(self.full_vocab.term_id)}"
                  f"\tterms: {len(self.term_freq)}")

    def make_min_freq_vocabulary(self, min_term_freq: int = None) -> None:
        if min_term_freq is None:
            min_term_freq = self.min_term_freq
        print('full collection size (tokens):', sum(self.term_freq.values()))
        print('full lexicon size (types):', len(self.term_freq))
        print('minimum term frequency:', min_term_freq)
        min_freq_term_ids = [term_id for term_id in self.term_freq if self.term_freq[term_id] >= min_term_freq]
        self.min_freq_vocab = make_selected_vocab(self.full_vocab, selected_ids=min_freq_term_ids)
        print('minimum frequency lexicon size:', len(self.min_freq_vocab))
        self.coll_size = sum(self.term_freq.values())

    def _get_selected_terms(self, doc: Doc,
                            min_cooc_freq: int = None,
                            min_neighbour_cooc: int = 1) -> List[Union[str, None]]:
        seq_ids = [self.min_freq_vocab.term2id(t) for t in doc]
        seq = [t if t in self.min_freq_vocab.term_id else None for t in doc.normalized]
        # print('_get_selected_terms - tokens:', doc.normalized)
        # print('_get_selected_terms - seq:', seq)
        # print('_get_selected_terms - seq_ids:', seq_ids)
        selected = []
        for ti, term1 in enumerate(seq):
            id1 = seq_ids[ti]
            if self.term_freq[id1] < min_cooc_freq:
                selected.append(None)
                continue
            terms = []
            own_index, context_terms, context_ids = get_context(ti, seq, seq_ids)
            # print('term1:', term1, 'seq index:', ti, 'own_index:', own_index, context_terms, context_ids)
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
                    # print('\tterm2:', term2, 'cooc_freq:', self.cooc_freq[(id2, id1)])
                else:
                    if self.cooc_freq[(id1, id2)] < min_cooc_freq:
                        continue
                    # print('\tterm2:', term2, 'cooc_freq:', self.cooc_freq[(id1, id2)])
                terms.append(term2)
            selected.append(term1 if len(terms) >= min_neighbour_cooc else None)
        # print('selected:', selected)
        return selected

    def _iter_get_doc_and_selected_terms(self, min_cooc_freq: int = None,
                                         min_neighbour_cooc: int = None,
                                         max_docs: int = None) -> Generator[dict, None, None]:
        if min_cooc_freq is None:
            min_cooc_freq = self.min_cooc_freq
        if min_cooc_freq is None:
            raise ValueError('No min_cooc_freq set')
        print('Minimum co-occurrence frequency:', min_cooc_freq)
        for si, doc in enumerate(self.doc_iterator):
            if (si+1) % 100000 == 0:
                print(si+1, 'sentences processed')
            if max_docs is not None and si >= max_docs:
                break
            if isinstance(doc, dict) and 'doc_id' not in doc:
                doc['doc_id'] = si
                doc['doc_id_type'] = 'doc_num'
            elif isinstance(doc, list):
                doc = {'doc_id': si, 'doc_id_type': 'doc_num', 'words': doc}
            yield {
                'doc': doc,
                'selected': self._get_selected_terms(doc, min_cooc_freq=min_cooc_freq,
                                                     min_neighbour_cooc=min_neighbour_cooc)
            }

    def _get_extract_function(self, phrase_type: str):
        type_extract_func = {
            'sub_phrases': self._extract_sub_phrases_from_selected,
            'long_phrases': self._extract_long_phrases_from_selected
        }
        if phrase_type not in type_extract_func:
            accepted_types = "\' \'".join(type_extract_func.keys())
            raise ValueError(f'invalid phrase_type "{phrase_type}", must be in {accepted_types}')
        else:
            return type_extract_func[phrase_type]

    def extract_phrases_from_docs(self, phrase_type: str, min_cooc_freq: int = None,
                                  min_neighbour_cooc: int = None, max_docs: int = None,
                                  *args, **kwargs) -> Generator[CandidatePhraseMatch, None, None]:
        if min_cooc_freq is None:
            min_cooc_freq = self.min_cooc_freq
        extract_func = self._get_extract_function(phrase_type)
        for doc_selected in self._iter_get_doc_and_selected_terms(min_cooc_freq=min_cooc_freq,
                                                                  min_neighbour_cooc=min_neighbour_cooc,
                                                                  max_docs=max_docs):
            # print('extract_phrases_from_doc - doc_selected:', doc_selected)
            for candidate_phrase_match in extract_func(doc=doc_selected['doc'],
                                                       selected=doc_selected['selected'],
                                                       *args, **kwargs):
                yield candidate_phrase_match

    def _extract_sub_phrases_from_selected(self, doc: Doc, selected: List[Union[str, None]],
                                           min_phrase_length: int = 3,
                                           max_phrase_length: int = 5,
                                           max_variables: int = 0,
                                           max_skips: int = 0) -> Generator[CandidatePhraseMatch, None, None]:
        # print('_extract_sub_phrases_from_selected - max_skips:', max_skips)
        phrase = []
        word_start = 0
        for ti, term in enumerate(selected):
            # print(ti, term)
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
            # print(phrase)
        if len(phrase) > min_phrase_length:
            candidate_gen = self.make_sub_phrase_matches(phrase, word_start, doc=doc,
                                                         max_phrase_length=max_phrase_length,
                                                         max_skips=max_skips)
            for candidate_phrase_match in candidate_gen:
                yield candidate_phrase_match

    def make_sub_phrase_matches(self, phrase, word_start: int,
                                max_phrase_length: int,
                                doc: Doc, max_skips: int = 0):
        # print('make_sub_phrase_matches - max_skips:', max_skips)
        min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase])
        # print('make_sub_phrase_matches - phrase:', phrase)
        # print('make_sub_phrase_matches - min_term_frac:', min_term_frac)
        if min_term_frac < self.max_min_term_frac:
            sub_phrases = extract_sub_phrases_with_skips(phrase, ngram_size=max_phrase_length,
                                                         max_skips=max_skips)
            # print('make_sub_phrase_matches - num sub_phrases:', len(sub_phrases))
            for si, sub_phrase in enumerate(sub_phrases):
                # print('make_sub_phrase_matches - sub_phrase:', sub_phrase)
                sub_start = word_start + si
                sub_phrase = make_candidate_phrase(sub_phrase)
                variable_match = doc.normalized[sub_start: sub_start + len(phrase)]
                yield CandidatePhraseMatch(sub_phrase, word_start=sub_start,
                                           variable_match=variable_match, doc=doc)
        else:
            print(f'minimum term fraction {min_term_frac} is higher than '
                  f'max_min_term_frac {self.max_min_term_frac} for phrase {phrase}')

    def _passes_freq_thresholds(self, phrase: list, min_phrase_length: int) -> bool:
        if len(phrase) - phrase.count(None) >= min_phrase_length:
            min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase])
            return min_term_frac < self.max_min_term_frac
        else:
            return False

    def _extract_long_phrases_from_selected(self, doc: Doc, selected: List[Union[str, None]],
                                            min_phrase_length: int = 3,
                                            max_variables: int = 0) -> Generator[CandidatePhraseMatch, None, None]:
        phrase = []
        phrase_start = 0
        # ti = 0
        # while ti < len(selected):
        for ti, term in enumerate(selected):
            # term = selected[ti]
            # ti += 1
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
                    variable_match = doc.normalized[phrase_start: phrase_start+len(phrase)]
                    yield CandidatePhraseMatch(candidate_phrase, word_start=phrase_start,
                                               variable_match=variable_match)

        return None

    def index_candidate_docs(self, candidate_phrases: List[Union[str, List[str]]],
                             min_cooc_freq: int = None, **kwargs) -> Dict[str, List[str]]:
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
            for candidate_phrase_match in extract_func(doc=doc,
                                                       selected=selected,
                                                       *args, **kwargs):
                if candidate_phrase_match.candidate_phrase.phrase_string in candidate_set:
                    # print(candidate_phrase_match)
                    # print(candidate_phrase_match.word_start)
                    # print(candidate_phrase_match.word_end)
                    variable_match = doc.normalized[candidate_phrase_match.word_start: candidate_phrase_match.word_end]
                    yield variable_match, candidate_phrase_match


def transform_candidate_to_list(candidate: Union[str, List[str]]) -> List[str]:
    if isinstance(candidate, str):
        return candidate.split(' ')
    elif isinstance(candidate, list) is False:
        raise TypeError(f'candidate must be str or list of str, not {type(candidate)}')
    else:
        return candidate


def transform_candidate_to_string(candidate: Union[str, List[str]]) -> str:
    if isinstance(candidate, list):
        return ' '.join(candidate)
    elif isinstance(candidate, str) is False:
        raise TypeError(f'candidate must be str or list of str, not {type(candidate)}')
    else:
        return candidate


def transform_candidates_to_lists(candidates: List[Union[str, List[str]]]) -> List[List[str]]:
    return [transform_candidate_to_list(candidate) for candidate in candidates]


def transform_candidates_to_strings(candidates: List[Union[str, List[str]]]) -> List[str]:
    return [transform_candidate_to_string(candidate) for candidate in candidates]
