from typing import Dict, Generator, Iterable, List, Tuple, Union
from collections import Counter
from collections import defaultdict

from hist_text_template.vocabulary import Vocabulary, make_selected_vocab
from hist_text_template.cooccurrence import make_cooc_freq, get_context
from hist_text_template.sentence import get_sent_terms


class TemplatePhrase:

    def __init__(self, phrase: Union[str, List[str]]):
        self.phrase_string = transform_template_to_string(phrase)
        self.phrase_list = transform_template_to_list(phrase)

    def __len__(self):
        return len(self.phrase_list)

    def __repr__(self):
        return f'({self.__class__.__name__}={self.phrase_string})'


def get_variable_terms_from_match(template_phrase: TemplatePhrase,
                                  variable_match: List[str]) -> List[str]:
    variable_terms = []
    for ti, term in enumerate(template_phrase.phrase_list):
        if term == '<VAR>':
            variable_terms.append(variable_match[ti])
    return variable_terms


class TemplatePhraseMatch:

    def __init__(self, template_phrase: TemplatePhrase, char_start: int = None,
                 word_start: int = None, variable_match: List[str] = None,
                 sent: Dict[str, any] = None):
        self.template_phrase = template_phrase
        self.char_start = None if char_start is None else char_start
        self.char_end = None if char_start is None else char_start + len(template_phrase.phrase_string)
        self.word_start = None if word_start is None else word_start
        self.word_end = None if word_start is None else word_start + len(template_phrase.phrase_list)
        self.variable_match = None if variable_match is None else variable_match
        self.variable_terms = []
        self.doc_id = sent['doc_id'] if sent is not None and 'doc_id' in sent else None
        if self.variable_match:
            self.variable_terms = get_variable_terms_from_match(template_phrase, variable_match)

    def __len__(self):
        return len(self.template_phrase.phrase_string)

    def __repr__(self):
        return f'({self.__class__.__name__}, char_start={self.char_start}, ' \
               f'word_start={self.word_start}, phrase={self.template_phrase.phrase_string})'


def make_template_phrase(phrase: Union[str, List[str]]) -> TemplatePhrase:
    phrase = transform_template_to_list(phrase)
    phrase = ' '.join([t if t else '<VAR>' for t in phrase])
    return TemplatePhrase(phrase)


def make_template_phrase_match(phrase: Union[str, List[str]],
                               char_start: int, word_start: int) -> TemplatePhraseMatch:
    template_phrase = make_template_phrase(phrase)
    return TemplatePhraseMatch(template_phrase=template_phrase, char_start=char_start,
                               word_start=word_start)


def extract_sub_phrases(phrase: List[str],
                        max_length: int = 5) -> List[List[str]]:
    sub_phrases = []
    for i in range(0, len(phrase) - max_length + 1):
        sub_phrase = phrase[i:i + max_length]
        sub_phrases.append(sub_phrase)
    return sub_phrases


class TextTemplateSearch:

    def __init__(self, sent_iterator: Iterable,
                 min_term_freq: int = 1,
                 skip_size: int = 4,
                 min_cooc_freq: int = None,
                 max_min_term_frac: float = 0.01):
        """Template Language Use detector class.

        :param sent_iterator: an iterable that yields sent objects with a 'words' property
        :type sent_iterator: Iterable
        :param min_term_freq: the frequency threshold for including a term in the vocabulary
        :type min_term_freq: int
        :param min_cooc_freq: the frequency threshold for including a cooccurrence in the template
        :type min_cooc_freq: int
        :param max_min_term_frac: the fraction threshold above which co-occurrence are considered
        too common to be of interest."""
        self.full_vocab = Vocabulary()
        self.min_freq_vocab = Vocabulary()
        self.term_freq = Counter()
        self.sent_iterator = sent_iterator
        self.min_term_freq = min_term_freq
        self.min_cooc_freq = min_cooc_freq
        self.skip_size = skip_size
        self.max_min_term_frac = max_min_term_frac
        self.cooc_freq = Counter()
        self.coll_size = 0
        self.cooc_freq = Counter()
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
        self.cooc_freq = make_cooc_freq(self.sent_iterator, self.min_freq_vocab,
                                        skip_size=self.skip_size)
        print(f'co-occurence index size: {len(self.cooc_freq)}')

    def calculate_term_frequencies(self):
        print('Iterating over sentences to calculate term frequencies')
        self.term_freq = Counter()
        for si, sent in enumerate(self.sent_iterator):
            terms = get_sent_terms(sent)
            term_ids = [self.full_vocab.add_term(term) for term in terms]
            self.term_freq.update(term_ids)

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

    def _get_selected_terms(self, sent: Union[List[str], Dict[str, any]],
                            min_cooc_freq: int = None) -> List[Union[str, None]]:
        words = get_sent_terms(sent)
        seq_ids = [self.min_freq_vocab.term2id(t) for t in words]
        seq = [t if t in self.min_freq_vocab.term_id else None for t in words]
        # print('words:', words)
        # print('seq:', seq)
        # print('seq_ids:', seq_ids)
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
            selected.append(term1 if len(terms) > 0 else None)
        # print('selected:', selected)
        return selected

    def _iter_get_sent_and_selected_terms(self, min_cooc_freq: int = None,
                                          max_sents: int = None) -> Generator[dict, None, None]:
        if min_cooc_freq is None:
            min_cooc_freq = self.min_cooc_freq
        if min_cooc_freq is None:
            raise ValueError('No min_cooc_freq set')
        print('Minimum co-occurrence frequency:', min_cooc_freq)
        for si, sent in enumerate(self.sent_iterator):
            if (si+1) % 100000 == 0:
                print(si+1, 'sentences processed')
            if max_sents is not None and si >= max_sents:
                break
            yield {
                'sent': sent,
                'selected': self._get_selected_terms(sent, min_cooc_freq=min_cooc_freq)
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

    def extract_phrases_from_sents(self, phrase_type: str, min_cooc_freq: int = None,
                                   max_sents: int = None,
                                   *args, **kwargs) -> Generator[TemplatePhraseMatch, None, None]:
        if min_cooc_freq is None:
            min_cooc_freq = self.min_cooc_freq
        extract_func = self._get_extract_function(phrase_type)
        for sent_selected in self._iter_get_sent_and_selected_terms(min_cooc_freq=min_cooc_freq, max_sents=max_sents):
            for template_phrase_match in extract_func(sent=sent_selected['sent'],
                                                      selected=sent_selected['selected'],
                                                      *args, **kwargs):
                yield template_phrase_match

    def _extract_sub_phrases_from_selected(self, sent: Dict[str, any], selected: List[Union[str, None]],
                                           min_phrase_length: int = 3,
                                           max_phrase_length: int = 5,
                                           max_variables: int = 0) -> Generator[TemplatePhraseMatch, None, None]:
        phrase = []
        word_start = 0
        words = get_sent_terms(sent)
        for ti, term in enumerate(selected):
            # print(ti, term)
            if term is None and phrase.count(None) == max_variables:
                if len(phrase) > min_phrase_length:
                    for template_phrase_match in self.make_sub_phrase_matches(phrase, word_start, words,
                                                                              max_phrase_length=max_phrase_length,
                                                                              sent=sent):
                        yield template_phrase_match
                phrase = []
                continue
            if term is None and len(phrase) == 0:
                continue
            elif len(phrase) == 0:
                word_start = ti
            phrase.append(term)
            # print(phrase)
        if len(phrase) > min_phrase_length:
            for template_phrase_match in self.make_sub_phrase_matches(phrase, word_start, words,
                                                                      max_phrase_length=max_phrase_length,
                                                                      sent=sent):
                yield template_phrase_match

    def make_sub_phrase_matches(self, phrase, word_start: int,
                                words: List[Union[str, None]], max_phrase_length: int,
                                sent: Dict[str, any] = None):
        min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase])
        # print('make_sub_phrase_matches - phrase:', phrase)
        # print('make_sub_phrase_matches - min_term_frac:', min_term_frac)
        if min_term_frac < self.max_min_term_frac:
            sub_phrases = extract_sub_phrases(phrase, max_length=max_phrase_length)
            for si, sub_phrase in enumerate(sub_phrases):
                sub_start = word_start + si
                sub_phrase = make_template_phrase(sub_phrase)
                variable_match = words[sub_start: sub_start + len(phrase)]
                yield TemplatePhraseMatch(sub_phrase, word_start=sub_start,
                                          variable_match=variable_match, sent=sent)
        else:
            print(f'minimum term fraction {min_term_frac} is higher than '
                  f'max_min_term_frac {self.max_min_term_frac} for phrase {phrase}')

    def _extract_long_phrases_from_selected(self, sent: Dict[str, any], selected: List[Union[str, None]],
                                            min_phrase_length: int = 3,
                                            max_variables: int = 0) -> Generator[TemplatePhraseMatch, None, None]:
        phrase = []
        phrase_start = 0
        # ti = 0
        # while ti < len(selected):
        words = get_sent_terms(sent)
        for ti, term in enumerate(selected):
            # term = selected[ti]
            # ti += 1
            if term is None and phrase.count(None) == max_variables:
                if len(phrase) - phrase.count(None) >= min_phrase_length:
                    min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase])
                    if min_term_frac < self.max_min_term_frac:
                        template_phrase = make_template_phrase(phrase)
                        variable_match = words[phrase_start: phrase_start + len(phrase)]
                        yield TemplatePhraseMatch(template_phrase, word_start=phrase_start,
                                                  variable_match=variable_match, sent=sent)
                phrase = []
            if term is None and len(phrase) == 0:
                continue
            elif len(phrase) == 0:
                phrase_start = ti
            phrase.append(term)
        if len(phrase) - phrase.count(None) >= min_phrase_length:
            min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase])
            if min_term_frac < self.max_min_term_frac:
                template_phrase = make_template_phrase(phrase)
                variable_match = words[phrase_start: phrase_start + len(phrase)]
                yield TemplatePhraseMatch(template_phrase, word_start=phrase_start,
                                          variable_match=variable_match, sent=sent)

    def _extract_template_phrases(self, min_length: int = 3, max_length: int = 5,
                                  min_cooc_freq: int = None, max_sents: int = None) -> Tuple[int, TemplatePhraseMatch]:
        for sent_selected in self._iter_get_sent_and_selected_terms(min_cooc_freq=min_cooc_freq, max_sents=max_sents):
            selected = sent_selected['selected']
            sent = sent_selected['sent']
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
                    template_phrase = make_template_phrase(phrase)
                    variable_match = sent['words'][phrase_start: phrase_start+len(phrase)]
                    yield TemplatePhraseMatch(template_phrase, word_start=phrase_start,
                                              variable_match=variable_match)

        return None

    def index_template_sents(self, template_phrases: List[Union[str, List[str]]],
                             min_cooc_freq: int = None, **kwargs) -> Dict[str, List[str]]:
        if min_cooc_freq is None:
            min_cooc_freq = self.min_cooc_freq
        template_sent_index = defaultdict(list)
        template_phrases = transform_templates_to_strings(template_phrases)
        for sent in self.sent_iterator:
            selected = self._get_selected_terms(sent, min_cooc_freq=min_cooc_freq)
            for phrase in self._extract_long_phrases_from_selected(selected, **kwargs):
                if phrase not in template_phrases:
                    continue
                template_sent_index[phrase].append(sent['id'])
        return template_sent_index

    def extract_template_variables(self, phrase_type: str, templates: List[Union[str, List[str]]],
                                   min_cooc_freq: int = None, max_sents: int = None, *args, **kwargs):
        if min_cooc_freq is None:
            if self.min_cooc_freq is None:
                raise ValueError(f'no min_cooc_freq passed, nor set in {self.__class__.__name__} instance')
            min_cooc_freq = self.min_cooc_freq
        template_set = {t for t in transform_templates_to_strings(templates)}
        extract_func = self._get_extract_function(phrase_type)
        for si, sent in enumerate(self.sent_iterator):
            if (si+1) >= max_sents:
                break
            selected = self._get_selected_terms(sent, min_cooc_freq=min_cooc_freq)
            for template_phrase_match in extract_func(sent=sent,
                                                      selected=selected,
                                                      *args, **kwargs):
                if template_phrase_match.template_phrase.phrase_string in template_set:
                    # print(template_phrase_match)
                    # print(template_phrase_match.word_start)
                    # print(template_phrase_match.word_end)
                    variable_match = sent['words'][template_phrase_match.word_start: template_phrase_match.word_end]
                    yield variable_match, template_phrase_match


def transform_template_to_list(template: Union[str, List[str]]) -> List[str]:
    if isinstance(template, str):
        return template.split(' ')
    elif isinstance(template, list) is False:
        raise TypeError(f'template must be str or list of str, not {type(template)}')
    else:
        return template


def transform_template_to_string(template: Union[str, List[str]]) -> str:
    if isinstance(template, list):
        return ' '.join(template)
    elif isinstance(template, str) is False:
        raise TypeError(f'template must be str or list of str, not {type(template)}')
    else:
        return template


def transform_templates_to_lists(templates: List[Union[str, List[str]]]) -> List[List[str]]:
    return [transform_template_to_list(template) for template in templates]


def transform_templates_to_strings(templates: List[Union[str, List[str]]]) -> List[str]:
    return [transform_template_to_string(template) for template in templates]
