from typing import Generator, Iterable, List, Union
from collections import Counter

from hist_text_template.vocabulary import Vocabulary, make_selected_vocab
from hist_text_template.cooccurrence import make_cooc_freq, get_context
from hist_text_template.sentence import get_sent_terms


def make_template_phrase(phrase):
    return ' '.join([t if t else '<VAR>' for t in phrase])


def extract_sub_phrases(phrase: List[str],
                        max_length: int = 5) -> List[str]:
    sub_phrases = []
    for i in range(0, len(phrase) - max_length + 1):
        sub_phrase = ' '.join(phrase[i:i + max_length])
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
        print('full lexicon size:', len(self.term_freq))
        print('minimum term frequency:', min_term_freq)
        min_freq_term_ids = [term_id for term_id in self.term_freq if self.term_freq[term_id] >= min_term_freq]
        self.min_freq_vocab = make_selected_vocab(self.full_vocab, selected_ids=min_freq_term_ids)
        print('minimum frequency lexicon size:', len(self.min_freq_vocab))
        self.coll_size = sum(self.term_freq.values())

    def _get_selected_terms(self, min_cooc_freq: int = None) -> Generator[List[Union[str, None]], None, None]:
        if min_cooc_freq is None:
            min_cooc_freq = self.min_cooc_freq
        print('Minimum co-occurrence frequency:', min_cooc_freq)
        for si, sent in enumerate(self.sent_iterator):
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
            yield selected

    def extract_phrases(self, min_lenght: int = 3, max_length: int = 5,
                        min_cooc_freq: int = None):
        if min_cooc_freq is None:
            min_cooc_freq = self.min_cooc_freq
        for selected in self._get_selected_terms(min_cooc_freq=min_cooc_freq):
            phrase = []
            for ti, term in enumerate(selected):
                if term is None:
                    if len(phrase) > min_lenght:
                        min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase])
                        if min_term_frac < self.max_min_term_frac:
                            sub_phrases = extract_sub_phrases(phrase, max_length=max_length)
                            for sub_phrase in sub_phrases:
                                yield sub_phrase
                        else:
                            print(min_term_frac, phrase)
                    phrase = []
                    continue
                if term is None and len(phrase) == 0:
                    continue
                phrase.append(term)

    def extract_long_phrases(self, min_phrase_lenght: int = 3, max_nones: int = 2,
                             min_cooc_freq: int = None):
        for selected in self._get_selected_terms(min_cooc_freq=min_cooc_freq):
            phrase = []
            ti = 0
            while ti < len(selected):
                term = selected[ti]
                ti += 1
                if term is None and phrase.count(None) == max_nones:
                    if len(phrase) - phrase.count(None) > min_phrase_lenght:
                        min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase])
                        if min_term_frac < self.max_min_term_frac:
                            template_phrase = make_template_phrase(phrase)
                            yield template_phrase
                    phrase = []
                    continue
                if term is None and len(phrase) == 0:
                    continue
                phrase.append(term)

    def extract_template_phrases(self, min_lenght: int = 3, max_length: int = 5,
                                 min_cooc_freq: int = None):
        for selected in self._get_selected_terms(min_cooc_freq=min_cooc_freq):
            for ti, term in enumerate(selected[:-max_length + 1]):
                if term is None:
                    continue
                phrase = selected[ti:ti + max_length]
                if phrase.count(None) >= min_lenght:
                    continue
                min_term_frac = min([self.term_freq[t] / self.coll_size for t in phrase if t is not None])
                if min_term_frac < self.max_min_term_frac:
                    template_phrase = make_template_phrase(phrase)
                    yield template_phrase
