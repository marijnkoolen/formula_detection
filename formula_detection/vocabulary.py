from collections import Counter
from typing import Iterable, List, Union

from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.token import Doc


def token_to_string(term: Union[str, Token]):
    if isinstance(term, str):
        return term
    elif isinstance(term, Token):
        return term.n


class Vocabulary:

    def __init__(self, terms: Union[List[Union[str, Token]], Doc] = None):
        self.term_id = {}
        self.id_term = {}
        if terms is not None:
            self.index_terms(terms)

    def __repr__(self):
        return f'{self.__class__.__name__}(vocabulary_size="{len(self.term_id)}")'

    def __len__(self):
        return len(self.term_id)

    def reset_index(self):
        self.term_id = {}
        self.id_term = {}

    def _add_term(self, term: Union[str, Token]):
        # print('_add_term token term:', term, len(self.term_id))
        # print(self.term_id)
        term = token_to_string(term)
        # print('_add_term term:', term, type(term), term in self.term_id)
        if term in self.term_id:
            return self.term_id[term]
        else:
            term_id = len(self.term_id)
            # print(f'_add_term adding term {term} with term_id {term_id}')
            self.term_id[term] = term_id
            # print(f'\tlen term_id:', len(self.term_id))
            self.id_term[term_id] = term
            return term_id

    def index_term(self, term: Union[str, Token]):
        return self._add_term(term)

    def index_terms(self, terms: Union[List[Union[str, Token]], str, Token, Doc],
                    reset_index: bool = False):
        if isinstance(terms, str) or isinstance(terms, Token):
            terms = [terms]
        # print('terms:', terms)
        if reset_index is True:
            self.reset_index()
        for term in terms:
            term = token_to_string(term)
            # print('index_terms - string token:', term, term in self.term_id)
            if term in self.term_id:
                continue
            self._add_term(term)

    def term2id(self, term: Union[str, Token]):
        # print('term2id - before:', term)
        term = token_to_string(term)
        # print('term2id - after:', term)
        # print(self.term_id)
        # print(term in self.term_id, self.term_id[term] if term in self.term_id else 'MISSING')
        return self.term_id[term] if term in self.term_id else None

    def id2term(self, term_id: int):
        return self.id_term[term_id] if term_id in self.id_term else None


def make_selected_vocab(full_vocab: Vocabulary, selected_terms: List[str] = None,
                        selected_ids: List[int] = None, term_freq: Counter = None,
                        min_term_freq: int = None) -> Vocabulary:
    selected_vocab = Vocabulary()
    if term_freq is not None:
        if not isinstance(min_term_freq, int):
            raise TypeError('if term_freq is passed, min_term_freq is required and must be an integer')
        selected_ids = [term_id for term_id in term_freq if term_freq[term_id] >= min_term_freq]
    if selected_terms is not None:
        for term in selected_terms:
            term_id = full_vocab.term2id(term)
            selected_vocab.term_id[term] = term_id
            selected_vocab.id_term[term_id] = term
    elif selected_ids is not None:
        for term_id in selected_ids:
            term = full_vocab.id2term(term_id)
            selected_vocab.term_id[term] = term_id
            selected_vocab.id_term[term_id] = term
    else:
        raise ValueError('must pass either selected_terms or selected_ids')
    return selected_vocab


def calculate_term_freq(doc_iterator: Iterable, vocab: Vocabulary) -> Counter:
    term_freq = Counter()
    for di, doc in enumerate(doc_iterator):
        term_ids = [vocab.index_term(term) for term in doc]
        term_freq.update(term_ids)
    return term_freq
