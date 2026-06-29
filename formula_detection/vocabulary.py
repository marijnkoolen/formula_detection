from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Union

from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.token import Doc


def token_to_string(term: Union[str, Token]) -> str:
    """
    Converts a given term (either a string or Token) to its string representation.

    Args:
        term (Union[str, Token]): The term to convert. Can be a string or a Token object.

    Returns:
        str: The string representation of the term.
    """
    if isinstance(term, str):
        return term
    elif isinstance(term, Token):
        return term.n


class Vocabulary:
    """
    A class to manage a vocabulary of terms, with functionality for indexing and mapping terms to IDs.

    Attributes:
        term_id (dict): A dictionary mapping canonical terms (strings) to their corresponding IDs.
        id_term (dict): A dictionary mapping term IDs to their corresponding canonical terms (strings).
        normaliser (callable or None): Optional function mapping a surface form to its canonical form.
            When set, all indexing and lookup operations apply it transparently so that surface variants
            are collapsed to a single ID. Pass None (the default) for no normalisation — this makes
            the class fully language-agnostic.
        surface_to_canonical (dict): Records every surface form seen and the canonical form it was
            mapped to. Populated only when normaliser is set.
    """

    def __init__(self, terms: Union[List[Union[str, Token]], Doc] = None,
                 normaliser: Optional[Callable[[str], str]] = None):
        """
        Initializes the Vocabulary object and optionally indexes the given terms.

        Args:
            terms (Union[List[Union[str, Token]], Doc], optional): A list of terms (strings or Token objects)
                or a Doc object containing terms to index. Defaults to None.
            normaliser (callable, optional): A function that maps a surface token string to its canonical
                form before indexing or lookup. Defaults to None (no normalisation). Keeping this None
                makes the vocabulary fully language-agnostic; supply a language-specific function
                (e.g. normalise_spelling from normalisation.rewrite_historic_dutch) when needed.
        """
        self.term_id: Dict[str, int] = {}
        self.id_term: Dict[int, str] = {}
        self.normaliser: Optional[Callable[[str], str]] = normaliser
        self.surface_to_canonical: Dict[str, str] = {}
        if terms is not None:
            self.index_terms(terms)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Vocabulary object.

        Returns:
            str: A string representation of the vocabulary, showing the size of the vocabulary.
        """
        return f'{self.__class__.__name__}(vocabulary_size="{len(self.term_id)}")'

    def __len__(self) -> int:
        """
        Returns the number of terms in the vocabulary.

        Returns:
            int: The size of the vocabulary (number of terms).
        """
        return len(self.term_id)

    def __contains__(self, item: Union[str, Token]):
        term = item.n if isinstance(item, Token) else item
        canonical = self.normaliser(term) if self.normaliser else term
        return canonical in self.term_id

    def reset_index(self):
        """
        Resets the vocabulary index, clearing all terms and their corresponding IDs.
        """
        self.term_id = {}
        self.id_term = {}

    def _add_term(self, term: Union[str, Token]) -> int:
        """
        Adds a new term to the vocabulary, or returns the ID of an existing term.
        If a normaliser is set, the canonical form is stored in term_id/id_term and
        the surface-to-canonical mapping is recorded in surface_to_canonical.

        Args:
            term (Union[str, Token]): The term to add. Can be a string or a Token object.

        Returns:
            int: The ID of the canonical term.
        """
        surface = token_to_string(term)
        canonical = self.normaliser(surface) if self.normaliser else surface
        if self.normaliser and surface != canonical:
            self.surface_to_canonical[surface] = canonical
        if canonical in self.term_id:
            return self.term_id[canonical]
        else:
            term_id = len(self.term_id)
            self.term_id[canonical] = term_id
            self.id_term[term_id] = canonical
            return term_id

    def index_term(self, term: Union[str, Token]) -> int:
        """
        Indexes a single term.

        Args:
            term (Union[str, Token]): The term to index.

        Returns:
            int: The ID of the indexed term.
        """
        return self._add_term(term)

    def index_terms(self, terms: Union[List[Union[str, Token]], str, Token, Doc],
                    reset_index: bool = False):
        """
        Indexes a list of terms or a single term.

        Args:
            terms (Union[List[Union[str, Token]], str, Token, Doc]): The terms to
                index. Can be a list of strings or Token objects, a single string
                or Token, or a Doc object containing tokens.
            reset_index (bool, optional): Whether to reset the index before indexing.
            Defaults to False.
        """
        if isinstance(terms, str) or isinstance(terms, Token):
            terms = [terms]
        if reset_index:
            self.reset_index()
        for term in terms:
            term = token_to_string(term)
            if term in self.term_id:
                continue
            self._add_term(term)

    def term2id(self, term: Union[str, Token]) -> int:
        """
        Returns the ID of a term. Applies the normaliser if one is set, so lookups
        using surface forms work transparently against a canonicalised vocabulary.

        Args:
            term (Union[str, Token]): The term to look up.

        Returns:
            int: The ID of the canonical term, or None if not found.
        """
        term = token_to_string(term)
        canonical = self.normaliser(term) if self.normaliser else term
        return self.term_id.get(canonical, None)

    def id2term(self, term_id: int) -> str:
        """
        Returns the term corresponding to a given ID.

        Args:
            term_id (int): The ID of the term.

        Returns:
            str: The term corresponding to the ID, or None if the ID is not found.
        """
        return self.id_term.get(term_id, None)


def make_selected_vocab(full_vocab: Vocabulary, selected_terms: List[str] = None,
                        selected_ids: List[int] = None, term_freq: Counter = None,
                        min_term_freq: int = None,
                        normaliser: Optional[Callable[[str], str]] = None) -> Vocabulary:
    """
    Creates a new vocabulary containing a subset of terms from the full vocabulary
    based on the provided criteria.

    Args:
        full_vocab (Vocabulary): The full vocabulary to select terms from.
            selected_terms (List[str], optional): A list of terms to include in the
            selected vocabulary.
        selected_terms (List[str], optional): A list of term strings to include in
            the selected vocabulary.
        selected_ids (List[int], optional): A list of term IDs to include in the
            selected vocabulary.
        term_freq (Counter, optional): A counter of term frequencies, used with
            min_term_freq to filter terms.
        min_term_freq (int, optional): The minimum frequency threshold to include
            terms in the selected vocabulary.

    Returns:
        Vocabulary: A new `Vocabulary` object containing the selected terms.

    Raises:
        ValueError: If neither `selected_terms` nor `selected_ids` are provided.
        TypeError: If `term_freq` is provided without `min_term_freq`.
    """
    if normaliser is None:
        normaliser = full_vocab.normaliser
    selected_vocab = Vocabulary(normaliser=normaliser)
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
    """
    Calculates the frequency of terms in a given iterator of documents.

    Args:
        doc_iterator (Iterable): An iterable of documents (can be a list of Doc objects).
        vocab (Vocabulary): The vocabulary used to index terms in the documents.

    Returns:
        Counter: A Counter object where keys are term IDs and values are term frequencies.
    """
    term_freq = Counter()
    for di, doc in enumerate(doc_iterator):
        term_ids = [vocab.index_term(term) for term in doc]
        term_freq.update(term_ids)
    return term_freq
