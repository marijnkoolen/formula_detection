from typing import List, Union

from fuzzy_search.tokenization.token import Doc
from fuzzy_search.tokenization.token import Token
from fuzzy_search.tokenization.token import tokens2string


def transform_candidate_to_list(candidate: Union[str, List[Union[str, Token]]]) -> List[str]:
    """
    Converts a candidate (either a string or a list of strings or Tokens) into a list of strings.
    If the input is a string, it splits it by spaces; if it's already a list of strings,
    it returns the list as is.

    Args:
        candidate (Union[str, List[str]]): The candidate input, which can be either a string
                                            or a list of strings.

    Returns:
        List[str]: A list of strings, either split from the input string or directly returned
                   if the input is already a list of strings.

    Raises:
        TypeError: If the input is neither a string nor a list of strings.
    """
    if isinstance(candidate, str):
        return candidate.split(' ')
    elif all(isinstance(token, Token) for token in candidate):
        return [token.n for token in candidate]
    elif isinstance(candidate, list) is False:
        raise TypeError(f'candidate must be str or list of str, not {type(candidate)}')
    else:
        return candidate


def transform_candidate_to_string(candidate: Union[str, List[str], List[Token]]) -> str:
    """
    Converts a candidate (either a string or a list of strings or Tokens) into a single string.
    If the input is a list of strings, it joins them with spaces; if it's already a string,
    it returns the string as is.

    Args:
        candidate (Union[str, List[str], List[Token]): The candidate input, which can be
                                                       either a string or a list of strings.

    Returns:
        str: A string, either joined from the input list of strings or returned directly
             if the input is already a string.

    Raises:
        TypeError: If the input is neither a string nor a list of strings.
    """
    if isinstance(candidate, list):
        if all(isinstance(t, Token) for t in candidate):
            string = ''
            for token in candidate:
                if token.char_index > len(string):
                    string += ' ' * (token.char_index - len(string))
                string += token.t
            return string
        else:
            return ' '.join(candidate)
    elif isinstance(candidate, str) is False:
        raise TypeError(f'candidate must be str or list of str, not {type(candidate)}')
    else:
        return candidate


def transform_candidates_to_lists(candidates: List[Union[str, List[str], List[Token]]]) -> List[List[str]]:
    """
    Converts a list of candidates (each of which can be a string or a list of strings or Token
    instances) into a list of lists of strings. Each candidate is processed using
    `transform_candidate_to_list`.

    Args:
        candidates (List[Union[str, List[str]]]): A list of candidates, where each candidate
                                                   can be either a string or a list of strings.

    Returns:
        List[List[str]]: A list of lists of strings, where each candidate has been transformed
                          into a list of strings.
    """
    return [transform_candidate_to_list(candidate) for candidate in candidates]


def transform_candidates_to_strings(candidates: List[Union[str, List[str], List[Token]]]) -> List[str]:
    """
    Converts a list of candidates (each of which can be a string or a list of strings or a Token
    instances) into a list of strings. Each candidate is processed using
    `transform_candidate_to_string`.

    Args:
        candidates (List[Union[str, List[str]]]): A list of candidates, where each candidate
                                                   can be either a string or a list of strings.

    Returns:
        List[str]: A list of strings, where each candidate has been transformed into a string.
    """
    return [transform_candidate_to_string(candidate) for candidate in candidates]


class CandidatePhrase:
    """
    A class representing a candidate phrase, which can be either a string or a list of strings.

    Attributes:
        phrase_string (str): The string representation of the candidate phrase.
        phrase_list (List[str]): The list representation of the candidate phrase.
    """

    def __init__(self, phrase: Union[str, List[str], List[Token]]):
        """
        Initializes a CandidatePhrase object by transforming the input phrase into both
        string and list representations.

        Args:
            phrase (Union[str, List[str]]): The candidate phrase, either as a string or a list of strings.
        """
        self.phrase_string = transform_candidate_to_string(phrase)
        self.phrase_list = transform_candidate_to_list(phrase)

    def __len__(self) -> int:
        """
        Returns the length of the candidate phrase, based on the list representation.

        Returns:
            int: The length of the candidate phrase (number of terms in the phrase list).
        """
        return len(self.phrase_list)

    def __repr__(self) -> str:
        """
        Returns a string representation of the CandidatePhrase object.

        Returns:
            str: A string representation of the candidate phrase.
        """
        return f'({self.__class__.__name__}={self.phrase_string})'


def get_variable_terms_from_match(candidate_phrase: CandidatePhrase,
                                  variable_match: List[str]) -> List[str]:
    """
    Extracts the variable terms from a candidate phrase match, based on a list of matched terms.

    Args:
        candidate_phrase (CandidatePhrase): The candidate phrase containing placeholder variables ('<VAR>').
        variable_match (List[str]): A list of terms from the document that match the variable placeholders.

    Returns:
        List[str]: A list of terms that correspond to the '<VAR>' placeholders in the candidate phrase.
    """
    variable_terms = []
    for ti, term in enumerate(candidate_phrase.phrase_list):
        if term == '<VAR>':
            variable_terms.append(variable_match[ti])
    return variable_terms


class CandidatePhraseMatch:
    """
    A class representing a match of a candidate phrase within a document.

    Attributes:
        candidate_phrase (CandidatePhrase): The candidate phrase that was matched.
        char_start (int): The starting character index of the match in the document.
        char_end (int): The ending character index of the match in the document.
        word_start (int): The starting word index of the match in the document.
        word_end (int): The ending word index of the match in the document.
        variable_match (List[str]): The list of matched terms corresponding to the candidate phrase.
        variable_terms (List[str]): The list of terms corresponding to '<VAR>' placeholders in the candidate phrase.
        doc_id (str): The ID of the document in which the match was found.
    """

    def __init__(self, candidate_phrase: CandidatePhrase, char_start: int = None,
                 word_start: int = None, variable_match: List[str] = None,
                 doc: Union[Doc, List[str], List[Token]] = None):
        """
        Initializes a CandidatePhraseMatch object with the given candidate phrase and match details.

        Args:
            candidate_phrase (CandidatePhrase): The candidate phrase that matched in the document.
            char_start (int, optional): The starting character index of the match. Defaults to None.
            word_start (int, optional): The starting word index of the match. Defaults to None.
            variable_match (List[str], optional): The list of variable match terms. Defaults to None.
            doc (Doc, optional): The document in which the match occurred. Defaults to None.
        """
        self.candidate_phrase = candidate_phrase
        self.phrase = candidate_phrase.phrase_string
        self.char_start = None if char_start is None else char_start
        self.char_end = None if char_start is None else char_start + len(self.phrase)
        self.word_start = None if word_start is None else word_start
        self.word_end = None if word_start is None else word_start + len(candidate_phrase.phrase_list)
        self.variable_match = None if variable_match is None else variable_match
        self.variable_terms = []
        self.doc_id = doc.id if isinstance(doc, Doc) else None
        if self.variable_match:
            self.variable_terms = get_variable_terms_from_match(candidate_phrase, variable_match)

    def __len__(self) -> int:
        """
        Returns the length of the matched candidate phrase (in terms of characters).

        Returns:
            int: The length of the matched phrase.
        """
        return len(self.phrase)

    def __repr__(self) -> str:
        """
        Returns a string representation of the CandidatePhraseMatch object.

        Returns:
            str: A string representation of the candidate phrase match.
        """
        return f'({self.__class__.__name__}, char_start={self.char_start}, ' \
               f'word_start={self.word_start}, phrase={self.phrase})'


def make_candidate_phrase(phrase: Union[str, List[Union[str, Token, None]]]) -> CandidatePhrase:
    """
    Creates a CandidatePhrase object from a given phrase, which can be a string or a
    list of strings and None elements.

    Args:
        phrase (Union[str, List[Union[str, Token, None]]]): The phrase to convert into a
        CandidatePhrase object. None elements are converted to <VAR> tokens.

    Returns:
        CandidatePhrase: A new CandidatePhrase object representing the input phrase.
    """
    # transform to string, replace None elements by '<VAR>'
    if isinstance(phrase, list):
        phrase = transform_candidate_to_string([t if isinstance(t, str) else '<VAR>' for t in phrase])
    return CandidatePhrase(phrase)


def make_candidate_phrase_match(phrase: Union[str, List[Union[str, None]]],
                                phrase_start: int, doc: Doc) -> CandidatePhraseMatch:
    """
    Creates a CandidatePhraseMatch object by matching a candidate phrase in a document.

    Args:
        phrase (Union[str, List[Union[str, None]]]): The candidate phrase to match.
        phrase_start (int): The starting index of the phrase in the document.
        doc (Doc): The document where the phrase is being matched.

    Returns:
        CandidatePhraseMatch: A new CandidatePhraseMatch object representing the match
            of the phrase in the document.
    """
    candidate_phrase = make_candidate_phrase(phrase)
    variable_match = doc.normalized[phrase_start: phrase_start + len(phrase)]
    return CandidatePhraseMatch(candidate_phrase, word_start=phrase_start,
                                variable_match=variable_match, doc=doc)
