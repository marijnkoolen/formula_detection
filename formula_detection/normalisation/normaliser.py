"""
normaliser.py — Token-level normalisation pipeline.

Wraps arbitrary word-normalisation logic (e.g. the historical-Dutch
spelling rules in rewrite_historic_dutch.py, or a static replace_map) so
it can be applied uniformly to fuzzy_search Token objects, and chains
multiple such normalisation steps together via the Normalizer class.
"""
import copy
from typing import Callable, Dict, List

from fuzzy_search.tokenization.token import Token


def normalize_tokens(tokens: List[Token], normalizer_func: Callable) -> List[Token]:
    """Apply a normalisation function to every token in a list.

    Args:
        tokens: The tokens to normalise.
        normalizer_func: A callable that takes a token's normalised
            string and returns a new normalised string.

    Returns:
        A new list of Token objects with normalised_string set per
        `normalizer_func`.
    """
    return [normalize_token(token, normalizer_func) for token in tokens]


def normalize_token(token: Token, normalizer_func: Callable) -> Token:
    """Build a new Token whose normalised string has been rewritten.

    Args:
        token: The token to normalise.
        normalizer_func: A callable applied to `token.n` (the token's
            current normalised string) to produce the new normalised
            string.

    Returns:
        A new Token with the same string and index as `token`, its
        normalised_string replaced by `normalizer_func(token.n)`, and a
        deep copy of `token.metadata`.
    """
    return Token(string=token.t, index=token.i,
                 normalised_string=normalizer_func(token.n),
                 metadata=copy.deepcopy(token.metadata))


def replace_token(token: Token, replace_map: Dict[str, str]) -> Token:
    """Build a new Token whose normalised string is looked up in a map.

    Args:
        token: The token to normalise.
        replace_map: Mapping from a token's current normalised string
            (`token.n`) to its replacement. Tokens whose normalised
            string is not in the map are left unchanged.

    Returns:
        A new Token with the same string and index as `token`, and its
        normalised_string replaced via `replace_map` if present, or kept
        as `token.n` otherwise.
    """
    replace_string = replace_map[token.n] if token.n in replace_map else token.n
    return Token(string=token.t, index=token.i, normalised_string=replace_string)


def replace_tokens(tokens: List[Token], replace_map: Dict[str, str]) -> List[Token]:
    """Apply a replacement map to every token in a list.

    Args:
        tokens: The tokens to normalise.
        replace_map: Mapping from a token's normalised string to its
            replacement, as used by replace_token.

    Returns:
        A new list of Token objects with normalised_string rewritten per
        `replace_map` where applicable.
    """
    return [replace_token(token, replace_map) for token in tokens]


def make_replace_func(replace_map: Dict[str, str]) -> Callable:
    """Create a token-list normalisation function bound to a replace map.

    Useful for building a `normalize_functions` entry for Normalizer
    out of a static replacement dictionary.

    Args:
        replace_map: Mapping from a token's normalised string to its
            replacement, as used by replace_token.

    Returns:
        A callable that takes a list of Token objects and returns a new
        list with normalised_string rewritten per `replace_map`.
    """

    def replace_func(tokens: List[Token]):
        return replace_tokens(tokens, replace_map)
    return replace_func


class Normalizer:
    """A pipeline that applies a sequence of normalisation functions to tokens.

    Attributes:
        normalize_functions: The ordered list of callables to apply to
            each token in turn; each takes a token and returns a
            (possibly new) token, so later functions see the output of
            earlier ones.
    """

    def __init__(self, normalize_functions: List[Callable]):
        """Initialise the pipeline with an ordered list of token normalisers.

        Args:
            normalize_functions: Callables applied in order to each
                token; each takes a token and returns a token.
        """
        self.normalize_functions = normalize_functions

    def normalize(self, tokens: List[Token]) -> List[Token]:
        """Run all normalisation functions over a list of tokens.

        For each token, applies `normalize_functions` in order, feeding
        the output of each function into the next.

        Args:
            tokens: The tokens to normalise.

        Returns:
            A new list of tokens, each having passed through every
            function in `normalize_functions`.
        """
        normalized_tokens = []
        for token in tokens:
            for normalize_func in self.normalize_functions:
                token = normalize_func(token)
            normalized_tokens.append(token)
        return normalized_tokens
