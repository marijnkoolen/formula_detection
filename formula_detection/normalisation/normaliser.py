import copy
from typing import Callable, Dict, List

from fuzzy_search.tokenization.token import Token


def normalize_tokens(tokens: List[Token], normalizer_func: Callable) -> List[Token]:
    return [normalize_token(token, normalizer_func) for token in tokens]


def normalize_token(token: Token, normalizer_func: Callable) -> Token:
    return Token(string=token.t, index=token.i,
                 normalised_string=normalizer_func(token.n),
                 metadata=copy.deepcopy(token.metadata))


def replace_token(token: Token, replace_map: Dict[str, str]) -> Token:
    replace_string = replace_map[token.n] if token.n in replace_map else token.n
    return Token(string=token.t, index=token.i, normalised_string=replace_string)


def replace_tokens(tokens: List[Token], replace_map: Dict[str, str]):
    return [replace_token(token, replace_map) for token in tokens]


def make_replace_func(replace_map: Dict[str, str]) -> Callable:

    def replace_func(tokens: List[Token]):
        return replace_tokens(tokens, replace_map)
    return replace_func


class Normalizer:

    def __init__(self, normalize_functions: List[Callable]):
        self.normalize_functions = normalize_functions

    def normalize(self, tokens: List[Token]) -> List[Token]:
        normalized_tokens = []
        for token in tokens:
            for normalize_func in self.normalize_functions:
                token = normalize_func(token)
            normalized_tokens.append(token)
        return normalized_tokens
