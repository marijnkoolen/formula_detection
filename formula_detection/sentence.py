"""
sentence.py — Small utilities for extracting and validating sentence tokens.

Helper functions used elsewhere to normalise the representation of a
sentence (either a plain list of token strings, or a dict with a "words"
key) into a flat list of token strings, and to validate that a list
actually contains only strings.
"""
from typing import Dict, List, Union


def check_sent_has_terms(sent: list) -> None:
    """Check that every element of a sentence is a string.

    Args:
        sent: A list expected to contain only string tokens.

    Returns:
        None.

    Raises:
        TypeError: If any element of ``sent`` is not a string.
    """
    for term in sent:
        if not isinstance(term, str):
            raise TypeError('sent should be a list of strings')
    return None


def get_sent_terms(sent: Union[List[str], Dict[str, any]]) -> List[str]:
    """Extract the list of token strings from a sentence representation.

    Args:
        sent: Either a list of token strings, or a dict with a "words"
            key mapping to a list of token strings.

    Returns:
        The list of token strings.

    Raises:
        KeyError: If ``sent`` is a dict without a "words" key.
        TypeError: If ``sent`` is neither a list nor a dict.
    """
    if isinstance(sent, dict):
        if 'words' not in sent:
            raise KeyError('sent dictionary should have a key "words" with a list of strings as value')
        return sent['words']
    elif isinstance(sent, list):
        return sent
    else:
        message = 'sent should be a list of strings or a dict with a "words" key and a list of strings as value'
        raise TypeError(f'{message}, not {type(sent)}')
