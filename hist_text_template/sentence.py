from typing import Dict, List, Union


def check_sent_has_terms(sent: list) -> None:
    for term in sent:
        if not isinstance(term, str):
            raise TypeError('sent should be a list of strings')
    return None


def get_sent_terms(sent: Union[List[str], Dict[str, any]]) -> List[str]:
    if isinstance(sent, dict):
        if 'words' not in sent:
            raise KeyError('sent dictionary should have a key "words" with a list of strings as value')
        return sent['words']
    elif isinstance(sent, list):
        return sent
    else:
        message = 'sent should be a list of strings or a dict with a "words" key and a list of strings as value'
        raise TypeError(f'{message}, not {type(sent)}')
