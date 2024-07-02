from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple, Union


def get_ngram_indices(word: str, ngram_replacements: List[Dict[str, str]]) -> List[Tuple[int, Dict[str, str]]]:
    ngram_indices = []
    for ngram_replacement in ngram_replacements:
        ngram_count = word.count(ngram_replacement['orig'])
        offset = 0
        for i in range(1, ngram_count+1):
            index = offset + word[offset:].index(ngram_replacement['orig'])
            offset = index + 2
            ngram_indices.append((index, ngram_replacement))
    return sorted(ngram_indices, reverse=True)


def get_replace_indices_list(ngram_indices: List[Tuple[int, Dict[str, str]]]):
    replace_indices_list = []
    for num_replaces in range(1, len(ngram_indices)+1):
        replace_indices_list.extend([c for c in combinations(ngram_indices, num_replaces)])
    return replace_indices_list


def replace_ngrams(word: str, ngram_replacements: List[Dict[str, str]]) -> List[str]:
    ngram_indices = get_ngram_indices(word, ngram_replacements)
    replace_indices_list = get_replace_indices_list(ngram_indices)
    replace_words = []

    for replace_indices in replace_indices_list:
        # print('replace_indices:', replace_indices)
        replace_word = word
        for index, ngram_replacement in replace_indices:
            orig, replace = ngram_replacement['orig'], ngram_replacement['replace']
            replace_word = replace_word[:index] + replace + replace_word[index+len(orig):]
        replace_words.append(replace_word)
        # print('replace_word:', replace_word)
    return replace_words


def get_highest_freq_word(words: List[str], word_freq: Counter) -> Union[Tuple[str, int], None]:
    wfs = [(word, word_freq[word]) for word in words if word in word_freq]
    if len(wfs) == 0:
        return None
    return max(wfs, key=lambda wf: wf[1])
