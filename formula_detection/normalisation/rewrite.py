"""
rewrite.py — Generation of candidate word rewrites via n-gram
substitution, and selection among candidates by corpus frequency.

Given a word and a list of n-gram replacement rules (each an 'orig' n-gram
and its 'replace' substitute, e.g. the historical-spelling n-grams in
NGRAM_REPLACEMENTS_NL from rewrite_historic_dutch.py), this module locates
every place a rule's n-gram occurs in the word and generates every
combination of applying a subset of those occurrences, producing a set of
rewrite candidates. get_highest_freq_word then picks, among a list of
candidate words, the one most frequent in a reference corpus (as a
Counter), to choose the most plausible normalised form.

Note: the n-gram occurrence scan in get_ngram_indices advances the search
offset by a fixed 2 characters per match, so it is only correct for
2-character n-grams (as in NGRAM_REPLACEMENTS_NL).
"""
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple, Union


def get_ngram_indices(word: str, ngram_replacements: List[Dict[str, str]]) -> List[Tuple[int, Dict[str, str]]]:
    """Locate every occurrence of each replacement n-gram in a word.

    For each replacement rule, counts how many times its 'orig' n-gram
    occurs in `word`, then scans left to right to find the start index of
    each occurrence (assuming non-overlapping 2-character n-grams, since
    the scan offset advances by 2 characters after each match).

    Args:
        word: The word to scan.
        ngram_replacements: A list of replacement rules, each a dict with
            'orig' (the n-gram to find) and 'replace' (its substitute).

    Returns:
        A list of (index, ngram_replacement) pairs, one per occurrence
        found, sorted by index in descending order.
    """
    ngram_indices = []
    for ngram_replacement in ngram_replacements:
        ngram_count = word.count(ngram_replacement['orig'])
        offset = 0
        for i in range(1, ngram_count+1):
            index = offset + word[offset:].index(ngram_replacement['orig'])
            offset = index + 2
            ngram_indices.append((index, ngram_replacement))
    return sorted(ngram_indices, reverse=True)


def get_replace_indices_list(ngram_indices: List[Tuple[int, Dict[str, str]]]) -> List[Tuple[Tuple[int, Dict[str, str]], ...]]:
    """Enumerate every non-empty subset of n-gram occurrences to replace.

    Used to generate all possible combinations of applying some subset of
    the found n-gram occurrences, so that every resulting rewrite
    candidate (replacing none, some, or all occurrences) can be produced.

    Args:
        ngram_indices: The (index, ngram_replacement) pairs found by
            get_ngram_indices.

    Returns:
        A list of tuples, each tuple being one combination (of any size
        from 1 up to all of them) of (index, ngram_replacement) pairs.
    """
    replace_indices_list = []
    for num_replaces in range(1, len(ngram_indices)+1):
        replace_indices_list.extend([c for c in combinations(ngram_indices, num_replaces)])
    return replace_indices_list


def replace_ngrams(word: str, ngram_replacements: List[Dict[str, str]]) -> List[str]:
    """Generate all candidate rewrites of a word by combinations of n-gram substitution.

    Finds every occurrence of every replacement n-gram in `word`
    (get_ngram_indices), then for every non-empty combination of those
    occurrences (get_replace_indices_list), produces the word with just
    that combination of occurrences substituted. Because
    get_ngram_indices returns occurrences sorted by descending index,
    indices within a combination are applied right-to-left so that
    earlier substitutions don't shift the offsets of later ones.

    Args:
        word: The word to generate rewrite candidates for.
        ngram_replacements: A list of replacement rules, each a dict with
            'orig' (the n-gram to find) and 'replace' (its substitute).

    Returns:
        A list of candidate rewritten words, one per combination of
        n-gram occurrences substituted (not including the unmodified
        original word unless no occurrences were found, in which case
        the list is empty).
    """
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
    """Pick the most frequent word among candidates, per a reference corpus.

    Args:
        words: Candidate words to choose among.
        word_freq: A Counter mapping words to their corpus frequency.

    Returns:
        A (word, frequency) tuple for the candidate with the highest
        frequency in `word_freq`, or None if none of the candidates
        appear in `word_freq`.
    """
    wfs = [(word, word_freq[word]) for word in words if word in word_freq]
    if len(wfs) == 0:
        return None
    return max(wfs, key=lambda wf: wf[1])
