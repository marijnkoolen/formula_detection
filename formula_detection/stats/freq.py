"""
freq.py — Key-token frequency and context analysis helpers.

Identifies which tokens in a context (e.g. the words surrounding a
candidate phrase) are statistically over-represented relative to a
reference frequency distribution, using log-likelihood ratio and
percentage-difference measures from fuzzy_search.stats.freq. Also
provides helpers to locate such "key" tokens within a context phrase,
e.g. to trim a context window down to its most informative boundary.
"""
from collections import Counter
from collections import namedtuple

from fuzzy_search.tokenization.token import Tokenizer
from fuzzy_search.stats.freq import compute_llr
from fuzzy_search.stats.freq import compute_percentage_diff


KeyToken = namedtuple("KeyToken", "token llr perc_diff")


def get_key_context_tokens(context_token_counter: Counter, reference_counter: Counter, llr_treshold: float = 10.83,
                           perc_diff_treshold: float = 100.0, min_token_freq: int = 0):
    """Identify tokens in a context that are statistically over-represented.

    Compares the frequency of each token in ``context_token_counter``
    against its frequency in ``reference_counter``, using log-likelihood
    ratio (LLR) and percentage difference. A token is considered "key" if
    it occurs more often than expected (direction is not 'less'), and both
    its LLR and percentage difference exceed the given thresholds.

    Args:
        context_token_counter: Counts of tokens in the context under study.
        reference_counter: Counts of tokens in the reference distribution
            to compare against.
        llr_treshold: Minimum log-likelihood ratio for a token to be
            considered key.
        perc_diff_treshold: Minimum percentage difference for a token to
            be considered key.
        min_token_freq: Minimum frequency in ``context_token_counter`` for
            a token to be considered.

    Returns:
        A list of KeyToken namedtuples (token, llr, perc_diff), in the
        order tokens appear in ``context_token_counter.most_common()``.
    """
    context_total = sum(context_token_counter.values())
    reference_total = sum(reference_counter.values())
    key_tokens = []
    for token, freq in context_token_counter.most_common():
        if freq < min_token_freq:
            continue
        llr, direction = compute_llr(token, context_token_counter, context_total,
                                     reference_counter, reference_total,
                                     include_direction=True)
        perc_diff = compute_percentage_diff(token, context_token_counter, context_total,
                                            reference_counter, reference_total)
        if direction == 'less':
            continue
        if llr < llr_treshold:
            continue
        if perc_diff < perc_diff_treshold:
            continue
        key_tokens.append(KeyToken(token, llr, perc_diff))
    return key_tokens


def get_context_token_freq(context_phrases: Counter, tokenizer: Tokenizer) -> Counter:
    """Build a token frequency Counter from a Counter of context phrases.

    Tokenises each phrase and accumulates its frequency onto each of its
    constituent tokens, so a phrase occurring N times contributes N to the
    count of each of its tokens.

    Args:
        context_phrases: Counter mapping context phrase strings to their
            frequency.
        tokenizer: Tokenizer used to split each phrase into tokens.

    Returns:
        A Counter mapping token normalised string (token.n) to its
        aggregated frequency across all context phrases.
    """
    token_freq = Counter()
    for phrase, freq in context_phrases.most_common():
        tokens = tokenizer.tokenize(phrase)
        for token in tokens:
            token_freq[token.n] += freq
    return token_freq


def find_best_index(doc, key_tokens, direction):
    """Find the index of the best-positioned key token in a tokenised document.

    Among the tokens in ``doc`` that match one of ``key_tokens``, selects
    the one furthest along in the requested direction: the last matching
    index for direction "post", or the first matching index for direction
    "pre".

    Args:
        doc: A tokenised sequence of token objects with an ``n``
            (normalised string) attribute.
        key_tokens: A list of KeyToken namedtuples to look for in ``doc``.
        direction: Either "post" (take the maximum matching index) or
            "pre" (take the minimum matching index).

    Returns:
        The selected index into ``context_tokens``, or None if none of
        ``key_tokens`` occur in ``doc``.

    Raises:
        ValueError: If ``direction`` is not "pre" or "post".
    """
    context_tokens = [token.n for token in doc]
    key_token_indexes = [context_tokens.index(key_token.token) for key_token in key_tokens if key_token.token in context_tokens]
    if len(key_token_indexes) == 0:
        return None
    if direction == 'post':
        return max(key_token_indexes)
    elif direction == 'pre':
        return min(key_token_indexes)
    else:
        raise ValueError(f'invalid direction "{direction}", must be "pre" or "post".')


def find_key_phrase(context_phrase, key_tokens, tokenizer, direction):
    """Trim a context phrase down to its key-token boundary.

    Tokenises ``context_phrase`` and finds the best-positioned key token
    (using ``find_best_index``), then slices the phrase up to (for "post")
    or from (for "pre") that token's character span.

    Args:
        context_phrase: The context phrase string to trim.
        key_tokens: A list of KeyToken namedtuples to search for.
        tokenizer: Tokenizer used to tokenise ``context_phrase``.
        direction: Either "post" (keep everything up to and including the
            last key token) or "pre" (keep everything from the first key
            token onward).

    Returns:
        The trimmed substring of ``context_phrase``, or None if no key
        token was found in the phrase.

    Raises:
        ValueError: If ``direction`` is not "pre" or "post" (raised by
            ``find_best_index``).
    """
    doc = tokenizer.tokenize(context_phrase)
    best_index = find_best_index(doc, key_tokens, direction)
    if best_index is None:
        return None
    if direction == 'post':
        return context_phrase[:doc[best_index].char_index+len(doc[best_index])]
    elif direction == 'pre':
        return context_phrase[doc[best_index].char_index:]
