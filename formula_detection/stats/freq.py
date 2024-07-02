from collections import Counter
from collections import namedtuple

from fuzzy_search.tokenization.token import Tokenizer
from fuzzy_search.stats.freq import compute_llr
from fuzzy_search.stats.freq import compute_percentage_diff


KeyToken = namedtuple("KeyToken", "token llr perc_diff")


def get_key_context_tokens(context_token_counter: Counter, reference_counter: Counter, llr_treshold: float = 10.83,
                           perc_diff_treshold: float = 100.0, min_token_freq: int = 0):
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
    token_freq = Counter()
    for phrase, freq in context_phrases.most_common():
        tokens = tokenizer.tokenize(phrase)
        for token in tokens:
            token_freq[token.n] += freq
    return token_freq


def find_best_index(doc, key_tokens, direction):
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
    doc = tokenizer.tokenize(context_phrase)
    best_index = find_best_index(doc, key_tokens, direction)
    if best_index is None:
        return None
    if direction == 'post':
        return context_phrase[:doc[best_index].char_index+len(doc[best_index])]
    elif direction == 'pre':
        return context_phrase[doc[best_index].char_index:]
