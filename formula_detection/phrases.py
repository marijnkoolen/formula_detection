"""phrases.py — Phrase variant detection and phrase-extension/split discovery.

Builds on formula_detection.transitions' transition-probability graphs to
answer two related questions about formulaic phrases: whether two phrase
strings are variants of each other (allowing for known word substitutions and
small edit differences), and how a phrase's surrounding context extends or
splits it into longer variant phrases. The extension logic walks the
transition graph produced by compute_transition_probs, classifying each
discovered extension as part of the dominant ("main") continuation or as a
lower-probability ("split") branch representing an alternative phrasing.
"""
from typing import Dict, Union

from formula_detection.variation.edit import compute_variant_similarity


def is_variant(phrase1: str, phrase2: str, known_variants: Dict[str, str],
               known_distractors: Dict[str, str]) -> bool:
    """Determine whether two equal-length phrases are variants of each other.

    Normalises both phrases via `known_variants`, then short-circuits based on
    edit-similarity (computed via `compute_variant_similarity`): an exact match
    (similarity 1.0) is always a variant pair, while similarity below 0.7
    immediately rules it out. Otherwise, word-by-word, a mismatch is tolerated
    only if it is explained by a known variant mapping in either direction; a
    mismatch matching a known *distractor* mapping immediately rules out a
    variant relationship. Returns True only if every mismatched word pair was
    explained by a known variant.

    Args:
        phrase1: First phrase string.
        phrase2: Second phrase string.
        known_variants: Mapping from a word to a word it is a known variant of
            (checked in both directions).
        known_distractors: Mapping from a word to a word that looks similar but is
            known NOT to be a variant (checked in both directions).

    Returns:
        True if the phrases are equal length and every word difference is explained
        by a known variant mapping (or the phrases are similar/identical enough);
        False if lengths differ, similarity is too low, or a known distractor mismatch
        is found. Implicitly returns None if there are unmatched word pairs that are
        not distractors (i.e. the function falls through without an explicit return).
    """
    words1 = [known_variants[w] if w in known_variants else w for w in phrase1.split(' ')]
    words2 = [known_variants[w] if w in known_variants else w for w in phrase2.split(' ')]
    if len(words1) != len(words2):
        return False
    var_sim = compute_variant_similarity(phrase1, phrase2)
    if var_sim == 1.0:
        return True
    if var_sim < 0.7:
        return False
    # print('var_sim:', var_sim, phrase1, phrase2)
    unmatched = []
    for wi, w1 in enumerate(words1):
        w2 = words2[wi]
        if w2 == w1:
            continue
        if w1 in known_variants and known_variants[w1] == w2:
            continue
        if w2 in known_variants and known_variants[w2] == w1:
            continue
        if w1 in known_distractors and known_distractors[w1] == w2:
            return False
        if w2 in known_distractors and known_distractors[w2] == w1:
            return False
        unmatched.append((w1, w2))
    if len(unmatched) == 0:
        return True


def get_partial_overlap(phrase1: str, phrase2: str, known_variants: Dict[str, str],
                        min_overlap: int = 3) -> Union[None, str]:
    """Find the longest suffix-of-one/prefix-of-other overlap between two phrases.

    Returns immediately if one phrase equals or is contained in the other.
    Otherwise normalises both phrases via `known_variants` and searches, from
    the largest possible overlap down to `min_overlap`, for a word sequence
    that is simultaneously a suffix of one phrase and a prefix of the other
    (in either direction).

    Args:
        phrase1: First phrase string.
        phrase2: Second phrase string.
        known_variants: Mapping from a word to a word it is a known variant of.
        min_overlap: Minimum number of overlapping words required to report a match.

    Returns:
        The shorter containing phrase if one phrase contains the other, the
        overlapping word sequence (joined with spaces) if a suffix/prefix overlap of
        at least `min_overlap` words is found, or None if no qualifying overlap exists.
    """
    if phrase1 == phrase2:
        return phrase1
    elif phrase1 in phrase2:
        return phrase1
    elif phrase2 in phrase1:
        return phrase2
    words1 = [known_variants[w] if w in known_variants else w for w in phrase1.split(' ')]
    words2 = [known_variants[w] if w in known_variants else w for w in phrase2.split(' ')]
    max_overlap = min(len(words1), len(words2))
    # print('max_overlap:', max_overlap)
    for i in range(max_overlap, min_overlap - 1, -1):
        if words1[-i:] == words2[:i]:
            return ' '.join(words1[-i:])
        if words2[-i:] == words1[:i]:
            return ' '.join(words2[-i:])
    return None


def find_phrase_extensions(transition_probs, min_split_prob: float = 0.1,
                           min_extend_prob: float = 0.9, debug: bool = False):
    """Find phrase extensions in both the 'pre' and 'post' context directions.

    For each direction, walks the transition graph starting from the
    `<PHRASE>` root node via `find_sub_phrase_extensions`, building a mapping
    of discovered extended-phrase tuples to whether each is the dominant
    ("main") continuation or a lower-probability ("split") branch.

    Args:
        transition_probs: Mapping from direction ('pre'/'post') to a transition-probability
            graph as produced by `compute_transition_probs`.
        min_split_prob: Minimum cumulative probability required for an extension to be
            kept at all (as a 'split' branch).
        min_extend_prob: Minimum cumulative probability required for an extension to be
            classified as the dominant 'main' continuation rather than a 'split'.
        debug: If True, print extension-discovery details from `find_sub_phrase_extensions`.

    Returns:
        Mapping from direction ('pre'/'post') to a mapping of extended-phrase tuple
        to its classification ('main' or 'split').
    """
    extended_phrases = {}
    for direction in {'pre', 'post'}:
        extensions = find_sub_phrase_extensions('<PHRASE>',
                                                transition_probs[direction],
                                                min_split_prob=min_split_prob,
                                                min_extend_prob=min_extend_prob,
                                                curr_prob=1.0, debug=debug)
        # extended_phrases[direction] = [tuple([phrase] + list(extension[1:])) for extension in extensions]
        extended_phrases[direction] = extensions
    return extended_phrases


def find_sub_phrase_extensions(curr_node, transition_probs, min_split_prob: float = 0.1,
                               min_extend_prob: float = 0.9, curr_prob: float = 1.0,
                               debug: bool = False):
    """Recursively build a tree of phrase extensions from a transition graph, classified by probability.

    Starting at `curr_node` (the phrase tuple/word reached so far, with
    cumulative probability `curr_prob`), follows every outgoing transition to
    a candidate extension. An extension is rejected outright (recursion stops
    and the function returns what it has so far) if: extending would repeat
    the same node more than twice (a cycle), the cumulative extended
    probability falls below `min_split_prob`, or the extended node is the same
    as the current node. Variable/placeholder nodes (containing '<VAR-') are
    skipped without stopping recursion. A surviving extension is recorded in
    the result mapping as 'main' once its cumulative probability reaches
    `min_extend_prob` (in which case it replaces — overwrites and removes — its
    parent's own entry, since the main path is just the parent extended
    further), and the function recurses into it to keep building the tree as
    long as its cumulative probability is at least `min_split_prob` (this
    threshold is checked again, redundantly, before recursing). Results
    returned by the recursive call are merged into the current result: any
    descendant marked 'main' replaces the immediate child phrase entry, and a
    parent phrase already in the result is dropped once a child's cumulative
    probability has fallen below `min_extend_prob` (i.e. once we're past the
    dominant path and exploring lower-probability splits).

    Args:
        curr_node: The current node in the transition graph — either a single word/token
            (for the first call, e.g. '<PHRASE>') or a phrase tuple whose last element
            is looked up in `transition_probs`.
        transition_probs: Mapping from node to a mapping of next-node to transition probability,
            for a single direction ('pre' or 'post').
        min_split_prob: Minimum cumulative probability required for an extension to be
            kept and explored further at all.
        min_extend_prob: Minimum cumulative probability required for an extension to be
            classified as the dominant 'main' continuation rather than a 'split'.
        curr_prob: Cumulative transition probability of reaching `curr_node` from the root.
        debug: If True, print detailed trace information about each extension decision.

    Returns:
        Mapping from extended-phrase tuple to its classification ('main' or 'split'),
        covering `curr_node` itself and every extension found beneath it that satisfies
        `min_split_prob`.

    Raises:
        TypeError: If a recursively discovered extension is classified as 'main' while
            its immediate parent phrase is not also 'main' (an invariant violation).
    """
    curr_type = 'main' if curr_prob >= min_extend_prob else 'split'
    if debug:
        print('find_sub_phrase_extensions start - extended_phrases curr_node:', curr_node, curr_type)
    if isinstance(curr_node, str):
        curr_phrase = tuple([curr_node])
    else:
        curr_phrase = curr_node
        curr_node = curr_node[-1]
    extended_phrases = {curr_phrase: curr_type}
    for extended_node in transition_probs[curr_node]:
        if isinstance(extended_node, str):
            extended_phrase = tuple(list(curr_phrase) + [extended_node])
        else:
            extended_phrase = extended_node
            extended_node = extended_node[-1]
        if debug:
            print('transition from', curr_node, curr_phrase, 'to', extended_node, extended_phrase)
            print('\textended_node count:', extended_phrase.count(extended_node))
        if extended_phrase.count(extended_node) > 2:
            return extended_phrases
        # if '<VAR-' in extended_node[-1]:
        if '<VAR-' in extended_node:
            continue
        extended_prob = curr_prob * transition_probs[curr_node][extended_node]
        if debug:
            print('\textended_prob:', extended_prob)
        if extended_prob < min_split_prob:
            return extended_phrases
        if debug:
            print('\textended_node:', extended_node, extended_prob)
            print('\textended_phrase:', extended_phrase, extended_prob)
        if extended_node == curr_node:
            return extended_phrases
        if extended_prob >= min_extend_prob:
            # replace current phrase with extended phrase, as it is always the follow up
            extended_phrases[extended_phrase] = 'main'
            if debug:
                print('find_sub_phrase_extensions start - setting main:', extended_node)
                print('\t\tadding extended_phrase:', extended_phrase, 'main')
                print('\t\tremoving curr_phrase:', curr_phrase)
                print('find_sub_phrase_extensions start - deleting:', curr_node)
            del extended_phrases[curr_phrase]
        if extended_prob >= min_split_prob:
            # print('find_sub_phrase_extensions recursing with extended_phrase:', extended_phrase)
            extra_phrases = find_sub_phrase_extensions(extended_phrase, transition_probs, min_split_prob=min_split_prob,
                                                       min_extend_prob=min_extend_prob, curr_prob=extended_prob)
            for extra_phrase in extra_phrases:
                if debug:
                    print('find_sub_phrase_extensions start - setting type:', extended_phrase)
                    print('find_sub_phrase_extensions start - setting type:', extra_phrase, extra_phrases[extra_phrase])
                if extra_phrases[extra_phrase] == 'main':
                    if extended_phrases[extended_phrase] != 'main':
                        print('phrase:', extended_phrase, extended_phrases[extended_phrase])
                        print('extension:', extra_phrase, extra_phrases[extra_phrase])
                        raise TypeError('extension of phrase is main but phrase itself is not')
                    if debug:
                        print('find_sub_phrase_extensions start - replacing main:', extended_phrase, extra_phrase, extra_phrases[extra_phrase])
                        print('\t\tremoving extended_phrase:', extended_phrase)
                    del extended_phrases[extended_phrase]
                extended_phrases[extra_phrase] = extra_phrases[extra_phrase]
                if debug:
                    print('\t\tadding extended_phrase:', extended_phrase, extended_phrases[extra_phrase])
                if curr_phrase in extended_phrases and curr_prob < min_extend_prob:
                    del extended_phrases[curr_phrase]
                    if debug:
                        print('\t\tremoving curr_phrase:', curr_phrase)
            # print('\tSPLITTING')
    if debug:
        print('END curr_node:', curr_node, '\tcurr_prob:', curr_prob, 'extended_phrases:', extended_phrases)
    return extended_phrases


def get_extended_phrases(sub_phrase, phrase_extensions):
    """Reconstruct the full main phrase and any split-variant phrase strings around a sub-phrase.

    Combines the 'main' pre- and post-extensions (as found by
    `find_phrase_extensions`) around `sub_phrase` into a single
    `main_phrase` string: pre-extensions are reversed (since 'pre' transitions
    walk backwards from the phrase) and prepended, post-extensions are
    appended in order. Separately, every 'split' extension (in either
    direction) is turned into its own candidate split-phrase string: pre
    splits replace the literal `<PHRASE>` placeholder with `main_phrase`, and
    post splits longer than the number of main elements found append the
    extra trailing words after `main_phrase` (replacing `<PHRASE>` with the
    original `sub_phrase` if present).

    Args:
        sub_phrase: The original phrase string the extensions are anchored to.
        phrase_extensions: Mapping from direction ('pre'/'post') to a mapping of
            extended-phrase tuple to its classification ('main'/'split'), as produced
            by `find_phrase_extensions`.

    Returns:
        Tuple of (main_phrase, split_phrases) where main_phrase is the phrase extended
        by all 'main' pre/post extensions, and split_phrases is a list of alternative
        phrase strings derived from 'split' extensions.
    """
    main_phrase = sub_phrase
    split_phrases = []

    for pre_extended_sub_phrase in phrase_extensions['pre']:
        if phrase_extensions['pre'][pre_extended_sub_phrase] == 'main':
            # print('get_extended_phrases - main pre:', pre_extended_sub_phrase, '\t')
            if len(pre_extended_sub_phrase) > 1:
                main_phrase = ' '.join(reversed(pre_extended_sub_phrase[1:])) + ' ' + main_phrase
                # print(f'pre main_phrase: #{main_phrase}#')

    main_elements = 1
    for post_extended_sub_phrase in phrase_extensions['post']:
        if phrase_extensions['post'][post_extended_sub_phrase] == 'main':
            if len(post_extended_sub_phrase) > 1:
                main_elements += len(post_extended_sub_phrase)
                main_phrase = main_phrase + ' ' + ' '.join(post_extended_sub_phrase[1:])
                # print(f'post main_phrase: #{main_phrase}#')

    for pre_extended_sub_phrase in phrase_extensions['pre']:
        if phrase_extensions['pre'][pre_extended_sub_phrase] == 'split':
            # print('main_phrase:', main_phrase)
            # print('pre_extended_sub_phrase:', pre_extended_sub_phrase)
            split_phrase = ' '.join(reversed(pre_extended_sub_phrase))
            if '<PHRASE>' in split_phrase:
                split_phrase = split_phrase.replace('<PHRASE>', main_phrase)
            split_phrases.append(split_phrase.strip())

    for post_extended_sub_phrase in phrase_extensions['post']:
        if phrase_extensions['post'][post_extended_sub_phrase] == 'split':
            if len(post_extended_sub_phrase) > main_elements:
                split_phrase = main_phrase + ' ' + ' '.join(post_extended_sub_phrase[main_elements:])
                if '<PHRASE>' in split_phrase:
                    split_phrase = split_phrase.replace('<PHRASE>', sub_phrase)
                split_phrases.append(split_phrase.strip())
    main_phrase = main_phrase.strip()
    return main_phrase, split_phrases


