from typing import Dict, Union

from formula_detection.context import map_context_word_variants
from formula_detection.transitions import compute_transition_probs
from formula_detection.variation.edit import compute_variant_similarity


def is_variant(phrase1: str, phrase2: str, known_variants: Dict[str, str],
               known_distractors: Dict[str, str]) -> bool:
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


def extend_phrase(sub_phrase, context_count, known_variants):
    pre_variant_of = map_context_word_variants(sub_phrase, context_count['pre'],
                                               None, known_variants=known_variants)

    pre_transition_probs = compute_transition_probs(sub_phrase, context_count['pre'], direction='pre',
                                                    variant_of=pre_variant_of, from_phrase=True)

    post_variant_of = map_context_word_variants(sub_phrase, context_count['post'],
                                                None, known_variants=known_variants)

    post_transition_probs = compute_transition_probs(sub_phrase, context_count['post'], direction='post',
                                                     variant_of=post_variant_of, from_phrase=True)

    pre_extended_sub_phrases = find_sub_phrase_extensions(tuple([sub_phrase]), pre_transition_probs)
    post_extended_sub_phrases = find_sub_phrase_extensions(tuple([sub_phrase]), post_transition_probs)
    return get_extended_phrases(sub_phrase, pre_extended_sub_phrases, post_extended_sub_phrases)
