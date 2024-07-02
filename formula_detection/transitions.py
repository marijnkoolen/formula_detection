import copy
from collections import Counter
from collections import defaultdict
from typing import Dict, List, Set, Union


def tokenise(phrase_context: str):
    return phrase_context.strip().split()


def normalise_context(context_words: List[str], variant_of: Dict[str, str]) -> List[str]:
    return [w if w not in variant_of else variant_of[w] for w in context_words]


def select_context_words(phrases: Union[str, List[str]], context_count: Dict[str, Counter],
                         variant_of: Dict[str, str], min_word_prob: float = 0.1) -> List[str]:
    total = 0
    context_word_freq = Counter()
    # print('select_context_words - phrases:', phrases)
    # print('select_context_words - context_count.keys():', context_count.keys())
    for phrase in phrases:
        # print('select_context_words - phrase:', phrase)
        try:
            total += sum(context_count[phrase].values())
        except TypeError:
            print(phrase)
            print(context_count)
            raise
        # print('select_context_words - total:', total)
        for pc in context_count[phrase]:
            # print('select_context_words - pc:', pc)
            norm_pc = normalise_context(tokenise(pc), variant_of)
            # print('select_context_words - norm_pc:', norm_pc)
            for norm_word in set(norm_pc):
                context_word_freq[norm_word] += context_count[phrase][pc]
    selected = []
    for word, freq in context_word_freq.most_common():
        if freq / total < min_word_prob:
            continue
        # print(f'{sub_phrase: <40}{word: <20}{freq: >8}{freq / total: >8.2f}')
        selected.append(word)
    return selected


def prune_branch(curr_node: str, transition_probs: Dict[str, Dict[str, float]], debug: bool = False):
    next_nodes = list(transition_probs[curr_node].keys())
    for next_node in next_nodes:
        if debug:
            print('REMOVING connection', curr_node, next_node)
        del transition_probs[curr_node][next_node]
        prune_branch(next_node, transition_probs)
    del transition_probs[curr_node]
    if debug:
        print('REMOVING leaf', curr_node)
    return None


def prune_word_transitions(transition_probs: Dict[str, Dict[str, float]],
                           min_prob_threshold: float = 0.01, debug: bool = False):
    curr_words = list(transition_probs.keys())
    for curr_word in curr_words:
        next_words = list(transition_probs[curr_word].keys())
        for next_word in next_words:
            if transition_probs[curr_word][next_word] < min_prob_threshold:
                # print('curr:', curr_word, '\tnext:', next_word, '\tprob:', transition_probs[curr_word][next_word])
                del transition_probs[curr_word][next_word]
                # prune_branch(next_word, transition_probs, debug=debug)
    for word in curr_words:
        if word in transition_probs and len(transition_probs[word]) == 0:
            del transition_probs[word]
    return None


def prune_phrase_transitions(transition_probs: Dict[str, Dict[str, float]],
                             min_prob_threshold: float = 0.1, debug: bool = False):
    phrases = sorted(transition_probs.keys(), key=lambda x: len(x))
    for phrase in phrases:
        extended_phrases = list(transition_probs[phrase].keys())
        # print('prune_phrase_transitions - extended_phrases:', extended_phrases)
        for extended_phrase in extended_phrases:
            if transition_probs[phrase][extended_phrase] < min_prob_threshold:
                # print(f"prune_phrase_transitions - pruning", phrase, extended_phrase,
                #       transition_probs[phrase][extended_phrase])
                del transition_probs[phrase][extended_phrase]
                prune_branch(extended_phrase, transition_probs, debug=debug)
            else:
                # print(f"prune_phrase_transitions - keeping", phrase, extended_phrase,
                #       transition_probs[phrase][extended_phrase])
                pass
    for phrase in phrases:
        if phrase in transition_probs and len(transition_probs[phrase]) == 0:
            del transition_probs[phrase]
        # print('prune_phrase_transitions - phrases after pruning:', len(transition_probs[phrase]), phrase)
    return None


def prune_transitions(transition_probs: Dict[str, Dict[str, Dict[str, float]]], min_prob_threshold: float = 0.1,
                      from_phrase: bool = True, debug: bool = False) -> Dict[str, Dict[str, Dict[str, float]]]:
    for direction in transition_probs:
        if from_phrase:
            prune_phrase_transitions(transition_probs[direction], min_prob_threshold=min_prob_threshold, debug=debug)
            start_node = ('<PHRASE>',)
        else:
            prune_word_transitions(transition_probs[direction], min_prob_threshold=min_prob_threshold, debug=debug)
            start_node = '<PHRASE>'
        # print(len(transition_probs))
        following_nodes = get_nodes_following_phrase(transition_probs[direction], {start_node})
        # print(f'prune_transitions - direction {direction} - following_nodes:', len(following_nodes))
        start_nodes = list(transition_probs[direction].keys())
        # print(f'prune_transitions - direction {direction} - start_nodes:', len(start_nodes))
        for s in start_nodes:
            if s not in following_nodes:
                del transition_probs[direction][s]
    return transition_probs


def compute_transition_probs(phrases: Union[str, List[str]], context_count: Dict[str, Dict[str, Counter]],
                             min_prob_threshold: float = 0.1, min_word_prob: float = 0.1,
                             variant_of: Dict[str, str] = None,
                             from_phrase: bool = True, exclude_var: bool = False,
                             debug: bool = False):
    if isinstance(phrases, str):
        phrases = [phrases]
    print('compute_transition_probs - context_count.keys():', context_count.keys())
    for direction in ['pre', 'post']:
        print(f'compute_transition_probs - context_count["{direction}"]:', len(context_count[direction]))
    transition_freq = count_transitions(phrases, context_count, min_word_prob=min_word_prob,
                                        variant_of=variant_of, from_phrase=from_phrase,
                                        exclude_var=exclude_var)
    print(f"compute_transition_probs - num transitions:", len(transition_freq['pre']))
    transition_probs = defaultdict(lambda: defaultdict(dict))
    for direction in transition_freq:
        for curr_node in transition_freq[direction]:
            total = sum(transition_freq[direction][curr_node].values())
            # print('compute_transition_probs - curr_node:', curr_node)
            # print('compute_transition_probs - total:', total)
            for next_node in transition_freq[direction][curr_node]:
                trans_freq = transition_freq[direction][curr_node][next_node]
                # print('compute_transition_probs - next_node:', next_node)
                if from_phrase:
                    if isinstance(curr_node, tuple):
                        next_node = tuple(list(curr_node) + [next_node])
                    elif isinstance(next_node, tuple):
                        next_node = tuple([curr_node] + list(next_node))
                    else:
                        print('compute_transition_probs - curr_node:', curr_node)
                        print('compute_transition_probs - next_node:', next_node)
                        raise TypeError('when using from_phrase=True, curr_node or next_node must be a tuple')
                # print('compute_transition_probs - next_node:', next_node)
                # print('\tfreq:', trans_freq, '\ttotal:', total, '\tprob:', trans_freq / total)
                transition_probs[direction][curr_node][next_node] = trans_freq / total
    # print(len(transition_probs))
    transition_probs = prune_transitions(transition_probs, min_prob_threshold=min_prob_threshold,
                                         from_phrase=from_phrase, debug=debug)
    return transition_probs


def get_nodes_following_phrase(transition_probs, start_nodes: Set[str]):
    # print('get_nodes_following_phrase - start nodes:', len(start_nodes))
    following_nodes = copy.deepcopy(start_nodes)
    # print(transition_probs)
    for curr_node in start_nodes:
        for next_node in transition_probs[curr_node]:
            if next_node not in following_nodes:
                # print('\tcurr_node:', curr_node, '\tadding', next_node)
                following_nodes.add(next_node)
                # print('\tnext node:', next_node)
                following_nodes = get_nodes_following_phrase(transition_probs, following_nodes)
    # print('get_nodes_following_phrase - returning nodes:', len(following_nodes))
    return following_nodes


def count_transitions(phrases: Union[str, List[str]], context_count: Dict[str, Dict[str, Counter]],
                      min_word_prob: float = 0.1, variant_of: Dict[str, str] = None,
                      from_phrase: bool = True, exclude_var: bool = False) -> Dict[str, Dict[str, Counter]]:
    if variant_of is None:
        variant_of = {}
    transition_freq = {}
    for direction in {'pre', 'post'}:
        # print('count_transitions - direction:', direction)
        # print(f'count_transitions - context_count["{direction}"]:', len(context_count[direction]))
        selected_words = select_context_words(phrases, context_count[direction], variant_of,
                                              min_word_prob=min_word_prob)
        selected_words.append('<PHRASE>')
        # print('count_transitions - selected_words:', selected_words)
        transition_freq[direction] = defaultdict(Counter)
        for phrase in phrases:
            # print('\n\ncount_transitions - direction:', direction)
            for pc in context_count[direction][phrase]:
                # print('count_transitions - pc:', pc)
                norm_pc = normalise_context(tokenise(pc), variant_of)
                # print('count_transitions - norm_pc:', norm_pc)
                if direction == 'pre':
                    norm_pc = [w for w in reversed(norm_pc)]
                # print('count_transitions - reversed norm_pc:', norm_pc)
                selected_pc = ['<PHRASE>'] + [w if w in selected_words else f'<VAR-{wi}>'
                                              for wi, w in enumerate(norm_pc)]
                # print('count_transitions - selected_pc:', selected_pc)
                for i in range(len(selected_pc)-1):
                    trans_word = selected_pc[i+1]
                    if exclude_var and trans_word.startswith('<VAR'):
                        break
                    if from_phrase is True:
                        curr_phrase = tuple(selected_pc[:i+1])
                        # curr_word = selected_pc[i]
                        # print(f"{i: <4}{curr_word: <40}{trans_word: <20}\t{curr_phrase}")
                        transition_freq[direction][curr_phrase][trans_word] += context_count[direction][phrase][pc]
                    else:
                        curr_word = selected_pc[i]
                        transition_freq[direction][curr_word][trans_word] += context_count[direction][phrase][pc]
    return transition_freq
