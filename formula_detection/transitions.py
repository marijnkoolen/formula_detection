"""transitions.py — Markov-chain-style transition probabilities for phrase-context extension.

Builds, for a set of candidate phrases, weighted "transition" graphs describing
how likely the phrase's surrounding context (the words immediately before or
after it) is to extend the phrase one word at a time. Context occurrences are
collected elsewhere (see formula_detection.context) as raw word sequences per
phrase; this module normalises and tallies those sequences into per-direction
transition-frequency/probability tables, then prunes low-probability or
disconnected branches so that only the contexts that recur consistently enough
remain. The resulting transition_probs structure is consumed by
formula_detection.phrases to decide how far a formulaic phrase extends before
and after its core, and where it likely splits into variant continuations.
"""
import copy
from collections import Counter
from collections import defaultdict
from typing import Dict, List, Set, Union


def tokenise(phrase_context: str):
    """Split a whitespace-joined context string into a list of word tokens.

    Args:
        phrase_context: A context string, e.g. words preceding or following a phrase.

    Returns:
        List of word tokens.
    """
    return phrase_context.strip().split()


def normalise_context(context_words: List[str], variant_of: Dict[str, str]) -> List[str]:
    """Replace each context word with its canonical variant, if one is known.

    Args:
        context_words: List of word tokens making up a context.
        variant_of: Mapping from a word to the canonical word it is a variant of.

    Returns:
        List of words with known variants replaced by their canonical form.
    """
    return [w if w not in variant_of else variant_of[w] for w in context_words]


def select_context_words(phrases: Union[str, List[str]], context_count: Dict[str, Counter],
                         variant_of: Dict[str, str], min_word_prob: float = 0.1) -> List[str]:
    """Determine which (normalised) context words occur frequently enough to keep.

    Aggregates, across all given phrases, the frequency of each normalised word
    appearing anywhere in their recorded contexts, then keeps only the words
    whose relative frequency (occurrences divided by the total number of
    context occurrences) is at least `min_word_prob`. Words that fall below the
    threshold are treated as too rare/variable to be informative and are later
    replaced with placeholder tokens by `count_transitions`.

    Args:
        phrases: A single phrase string or list of phrase strings to aggregate contexts for.
        context_count: Mapping from phrase to a Counter of context strings to their frequency.
        variant_of: Mapping from a word to the canonical word it is a variant of.
        min_word_prob: Minimum relative frequency (0-1) a normalised word must reach to be selected.

    Returns:
        List of normalised context words whose relative frequency meets the threshold,
        ordered from most to least frequent.
    """
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
    """Recursively remove a node and everything reachable from it from the transition graph.

    Used to discard an entire sub-tree of (phrase or word) transitions once an
    ancestor transition has been pruned, so that orphaned descendant nodes are
    not left dangling in `transition_probs`.

    Args:
        curr_node: The node (phrase tuple or word) whose outgoing edges and descendants
            should be deleted.
        transition_probs: Mapping from node to a mapping of next-node to transition
            probability. Mutated in place.
        debug: If True, print each connection/leaf removal as it happens.

    Returns:
        None.
    """
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
    """Remove word-to-word transitions below a probability threshold, then drop empty nodes.

    Unlike `prune_phrase_transitions`, this does not recursively prune the
    descendants of a removed edge (the `prune_branch` call for the removed
    next-word is commented out) — it only deletes the direct edge, and
    afterwards removes any current word left with no outgoing transitions.

    Args:
        transition_probs: Mapping from current word to a mapping of next-word to
            transition probability. Mutated in place.
        min_prob_threshold: Minimum transition probability required to keep an edge.
        debug: Present for interface symmetry with `prune_phrase_transitions`; unused here.

    Returns:
        None.
    """
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
    """Prune low-probability phrase-extension transitions and their entire descendant sub-trees.

    Processes phrases shortest-first so that ancestors are pruned before their
    extensions are considered. For each phrase, any extension whose transition
    probability falls below `min_prob_threshold` is removed, along with the
    extended phrase's entire branch (via `prune_branch`), since an
    extension that is itself unlikely makes any further extension of it moot.
    After all phrases are processed, any phrase left with no remaining
    extensions is deleted from `transition_probs`.

    Args:
        transition_probs: Mapping from phrase (tuple) to a mapping of extended-phrase
            (tuple) to transition probability. Mutated in place.
        min_prob_threshold: Minimum transition probability required to keep an edge.
        debug: If True, print pruning decisions (currently only forwarded to `prune_branch`).

    Returns:
        None.
    """
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
    """Prune low-probability transitions in both directions and discard unreachable start nodes.

    For each direction ('pre' and 'post'), delegates to either
    `prune_phrase_transitions` or `prune_word_transitions` (depending on
    `from_phrase`) to remove weak edges and their descendants, then walks the
    remaining graph from the `<PHRASE>` root via `get_nodes_following_phrase`
    to find every node still reachable from it. Any top-level start node that
    is not reachable from `<PHRASE>` (i.e. became disconnected after pruning)
    is removed from `transition_probs[direction]`.

    Args:
        transition_probs: Mapping from direction ('pre'/'post') to a mapping of
            current node to a mapping of next node to transition probability.
            Mutated in place.
        min_prob_threshold: Minimum transition probability required to keep an edge,
            passed through to the per-direction pruning function.
        from_phrase: If True, nodes are phrase tuples and `<PHRASE>` is represented
            as the tuple `('<PHRASE>',)`; if False, nodes are plain words and the
            root is the string `'<PHRASE>'`.
        debug: If True, print pruning decisions in the underlying pruning functions.

    Returns:
        The same `transition_probs` dict, pruned in place.
    """
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
    """Compute and prune transition probabilities describing how phrases extend into their context.

    End-to-end pipeline: counts raw context-word transitions via
    `count_transitions`, converts the resulting frequencies into probabilities
    by normalising each current-node's outgoing transition counts to sum to 1,
    builds extended-phrase keys (tuples) when `from_phrase` is True, and
    finally prunes the resulting probability tables via `prune_transitions`.

    Args:
        phrases: A single phrase string or list of phrase strings to compute transitions for.
        context_count: Mapping from direction ('pre'/'post') to a mapping of phrase to
            a Counter of context strings to their frequency.
        min_prob_threshold: Minimum transition probability required to keep an edge
            during pruning.
        min_word_prob: Minimum relative frequency a context word must reach to be
            selected as a transition node rather than collapsed to a `<VAR-i>` placeholder.
        variant_of: Optional mapping from a word to the canonical word it is a variant of.
        from_phrase: If True, build up phrase tuples (e.g. extending `('w1',)` to
            `('w1', 'w2')`) as transition nodes; if False, use plain words as nodes.
        exclude_var: If True, stop counting a context's transitions as soon as a
            variable/placeholder word is reached.
        debug: If True, print pruning decisions in the underlying pruning functions.

    Returns:
        Mapping from direction ('pre'/'post') to a mapping of current node to a
        mapping of next node to transition probability, with low-probability and
        unreachable branches pruned.

    Raises:
        TypeError: If `from_phrase` is True but neither the current nor next node
            in a transition is a tuple.
    """
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
    """Recursively collect every node reachable from a set of start nodes in the transition graph.

    Performs a depth-first traversal: for each start node, follows every
    outgoing transition and adds the destination node to the result set,
    recursing further from each newly discovered node, until no new nodes are
    found.

    Args:
        transition_probs: Mapping from node to a mapping of next-node to transition probability.
        start_nodes: Initial set of nodes to begin the traversal from.

    Returns:
        Set containing the start nodes plus every node reachable from them.
    """
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
    """Count word-to-word or phrase-to-phrase transitions observed in phrase contexts.

    For each direction ('pre' and 'post'), selects the context words frequent
    enough to be treated as informative (via `select_context_words`; all
    others are replaced with positional `<VAR-i>` placeholders), prepends a
    `<PHRASE>` start marker to each (normalised, and for 'pre' reversed so that
    traversal always moves away from the phrase) context word sequence, and
    tallies, for every adjacent pair of tokens in that sequence, how often the
    transition from the current node to the next occurs. When `from_phrase` is
    True the "current node" is the growing tuple of tokens seen so far
    (so transitions are phrase-to-extended-phrase); when False it is just the
    single preceding word.

    Args:
        phrases: A single phrase string or list of phrase strings to count transitions for.
        context_count: Mapping from direction ('pre'/'post') to a mapping of phrase to
            a Counter of context strings to their frequency.
        min_word_prob: Minimum relative frequency a context word must reach to be
            selected as a transition node rather than collapsed to a `<VAR-i>` placeholder.
        variant_of: Optional mapping from a word to the canonical word it is a variant of.
            Defaults to an empty mapping.
        from_phrase: If True, accumulate transitions keyed by the growing phrase tuple;
            if False, key transitions by the single current word.
        exclude_var: If True, stop counting further transitions in a context as soon as
            a `<VAR-i>` placeholder token is reached.

    Returns:
        Mapping from direction ('pre'/'post') to a mapping of current node (word or
        phrase tuple) to a Counter of next-node to transition frequency.
    """
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
