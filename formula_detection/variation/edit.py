import re
from collections import defaultdict
from string import punctuation
from typing import Tuple

from Levenshtein import editops as get_editops

OPS = dict(replace="#", insert="+", delete="-")
PUNCTUATION = '—' + punctuation

vowels = {'a', 'e', 'i', 'o', 'u', 'y'}

multi_char_swap_map = {
    ('c', 'k'),
    ('s', 'z'),
    ('a', 'e'),
    ('y', 'i'),
    ('ch', 'g'),
    ('g', 'ch'),
}

char_swap_map = {
    ('c', 'k'),
    ('s', 'z'),
    ('a', 'e'),
    ('y', 'i'),
}


def get_change_info(source: str, dest: str, edit: Tuple[str, int, int],
                    debug: bool = False):
    op, i_source, i_dest = edit
    if op == 'replace':
        return False
    elif op == 'insert':
        change_char = dest[i_dest]
        change_index = i_dest
        change_word = dest
    elif op == 'delete':
        change_char = source[i_source]
        change_index = i_source
        change_word = source
    else:
        print('unknown op:', op)
        raise ValueError(f'unknown edit operation {op}')
    if debug:
        next_char = change_word[change_index + 1] if len(change_word) > change_index + 1 else None
        print(f'is_shortening - {op} - CHANGE_CHAR:', change_char, 'CHANGE_WORD:', change_word, 'NEXT_CHAR:', next_char)
    return change_char, change_index, change_word


def get_alignment_change(source: str, dest: str, op: str, i_source: int, i_dest: int):
    aligned_chunks = []

    if op == 'delete':
        source_head = source[:i_source]
        source_tail = source[i_source + 1:]
        delete_char = source[i_source]
        dest_head = dest[:i_dest]
        dest_tail = dest[i_dest:]
        insert_char = ''
    elif op == 'insert':
        source_head = source[:i_source]
        source_tail = source[i_source:]
        delete_char = ''
        dest_head = dest[:i_dest]
        dest_tail = dest[i_dest+1:]
        insert_char = dest[i_dest]
    else:
        source_head = source[:i_source]
        source_tail = source[i_source + 1:]
        delete_char = source[i_source]
        dest_head = dest[:i_dest]
        dest_tail = dest[i_dest + 1:]
        insert_char = dest[i_dest]
    if len(source_head) > 0 or len(dest_head) > 0:
        aligned_chunks.append({'source': source_head, 'dest': dest_head, 'type': 'aligned'})
    aligned_chunks.append({'source': delete_char, 'dest': insert_char, 'type': op})
    return aligned_chunks, source_tail, dest_tail


def code_diff(source: str, dest: str) -> Tuple[str, str]:
    ops = defaultdict(list)
    for (op, i_source, i_dest) in get_editops(source, dest):
        abb = OPS[op]
        # print('\t\t', op, i_source, i_dest)
        if abb == "#":
            ops["-"].append(i_source)
            ops["+"].append(i_dest)
        elif abb == "+":
            ops[abb].append(i_dest)
        elif abb == "-":
            ops[abb].append(i_source)
    material_min = ""
    prev_i = len(source)
    end_i = len(source) - 1
    for i in sorted(ops["-"]):
        pre = "|" if i == 0 else "." if i > prev_i + 1 else ""
        post = "|" if i == end_i else ""
        material_min += f"{pre}{source[i]}{post}"
        prev_i = i

    material_plus = ""
    prev_i = len(dest)
    end_i = len(dest) - 1
    for i in sorted(ops["+"]):
        pre = "|" if i == 0 else "." if i > prev_i + 1 else ""
        post = "|" if i == end_i else ""
        material_plus += f"{pre}{dest[i]}{post}"
        prev_i = i

    return material_min, material_plus


def is_multi_term(string: str):
    return ' ' in string.strip()


def get_token_terms(string: str):
    return re.split(r'\s+', string.strip())


def is_punct(string: str):
    return all([c in PUNCTUATION for c in string])


def has_punct(string: str):
    return any([c in PUNCTUATION for c in string])


def is_char_swap(source: str, dest: str, edit: Tuple[str, int, int], debug: bool = False) -> bool:
    op, i_source, i_dest = edit
    if op != 'replace':
        return False
    insert_char = dest[i_dest]
    delete_char = source[i_source]
    if debug:
        print(f'is_char_swap - {op} - DELETE_CHAR:', delete_char, 'DELETE_WORD:', source, 'INSERT_CHAR:', insert_char,
              'INSERT_WORD:', dest)
    if (delete_char, insert_char) in char_swap_map:
        char_swap = True
    elif (insert_char, delete_char) in char_swap_map:
        char_swap = True
    else:
        char_swap = False
    if debug:
        print('is_char_swap -', char_swap)
    return char_swap


def is_add_punctuation(dest: str, change_char: str = None,
                       change_word: str = None):
    return is_punct(change_char) and change_word == dest


def is_drop_punctuation(source: str, change_char: str = None,
                        change_word: str = None):
    return is_punct(change_char) and change_word == source


def is_add_whitespace(dest: str, change_char: str = None,
                      change_word: str = None):
    return change_char.isspace() and change_word == dest


def is_drop_whitespace(source: str, change_char: str = None,
                       change_word: str = None):
    return change_char.isspace() and change_word == source


def is_swap_ch_g(source: str, dest: str):
    material_min, material_plus = code_diff(source, dest)
    if material_min == 'ch' and material_plus == 'g':
        return True


def is_swap_g_ch(source: str, dest: str):
    material_min, material_plus = code_diff(source, dest)
    if material_min == 'g' and material_plus == 'ch':
        return True


def is_shortening(source: str, dest: str, edit: Tuple[str, int, int],
                  debug: bool = False) -> bool:
    change_char, change_index, change_word = get_change_info(source, dest, edit)
    if change_char in vowels:
        shortening = is_vowel_shortening(source, dest, edit, change_char=change_char,
                                         change_index=change_index, change_word=change_word,
                                         debug=debug)
    elif change_char == 'c':
        shortening = is_ck_shortening(source, dest, edit, change_char=change_char,
                                      change_index=change_index, change_word=change_word,
                                      debug=debug)
    else:
        shortening = False
    if debug:
        print('is_shortening -', shortening)
    return shortening


def is_ck_shortening(source, dest, edit, change_char: str = None,
                     change_index: int = None, change_word: str = None,
                     debug: bool = False):
    if change_char is None or change_index is None or change_word is None:
        change_char, change_index, change_word = get_change_info(source, dest, edit)
    op, i_source, i_dest = edit
    if debug:
        print('is_ck_shortening - CHANGE_CHAR:', change_char, 'CHANGE_WORD:', change_word, 'NEXT_CHAR:',
              change_word[change_index + 1])
    if change_char != 'c':
        return False
    if change_index < len(change_word) - 1 and change_word[change_index + 1] == 'k':
        if op == 'delete':
            other_context = dest[i_dest - 1:i_dest + 2]
            own_context = source[i_source - 2:i_source]
            if debug:
                print('DEST OTHER CONTEXT:', other_context)
                print('SOURCE OWN CONTEXT:', own_context)
        else:
            other_context = source[i_source - 1:i_source + 2]
            own_context = dest[i_dest - 2:i_dest]
            if debug:
                print('SOURCE OTHER CONTEXT:', other_context)
                print('DEST OWN CONTEXT:', own_context)
        if len(other_context) == 3:
            # other word has single k surrounded by vowels
            # if own context has short vowel preceding c, do not remove it
            # as together with if represents a necessary double kk
            if debug:
                print('\tother:', other_context, other_context[0] in vowels and other_context[2] in vowels)
            if other_context[0] in vowels and other_context[2] in vowels:
                if len(own_context) == 2:
                    # if own context has a long vowel preceding c, remove it
                    # as the double vowel does not need a double k
                    if debug:
                        print('\town:', own_context, own_context[0] in vowels and own_context[1] in vowels)
                    if own_context[0] in vowels and own_context[1] in vowels:
                        return True
                return False
        return True
    else:
        return False


def is_vowel_shortening(source: str, dest: str, edit: Tuple[str, int, int],
                        change_char: str = None, change_index: int = None, change_word: str = None,
                        debug: bool = False):
    if change_char is None or change_index is None or change_word is None:
        change_char, change_index, change_word = get_change_info(source, dest, edit)
    if debug:
        print('is_vowel_shortening - CHANGE_CHAR:', change_char)
    if change_char not in vowels:
        return False
    if change_word[change_index - 1] == change_char:
        return True
    if len(change_word) > change_index + 1 and change_word[change_index + 1] == change_char:
        return True
    elif change_word[change_index - 1] == 'a' and change_char == 'e':
        return True
    else:
        return False


def is_case_swap(source: str, dest: str, edit: Tuple[str, int, int], debug: bool = False) -> bool:
    op, i_source, i_dest = edit
    if op != 'replace':
        return False
    source_char = source[i_source]
    dest_char = dest[i_dest]
    if debug:
        print(f'CASE SWAP - source {source} dest {dest}\tsource index,char: {i_source} {source_char}\t'
              f'dest index,char: {i_dest} {dest_char}')
    return source_char.lower() == dest_char.lower()


def is_variant_edit(source, dest, edit, debug: bool = False):
    if is_shortening(source, dest, edit, debug=debug):
        return True
    elif is_char_swap(source, dest, edit, debug=debug):
        return True
    elif is_case_swap(source, dest, edit, debug=debug):
        return True
    else:
        return False


def swap_order(source, dest, curr_edit, next_edit, debug: bool = False):
    swap_curr_edit, swap_next_edit = None, None
    curr_op, curr_i_source, curr_i_dest = curr_edit
    if next_edit:
        next_op, next_i_source, next_i_dest = next_edit
        if curr_edit[0] == 'insert' and next_edit[0] == 'replace':
            swap_curr_edit = ('replace', curr_i_source, curr_i_dest)
            swap_next_edit = ('insert', next_i_source + 1, next_i_dest)
        elif curr_edit[0] == 'delete' and next_edit[0] == 'replace':
            swap_curr_edit = ('replace', curr_i_source, curr_i_dest)
            swap_next_edit = ('delete', next_i_source, next_i_dest)
        if swap_curr_edit and swap_next_edit:
            curr_class = is_variant_edit(source, dest, curr_edit, debug=debug)
            next_class = is_variant_edit(source, dest, next_edit, debug=debug)
            swap_curr_class = is_variant_edit(source, dest, swap_curr_edit, debug=debug)
            swap_next_class = is_variant_edit(source, dest, swap_next_edit, debug=debug)
            init_score = [curr_class, next_class].count(True)
            swap_score = [swap_curr_class, swap_next_class].count(True)
            if debug:
                print('initial edits:', curr_op, source[curr_i_source], dest[curr_i_dest], ' -> ',
                      next_op, source[next_i_source], dest[next_i_dest])
                print('potential swap:', next_op, source[curr_i_source], dest[curr_i_dest], ' -> ', swap_curr_edit[0],
                      source[swap_curr_edit[1]], dest[swap_curr_edit[2]])
                print('\tinit:', curr_class, next_class, init_score)
                print('\tswap:', swap_curr_class, swap_next_class, swap_score)
            if swap_score > init_score:
                return swap_curr_edit, swap_next_edit
    return None


def optimise_op_order(source: str, dest: str, editops, debug: bool = False):
    """Update the order of edit operations to ensure that they are aligned
    with certain changes in Dutch spelling."""
    if len(editops) < 2:
        return editops
    optimised = []
    swapped = False
    for oi, curr_edit in enumerate(editops):
        if swapped is True:
            # current edit has been swapped with an updated edit
            swapped = False
            continue
        next_edit = editops[oi + 1] if oi + 1 < len(editops) else None
        swap_edits = swap_order(source, dest, curr_edit, next_edit, debug)
        if swap_edits:
            optimised.append(swap_edits[0])
            optimised.append(swap_edits[1])
            swapped = True
        else:
            optimised.append(curr_edit)
    return optimised


def compute_edit_score(source, dest, edit, debug: bool = False):
    if debug:
        print(source, dest, edit)
    try:
        return 0 if is_variant_edit(source, dest, edit, debug=debug) else 1
    except IndexError:
        print(source, dest, edit)
        raise


def compute_variant_dist(source: str, dest: str, debug: bool = False):
    editops = [op for op in get_editops(source, dest)]
    dist = 0
    swapped = False
    for oi, curr_edit in enumerate(editops):
        if swapped is True:
            # current edit has been swapped with an updated edit
            swapped = False
            continue
        next_edit = editops[oi + 1] if oi + 1 < len(editops) else None
        swap_edits = swap_order(source, dest, curr_edit, next_edit, debug)
        if swap_edits:
            dist += compute_edit_score(source, dest, swap_edits[0], debug=debug)
            dist += compute_edit_score(source, dest, swap_edits[1], debug=debug)
            swapped = True
        else:
            dist += compute_edit_score(source, dest, curr_edit, debug=debug)
    return dist


def compute_variant_similarity(w1: str, w2: str) -> float:
    dist = compute_variant_dist(w1, w2)
    return 1 - (dist / min(len(w1), len(w2)))


def classify_diff(source: str, dest: str, debug: bool = False):
    spelling_variation_diffs = []
    editops = [op for op in get_editops(source, dest)]
    editops = optimise_op_order(source, dest, editops, debug=debug)
    for edit in editops:
        change_char, change_index, change_word = get_change_info(source, dest, edit)
        if is_variant_edit(source, dest, edit, debug=debug):
            spelling_variation_diffs.append(True)
        else:
            spelling_variation_diffs.append(False)
    return spelling_variation_diffs
