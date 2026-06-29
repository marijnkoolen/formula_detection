"""edit.py — Edit-distance-based spelling-variant detection for historical Dutch text.

Historical Dutch orthography varies in systematic ways: vowel length is
marked inconsistently (single vs. doubled vowel before a consonant),
"ck" vs "c"/"k" alternates for the same /k/ sound, certain consonant
pairs are interchangeable (c/k, s/z, a/e, y/i, ch/g), casing of the same
word may differ, and punctuation/whitespace is inserted or dropped at
word boundaries. A plain Levenshtein edit distance treats all of these
the same as an arbitrary typo, which inflates the apparent distance
between two strings that are, historically, the same word.

This module inspects the individual edit operations (insert/delete/replace)
returned by python-Levenshtein's `editops` and classifies each one as
either a "spelling variant" edit (consistent with one of the known
historical patterns above) or a "real" edit. `compute_variant_dist` then
scores variant edits as free (0) and real edits as costing 1, giving a
distance metric that is robust to known historical spelling variation.
`optimise_op_order`/`swap_order` additionally reorder adjacent edit
operations before classification, since `editops` can produce an
operation order that obscures a recognisable variant pattern (e.g. an
insert+replace pair that is really a single character swap once
reordered as replace+insert).
"""
import re
from collections import defaultdict
from string import punctuation
from typing import Tuple

from Levenshtein import distance as edit_distance
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
    """Extract the changed character, its index and its containing word for an edit op.

    For an 'insert' op the change is located in `dest`; for a 'delete' op it
    is located in `source`. 'replace' ops are not handled (there is no
    single inserted/deleted character) and short-circuit to False.

    Args:
        source: The source string.
        dest: The destination string.
        edit: A (op, i_source, i_dest) edit operation as returned by
            `Levenshtein.editops`, where op is 'replace', 'insert' or 'delete'.
        debug: If True, print diagnostic information about the change.

    Returns:
        False if op is 'replace'. Otherwise a tuple
        (change_char, change_index, change_word) where change_word is
        whichever of `source`/`dest` contains the inserted/deleted character.

    Raises:
        ValueError: If op is not one of 'replace', 'insert', 'delete'.
    """
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
    """Split source/dest into an aligned prefix chunk and a changed chunk, plus remaining tails.

    Args:
        source: The source string.
        dest: The destination string.
        op: The edit operation type ('delete', 'insert', or any other value,
            treated as a replace-like operation).
        i_source: Index of the affected character in `source`.
        i_dest: Index of the affected character in `dest`.

    Returns:
        A tuple (aligned_chunks, source_tail, dest_tail) where aligned_chunks
        is a list of dicts with keys 'source', 'dest', 'type' describing the
        (optional) unchanged prefix chunk followed by the changed chunk, and
        source_tail/dest_tail are the remaining unprocessed suffixes of
        `source` and `dest`.
    """
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
    """Summarise the deleted and inserted character material between two strings.

    Computes the editops between `source` and `dest` and builds two compact
    strings: one listing the characters removed from `source`
    (`material_min`) and one listing the characters inserted into `dest`
    (`material_plus`). Each contiguous block of changed characters is
    wrapped with '|' markers if it touches the start/end of the string, and
    non-contiguous blocks are separated with '.'. This gives a short
    representation of "what changed" useful for pattern matching against
    known spelling-variant rules (e.g. 'ch' -> 'g').

    Args:
        source: The source string.
        dest: The destination string.

    Returns:
        A tuple (material_min, material_plus): the encoded deleted material
        from `source` and the encoded inserted material into `dest`.
    """
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
    """Check whether a string contains more than one whitespace-separated term.

    Args:
        string: The string to check.

    Returns:
        True if `string` (after stripping) contains a space character.
    """
    return ' ' in string.strip()


def get_token_terms(string: str):
    """Split a string into whitespace-separated tokens.

    Args:
        string: The string to split.

    Returns:
        A list of token strings.
    """
    return re.split(r'\s+', string.strip())


def is_punct(string: str):
    """Check whether every character in a string is punctuation.

    Args:
        string: The string to check.

    Returns:
        True if all characters of `string` are in PUNCTUATION.
    """
    return all([c in PUNCTUATION for c in string])


def has_punct(string: str):
    """Check whether a string contains at least one punctuation character.

    Args:
        string: The string to check.

    Returns:
        True if any character of `string` is in PUNCTUATION.
    """
    return any([c in PUNCTUATION for c in string])


def is_char_swap(source: str, dest: str, edit: Tuple[str, int, int], debug: bool = False) -> bool:
    """Check whether a 'replace' edit matches a known historical character-swap pair.

    Known swap pairs (in either direction) are defined in `char_swap_map`:
    c/k, s/z, a/e, y/i. These are interchangeable spellings in historical
    Dutch orthography, so a replace between them is treated as a spelling
    variant rather than a real edit.

    Args:
        source: The source string.
        dest: The destination string.
        edit: A (op, i_source, i_dest) edit operation. Only 'replace' ops
            can be a char swap.
        debug: If True, print diagnostic information about the comparison.

    Returns:
        True if the edit is a 'replace' op whose source/dest characters form
        one of the known swap pairs (in either direction); False otherwise.
    """
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
    """Check whether an edit is purely the insertion of a punctuation character into `dest`.

    Args:
        dest: The destination string.
        change_char: The inserted/deleted character, typically from
            `get_change_info`.
        change_word: The string the change belongs to (should be `dest` for
            this check to be meaningful).

    Returns:
        True if `change_char` is punctuation and `change_word` is `dest`.
    """
    return is_punct(change_char) and change_word == dest


def is_drop_punctuation(source: str, change_char: str = None,
                        change_word: str = None):
    """Check whether an edit is purely the removal of a punctuation character from `source`.

    Args:
        source: The source string.
        change_char: The inserted/deleted character, typically from
            `get_change_info`.
        change_word: The string the change belongs to (should be `source`
            for this check to be meaningful).

    Returns:
        True if `change_char` is punctuation and `change_word` is `source`.
    """
    return is_punct(change_char) and change_word == source


def is_add_whitespace(dest: str, change_char: str = None,
                      change_word: str = None):
    """Check whether an edit is purely the insertion of a whitespace character into `dest`.

    Args:
        dest: The destination string.
        change_char: The inserted/deleted character, typically from
            `get_change_info`.
        change_word: The string the change belongs to (should be `dest` for
            this check to be meaningful).

    Returns:
        True if `change_char` is whitespace and `change_word` is `dest`.
    """
    return change_char.isspace() and change_word == dest


def is_drop_whitespace(source: str, change_char: str = None,
                       change_word: str = None):
    """Check whether an edit is purely the removal of a whitespace character from `source`.

    Args:
        source: The source string.
        change_char: The inserted/deleted character, typically from
            `get_change_info`.
        change_word: The string the change belongs to (should be `source`
            for this check to be meaningful).

    Returns:
        True if `change_char` is whitespace and `change_word` is `source`.
    """
    return change_char.isspace() and change_word == source


def is_swap_ch_g(source: str, dest: str):
    """Check whether the difference between source and dest is exactly 'ch' replaced by 'g'.

    Args:
        source: The source string.
        dest: The destination string.

    Returns:
        True if `code_diff` reports deleted material 'ch' and inserted
        material 'g'; None otherwise (implicit falsy return).
    """
    material_min, material_plus = code_diff(source, dest)
    if material_min == 'ch' and material_plus == 'g':
        return True


def is_swap_g_ch(source: str, dest: str):
    """Check whether the difference between source and dest is exactly 'g' replaced by 'ch'.

    Args:
        source: The source string.
        dest: The destination string.

    Returns:
        True if `code_diff` reports deleted material 'g' and inserted
        material 'ch'; None otherwise (implicit falsy return).
    """
    material_min, material_plus = code_diff(source, dest)
    if material_min == 'g' and material_plus == 'ch':
        return True


def is_shortening(source: str, dest: str, edit: Tuple[str, int, int],
                  debug: bool = False) -> bool:
    """Check whether an edit corresponds to a historical vowel- or ck-shortening pattern.

    Dispatches to `is_vowel_shortening` if the changed character is a vowel,
    or to `is_ck_shortening` if it is a 'c' (covering the ck/c/k
    alternation). Any other changed character is not a shortening pattern.

    Args:
        source: The source string.
        dest: The destination string.
        edit: A (op, i_source, i_dest) edit operation.
        debug: If True, print diagnostic information about the classification.

    Returns:
        True if the edit matches a vowel- or ck-shortening pattern; False
        otherwise.
    """
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
    """Check whether an edit involving 'c' is a historical ck/c/k spelling alternation.

    Historical Dutch sometimes writes the /k/ sound as 'ck' and sometimes
    as a single 'c' or 'k'; this is only a real alternation (rather than a
    coincidental edit) when the 'c' is immediately followed by a 'k' in its
    own word. When that holds, the surrounding vowel context of the *other*
    word (the one without the extra 'c') is inspected: if the other word's
    'k' sits between two vowels (a single intervocalic k, i.e. it does not
    need doubling) and the changed word's own preceding context is a single
    short vowel followed by the consonant (own_context length 2 with both
    a vowel and the consonant in front of 'c'), the alternation is treated
    as a real spelling difference (not a free shortening) because the short
    vowel requires the doubled 'kk' for correct pronunciation; otherwise it
    counts as a shortening variant.

    Args:
        source: The source string.
        dest: The destination string.
        edit: A (op, i_source, i_dest) edit operation.
        change_char: The inserted/deleted character; computed from
            `get_change_info` if not given.
        change_index: Index of `change_char` within `change_word`; computed
            from `get_change_info` if not given.
        change_word: The string (`source` or `dest`) containing the change;
            computed from `get_change_info` if not given.
        debug: If True, print diagnostic information about the contexts
            being compared.

    Returns:
        True if the edit is a recognised ck-shortening spelling variant;
        False otherwise.
    """
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
    """Check whether an inserted/deleted vowel matches a historical vowel-doubling/shortening pattern.

    Historical Dutch sometimes marks a long vowel by doubling it (e.g. 'aa')
    where the modern/other spelling uses a single vowel. This is detected
    when the changed character is itself a vowel and either: it is
    identical to its immediate predecessor in `change_word` (i.e. it is
    part of a doubled vowel, so removing/adding it just changes the vowel
    length marking), it is identical to its immediate successor, or the
    preceding character is 'a' and the changed character is 'e' (the
    historical 'ae' digraph alternation).

    Args:
        source: The source string.
        dest: The destination string.
        edit: A (op, i_source, i_dest) edit operation.
        change_char: The inserted/deleted character; computed from
            `get_change_info` if not given.
        change_index: Index of `change_char` within `change_word`; computed
            from `get_change_info` if not given.
        change_word: The string (`source` or `dest`) containing the change;
            computed from `get_change_info` if not given.
        debug: If True, print diagnostic information about `change_char`.

    Returns:
        True if the edit matches a recognised vowel-shortening pattern;
        False otherwise (including when `change_char` is not a vowel).
    """
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
    """Check whether a 'replace' edit is purely a case difference (same letter, different case).

    Args:
        source: The source string.
        dest: The destination string.
        edit: A (op, i_source, i_dest) edit operation. Only 'replace' ops
            can be a case swap.
        debug: If True, print diagnostic information about the comparison.

    Returns:
        True if the edit is a 'replace' op where the source and dest
        characters are the same letter differing only in case; False
        otherwise.
    """
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
    """Check whether an edit operation matches any recognised historical spelling-variant pattern.

    Combines the shortening, character-swap, and case-swap checks: an edit
    is considered a "free" spelling variant (rather than a real edit) if it
    is a vowel/ck shortening, a known character swap (c/k, s/z, a/e, y/i),
    or a case difference.

    Args:
        source: The source string.
        dest: The destination string.
        edit: A (op, i_source, i_dest) edit operation.
        debug: If True, print diagnostic information from the underlying checks.

    Returns:
        True if the edit matches any recognised spelling-variant pattern;
        False otherwise.
    """
    if is_shortening(source, dest, edit, debug=debug):
        return True
    elif is_char_swap(source, dest, edit, debug=debug):
        return True
    elif is_case_swap(source, dest, edit, debug=debug):
        return True
    else:
        return False


def swap_order(source, dest, curr_edit, next_edit, debug: bool = False):
    """Try reordering two adjacent edit ops to see if that reveals a recognisable spelling-variant pattern.

    `editops` sometimes encodes what is conceptually a single character
    substitution as an adjacent insert+replace or delete+replace pair (in
    that order), which `is_variant_edit` cannot recognise in its original
    form. This function builds the equivalent swapped pair — turning
    insert+replace into replace+insert (shifted) or delete+replace into
    replace+delete — and compares how many of the two ops classify as
    spelling variants before vs. after the swap. The swap is only adopted
    if it increases the variant count, i.e. it makes the edit pair look
    more like a known historical spelling pattern.

    Args:
        source: The source string.
        dest: The destination string.
        curr_edit: The current (op, i_source, i_dest) edit operation.
        next_edit: The following (op, i_source, i_dest) edit operation, or
            None if `curr_edit` is the last operation.
        debug: If True, print diagnostic information about the comparison.

    Returns:
        A tuple (swap_curr_edit, swap_next_edit) with the reordered edit
        operations if the swap improves the variant classification count;
        None if no swap is applicable or the swap does not improve on the
        original order.
    """
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
    """Update the order of edit operations to ensure that they are aligned with certain changes in Dutch spelling.

    Walks through the list of edit operations and, for each adjacent pair,
    asks `swap_order` whether reordering them better reveals a known
    historical spelling-variant pattern. Operations that get swapped are
    consumed in pairs so they are not reprocessed.

    Args:
        source: The source string.
        dest: The destination string.
        editops: A list of (op, i_source, i_dest) edit operations, as
            returned by `Levenshtein.editops`.
        debug: If True, print diagnostic information from `swap_order`.

    Returns:
        A list of edit operations with adjacent pairs reordered where doing
        so improves spelling-variant recognition. Returns `editops`
        unchanged if it has fewer than 2 operations.
    """
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


def compute_edit_score(source, dest, edit, debug: int = 0):
    """Score a single edit operation as 0 (spelling variant) or 1 (real edit).

    Args:
        source: The source string.
        dest: The destination string.
        edit: A (op, i_source, i_dest) edit operation.
        debug: Truthy/nonzero to print diagnostic information.

    Returns:
        0 if `is_variant_edit` classifies the edit as a recognised spelling
        variant; 1 otherwise.

    Raises:
        IndexError: Re-raised (after printing diagnostic context) if the
        edit's indices are out of bounds for `source`/`dest`.
    """
    if debug > 0:
        print(source, dest, edit)
    try:
        return 0 if is_variant_edit(source, dest, edit, debug=debug) else 1
    except IndexError:
        print(source, dest, edit)
        raise


def compute_variant_dist(source: str, dest: str, use_editops: bool = False, debug: int = 0):
    """Compute an edit distance between two strings that ignores known historical spelling variation.

    When `use_editops` is True, each individual edit operation between
    `source` and `dest` (after `swap_order`-based reordering of adjacent
    pairs) is scored via `compute_edit_score`: spelling-variant edits cost
    0 and real edits cost 1, and the distance is the sum of these scores.
    When `use_editops` is False, the plain Levenshtein edit distance is
    returned instead.

    Args:
        source: The source string.
        dest: The destination string.
        use_editops: If True, use the spelling-variant-aware scoring based
            on individual edit operations; if False, use plain Levenshtein
            distance.
        debug: Truthy/nonzero to print diagnostic information from the
            underlying classification functions.

    Returns:
        The (possibly variant-adjusted) edit distance between `source` and
        `dest`, as an int.
    """
    if use_editops is True:
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
    else:
        dist = edit_distance(source, dest)
    return dist


def compute_variant_similarity(w1: str, w2: str, debug: int = 0) -> float:
    """Compute a normalised similarity score between two words, ignoring spelling variation.

    Uses `compute_variant_dist` (with `use_editops` defaulting to False,
    i.e. plain Levenshtein distance) normalised by the length of the
    shorter word.

    Args:
        w1: The first word.
        w2: The second word.
        debug: Truthy/nonzero to print diagnostic information from the
            underlying distance computation.

    Returns:
        A similarity score as 1 minus the normalised edit distance; 1.0
        means identical, lower values mean more different.
    """
    dist = compute_variant_dist(w1, w2, debug=debug)
    return 1 - (dist / min(len(w1), len(w2)))


def classify_diff(source: str, dest: str, debug: bool = False):
    """Classify each edit operation between two strings as a spelling variant or a real edit.

    Computes the editops between `source` and `dest`, reorders adjacent
    pairs via `optimise_op_order` where that better reveals a known
    spelling pattern, then classifies each (possibly reordered) operation
    with `is_variant_edit`.

    Args:
        source: The source string.
        dest: The destination string.
        debug: If True, print diagnostic information from the underlying
            classification functions.

    Returns:
        A list of booleans, one per edit operation (in optimised order),
        True if that operation is a recognised spelling variant and False
        if it is a real edit.
    """
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
