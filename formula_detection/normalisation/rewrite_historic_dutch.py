"""
rewrite_historic_dutch.py — Rule-based rewriting of historical Dutch
spelling variants to their modern(-ish) equivalents.

Historical Dutch text uses a range of spelling conventions that have since
been standardised away: doubled consonants written with 'ck', the long 'a'
written as 'ae', the 'g'/'ch' digraph 'gh', and the diphthongs/glides 'ey',
'uy' and 'y' that map onto modern 'ei', 'ui'/'uu' and 'ij'/'i'/'y'
respectively, plus a handful of word-final 't'-spellings (e.g. '-dt',
'-heit', '-lant'/'-landt'). Each ``replace_*`` function implements the
character-context rules for one such variant, generally by splitting the
word on the historical n-gram and deciding, n-gram occurrence by n-gram
occurrence, what the surrounding letters imply the modern spelling should
be. A number of known exceptions (place names, loanwords, Latin phrases,
etc.) are hardcoded and returned unchanged or rewritten as a special case.

``normalise_spelling`` chains these per-variant rewrites into a single
pass over a word, and ``normalise_word`` wraps that with optional ASCII
folding and an optional dictionary lookup (e.g. a fuzzy-matched spelling
variant dictionary) consulted before falling back to the rule-based
rewrite.
"""
import unicodedata
from typing import Dict


NGRAM_REPLACEMENTS_NL = [
    {'orig': 'ff', 'replace': 'f'},
    {'orig': 'ue', 'replace': 've'},
    {'orig': 'ua', 'replace': 'va'},
    {'orig': 'uo', 'replace': 'vo'},
    {'orig': 'ui', 'replace': 'vi'},
    {'orig': 'ur', 'replace': 'vr'},
    {'orig': 'uu', 'replace': 'vu'},
    {'orig': 'cx', 'replace': 'cks'},
]


def unicode_to_ascii(s):
    """Turn a Unicode string to plain ASCII by stripping diacritics.

    Decomposes the string (NFD) and drops all combining marks (Unicode
    category 'Mn'), e.g. turning 'é' into 'e'.

    Source: https://stackoverflow.com/a/518232/2809427

    Args:
        s: The string to strip diacritics from.

    Returns:
        The input string with combining diacritical marks removed.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def replace_ck(word: str) -> str:
    """Rewrite historical 'ck' spellings to modern 'k' or 'kk'.

    Splits the word on every 'ck' occurrence and decides, for each split
    point, whether 'ck' becomes 'k' or 'kk' based on the letters
    immediately before and after it:

    - If 'ck' is at the start of the word (nothing precedes it), or the
      preceding letter is a consonant, or the two letters preceding it are
      both vowels (a double vowel), it becomes 'k'.
    - If 'ck' is preceded by a single vowel and followed by a consonant
      (or by nothing), it also becomes 'k'.
    - Otherwise (preceded by a single vowel and followed by another
      vowel), it becomes 'kk'.

    Vowels considered are a, e, i, o, u ('y' is treated as a diphthong and
    excluded).

    Args:
        word: The word to rewrite.

    Returns:
        The word with every 'ck' occurrence rewritten to 'k' or 'kk'.
    """
    vowels = {'a', 'e', 'i', 'o', 'u'}  # 'y' is a diphthong
    parts = word.split('ck')
    rewrite_word = ''
    for pi, curr_part in enumerate(parts[:-1]):
        rewrite_word += curr_part
        next_part = parts[pi + 1]
        # print('curr_part:', curr_part, 'next_part:', next_part, 'rewrite_word:', rewrite_word)
        if len(curr_part) == 0:
            rewrite_word += 'k'
        elif curr_part[-1].lower() not in vowels:
            # ck after a consonant becomes k
            rewrite_word += 'k'
        elif curr_part[-1].lower() in vowels and len(curr_part) >= 2 and curr_part[-2].lower() in vowels:
            # ck after a double vowel becomes k
            rewrite_word += 'k'
        elif len(next_part) == 0 or next_part[0].lower() not in vowels:
            # ck after single vowel and before a consonant becomes k
            rewrite_word += 'k'
        else:
            # ck after a single vowel and before a vowel becomes kk
            rewrite_word += 'kk'
    rewrite_word += parts[-1]
    return rewrite_word


def replace_ae(word: str) -> str:
    """Rewrite historical 'ae' spellings to modern 'aa' (with exceptions).

    A leading 'Ae' (length > 2) is first normalised to 'Aa'. The word
    'lunae' (case-insensitively) is returned unchanged, since it is Latin
    rather than historical Dutch. Otherwise the word is split on every
    'ae' occurrence and, for each occurrence:

    - If it is the first occurrence and is immediately preceded by 'pr'
      (case-sensitive 2-letter match), it is treated as a Latin loanword:
      the literal word 'prael' is special-cased to 'praal', any other
      'pr...ae' word gets 'ae' rewritten to 'e'.
    - If what has been rewritten so far is exactly 'portug' (i.e. the word
      starts with 'Portugae...'), 'ae' becomes 'a' (-> 'Portugal'-style).
    - Otherwise 'ae' becomes 'aa'.

    Args:
        word: The word to rewrite.

    Returns:
        The word with 'ae' occurrences rewritten per the rules above.
    """
    if len(word) > 2 and word.startswith('Ae'):
        word = 'Aa' + word[2:]
    parts = word.split('ae')
    rewrite_word = ''
    if word.lower() == 'lunae':
        return word
    for pi, curr_part in enumerate(parts[:-1]):
        rewrite_word += curr_part
        if pi == 0 and len(curr_part) == 2 and curr_part.lower() == 'pr':
            if word == 'prael':
                return 'praal'
            else:
                # latin phrase so ae becomes 'e'
                rewrite_word += 'e'
        elif rewrite_word.lower() == 'portug':
            rewrite_word += 'a'
        else:
            rewrite_word += 'aa'
    rewrite_word += parts[-1]
    return rewrite_word


def replace_gh(word: str) -> str:
    """Rewrite historical 'gh' spellings to modern 'g', 'ch', or kept 'gh'.

    Hardcoded exceptions are checked first: 'vught' (case-insensitive) is
    returned unchanged, and 'dight'/'sigh' (case-insensitive) have their
    'gh' replaced with 'ch' directly.

    Otherwise the word is split on every 'gh' occurrence and, for each
    occurrence, the surrounding text determines the replacement:

    - If the text right after 'gh' starts with 'eid', 'eit', 'eyd' or
      'eyt' (case-insensitive), 'gh' is kept as-is.
    - Else if it starts with 'uy' or 'ui' (case-insensitive), 'gh' is kept
      as-is.
    - Else if it starts with 't' (case-insensitive), the text right before
      'gh' decides further:
        - ends with 'sle' -> 'ch'
        - ends with one of 'vol', 'voe', 'voo', 'ver', 'lee', 'laa',
          'lan', 'len', 'raa', 'rey', 'rei', 'haa', 'hoo', 'tuy', 'tui'
          (last 3 chars) -> 'g'
        - ends with 'le', 'ti', 'di', 'ni', 'se' (last 2 chars) -> 'g'
        - ends with 're' (last 2 chars) -> 'ch'
        - otherwise -> 'ch' (e.g. 'gevoecht')
    - Else if the text right before 'gh' ends with 'rou' (last 3 chars),
      'gh' is kept as-is.
    - Else if the text right before 'gh' ends with 'li' (last 2 chars),
      'gh' becomes 'g' if 'gh' is at the end of the word, otherwise 'ch'.
    - Otherwise, 'gh' becomes 'g'.

    Args:
        word: The word to rewrite.

    Returns:
        The word with 'gh' occurrences rewritten per the rules above.
    """
    if word.lower() in {'vught'}:
        return word
    if word.lower() in {'dight', 'sigh'}:
        return word.replace('gh', 'ch')
    parts = word.split('gh')
    rewrite_word = ''
    for pi, curr_part in enumerate(parts[:-1]):
        next_part = parts[pi + 1]
        rewrite_word += curr_part
        if len(next_part) >= 3 and next_part[:3].lower() in {'eid', 'eit', 'eyd', 'eyt'}:
            rewrite_word += 'gh'
        elif len(next_part) >= 2 and next_part[:2].lower() in {'uy', 'ui'}:
            rewrite_word += 'gh'
        elif len(next_part) >= 1 and next_part[0].lower() == 't':
            if len(curr_part) >= 3 and curr_part[-3:].lower() in {'sle'}:
                rewrite_word += 'ch'
            elif len(curr_part) >= 3 and curr_part[-3:].lower() in {'vol', 'voe', 'voo', 'ver', 'lee', 'laa', 'lan',
                                                                    'len', 'raa', 'rey', 'rei', 'haa', 'hoo', 'tuy',
                                                                    'tui'}:
                rewrite_word += 'g'
            elif len(curr_part) >= 2 and curr_part[-2:].lower() in {'le', 'ti', 'di', 'ni', 'se'}:
                rewrite_word += 'g'
            elif len(curr_part) >= 2 and curr_part[-2:].lower() == 're':
                rewrite_word += 'ch'
            # gevoecht
            else:
                rewrite_word += 'ch'
        elif len(curr_part) >= 3 and curr_part[-3:].lower() in {'rou'}:
            rewrite_word += 'gh'
        elif len(curr_part) >= 2 and curr_part[-2:].lower() in {'li'}:
            if next_part == '':
                rewrite_word += 'g'
            else:
                rewrite_word += 'ch'
        else:
            rewrite_word += 'g'
    rewrite_word += parts[-1]
    return rewrite_word


def replace_ey(word: str) -> str:
    """Rewrite historical 'ey' spellings to modern 'ei' (with exceptions).

    A leading 'Ey' is first normalised to 'Ei'. If the (possibly already
    'Ei'-normalised) word exactly matches one of the hardcoded exceptions
    {"Hoey", "Bey", "Dey", "Peyrou", "Beyer", "Orkney"}, it is returned
    unchanged.

    Otherwise the word is split on every 'ey' occurrence and, for each
    occurrence with text following it:

    - If the following text starts with 'ck', 'ey' becomes 'e' when the
      preceding character is 't', otherwise it becomes 'ei'.
    - Else if the preceding character is 'l' or 'o', 'ey' becomes 'ei'.
    - Else if the following character is one of
      'c', 'd', 'g', 'k', 'l', 'm', 'n', 's', 't', 'z', 'ey' becomes 'ei'.
    - Otherwise 'ey' is kept as 'ey'.

    If an 'ey' occurrence is at the end of the word (nothing follows),
    it becomes 'ei'.

    Args:
        word: The word to rewrite.

    Returns:
        The word with 'ey' occurrences rewritten per the rules above, or
        unchanged if it is a hardcoded exception.
    """
    if word.startswith('Ey'):
        word = 'Ei' + word[2:]
    parts = word.split('ey')
    rewrite_word = ''
    exceptions = {"Hoey", "Bey", "Dey", "Peyrou", "Beyer", "Orkney"}
    if word in exceptions:
        return word
    for pi, curr_part in enumerate(parts[:-1]):
        rewrite_word += curr_part
        if len(parts) > pi + 1 and len(parts[pi + 1]) > 0:
            next_part = parts[pi + 1]
            if len(next_part) >= 2 and next_part[:2] in {'ck'}:
                if len(curr_part) >= 1 and curr_part[-1] in {'t'}:
                    rewrite_word += 'e'
                else:
                    rewrite_word += 'ei'
            elif len(curr_part) >= 1 and curr_part[-1:] in {'l', 'o'}:
                rewrite_word += 'ei'
            elif len(next_part) > 0 and next_part[0] in {'c', 'd', 'g', 'k', 'l', 'm', 'n', 's', 't', 'z'}:
                rewrite_word += 'ei'
            else:
                rewrite_word += 'ey'
        else:
            rewrite_word += 'ei'
    rewrite_word += parts[-1]
    return rewrite_word


def replace_uy(word: str) -> str:
    """Rewrite historical 'uy' spellings to modern 'ui' or 'uu' (with exceptions).

    If the word is exactly one of the hardcoded exceptions
    {'Huy', 'Guy', 'Tuyl', 'Stuyling', 'celuy', 'Vauguyon', 'Uytters'}, it
    is returned unchanged.

    If the word starts with 'Uy', it is special-cased: 'Uytrecht' and
    'Uytregt' are rewritten directly to 'Utrecht'; any other 'Uy'-initial
    word has its leading 'Uy' normalised to 'Ui' before further
    processing.

    Otherwise the word is split on every 'uy' occurrence and, for each
    occurrence with text following it:

    - If the preceding text ends with 'app' (last 3 chars), 'uy' is kept
      as 'uy'.
    - Else if the following character is 'r', 'uy' becomes 'uu'.
    - Else if the following text starts with 'cl', 'uy' is kept as 'uy'.
    - Else if the following text starts with 'k' or 'ck', 'uy' is kept as
      'uy' if the preceding character is 'c', 'k', 'C' or 'K', otherwise
      it becomes 'ui'.
    - Otherwise 'uy' becomes 'ui'.

    If a 'uy' occurrence is at the end of the word (nothing follows), it
    becomes 'ui'.

    Args:
        word: The word to rewrite.

    Returns:
        The word with 'uy' occurrences rewritten per the rules above, or
        unchanged/normalised per the hardcoded exceptions above.
    """
    exceptions = {'Huy', 'Guy', 'Tuyl', 'Stuyling', 'celuy', 'Vauguyon', 'Uytters'}
    if word in exceptions:
        return word
    if word[:2] == 'Uy':
        if word in {'Uytrecht', 'Uytregt'}:
            return 'Utrecht'
        else:
            word = 'Ui' + word[2:]
    parts = word.split('uy')
    rewrite_word = ''
    for pi, curr_part in enumerate(parts[:-1]):
        rewrite_word += curr_part
        if len(parts) > pi + 1 and len(parts[pi + 1]) > 0:
            next_part = parts[pi + 1]
            if len(curr_part) >= 3 and curr_part.endswith('app'):
                rewrite_word += 'uy'
            elif len(next_part) > 0 and next_part[0] in {'r'}:
                rewrite_word += 'uu'
            elif len(next_part) > 1 and next_part[:2] in {'cl'}:
                rewrite_word += 'uy'
            elif next_part.startswith('k') or next_part.startswith('ck'):
                if len(curr_part) > 0 and curr_part[-1] in {'c', 'k', 'C', 'K'}:
                    rewrite_word += 'uy'
                else:
                    rewrite_word += 'ui'
            else:
                rewrite_word += 'ui'
        else:
            rewrite_word += 'ui'
    rewrite_word += parts[-1]
    return rewrite_word


def replace_y(word: str) -> str:
    """Rewrite historical 'y' spellings to modern 'ij', 'i' or 'y'.

    Run after replace_ey/replace_uy, so most remaining 'y' occurrences are
    not part of an 'ey' or 'uy' digraph (those are normally already
    resolved); this function decides what a bare 'y' should become.

    A leading capital 'Y' is temporarily lower-cased (and restored at the
    end) so the splitting/matching logic below is case-uniform. If the
    word is exactly one of the hardcoded exceptions {'Haye', 'Hoey',
    'Meyerye', 'Dey', 'Bey', 'Pays', 'payer', 'Bayreuth', 'Jacoby', 'York',
    'york'}, it is returned unchanged.

    Otherwise the word is split on every 'y' occurrence. For each
    occurrence (with `curr_part` the text since the previous split point,
    lower-cased, and `next_part` the text up to the next 'y' or the end of
    the word), the replacement is chosen by the first matching rule:

    1. `curr_part` ends with 'u' or 'e' -> keep 'y' (an unresolved 'uy'/
       'ey' digraph should not be touched here).
    2. Rewritten-so-far is exactly 'Baronn' -> 'i' (Baronnye -> Baronnie).
    3. Rewritten-so-far is exactly 'Jul' or 'Jun' -> 'i' (Juny/July ->
       Juni/Juli).
    4. `curr_part` ends with 'hoo', 'koo', 'doo', 'moo', 'noo' or 'foo'
       (last 3 chars) -> 'i' (e.g. hooy -> hooi, dooyen -> dooien).
    5. `curr_part` ends with 'troo' (last 4 chars) -> 'i' (octrooy ->
       octrooi).
    6. This is the first split (pi == 0), `curr_part` is empty, and the
       next part starts with 'e' -> 'i' (yemand -> iemand).
    7. This is the first split, `curr_part` is empty, and the next part
       starts with 'r' -> 'ie' (Yrland -> Ierland).
    8. `curr_part` ends with 'o' or 'on' -> keep 'y'.
    9. If there is a non-empty next part, further sub-rules apply:
       - `curr_part` ends with 'a' and next part starts with 'r' -> 'i'
         (ayr -> air).
       - next part starts with 'ork' -> keep 'y' (York stays York).
       - `curr_part` ends with 'pl'/'Pl' and next part starts with 'm'
         -> keep 'y' (Plym... as in Plymouth).
       - `curr_part` ends with 'g' and next part starts with 'p' -> keep
         'y' (gyp, as in Egypten).
       - `curr_part` ends with 'e': if next part starts with 'er' -> keep
         'y' (eyer -> eier); otherwise -> 'i' (this branch documents that
         it should never be reached, since replace_ey already handles
         'ey').
       - `curr_part` ends with 'r' and next part starts with 'e' -> 'i'
         (rye -> rie, as in artillerye -> artillerie).
       - `curr_part` ends with 'aa' -> 'i'.
       - `curr_part` ends with 'a' -> keep 'y'.
       - otherwise -> 'ij'.
    10. If there is no next part (end of word) and `curr_part` ends with
        'lar', 'nar', 'tar' or 'ist' (last 3 chars) -> 'ie'.
    11. If there is no next part and `curr_part` ends with 'uar' or 'ust'
        (last 3 chars) -> keep 'y'.
    12. If there is no next part and `curr_part` ends with 'ar' (checked
        via the last 3 characters, despite slicing 3 chars for what is
        described as a 2-char suffix) -> keep 'y'.
    13. `curr_part` ends with 'er' -> 'ij'.
    14. `curr_part` ends with 'nn', 'rr' or 'ic' -> keep 'y'.
    15. `curr_part` ends with 'aa' -> 'i'.
    16. `curr_part` ends with 'a' -> keep 'y'.
    17. `curr_part` ends with 'b' -> 'ij'.
    18. Rewritten-so-far is exactly 'h', 's', 'z', 'H', 'S' or 'Z' -> 'ij'
        (hy/sy/zy/Hy/Sy/Zy -> hij/sij/zij/Hij/Sij/Zij).
    19. Otherwise -> keep 'y'.

    If the original word started with a capital 'Y', the rewritten word's
    leading character(s) are recapitalised: 'ij' -> 'IJ', 'i' -> 'I', 'y'
    -> 'Y'. If the rewritten word starts with none of these, a ValueError
    is raised.

    Args:
        word: The word to rewrite.

    Returns:
        The word with 'y' occurrences rewritten per the rules above, or
        unchanged if it is a hardcoded exception.

    Raises:
        ValueError: If the original word started with 'Y' but the
            rewritten word does not start with 'ij', 'i' or 'y', so the
            capitalisation cannot be restored.
    """
    if word[0] == 'Y':
        capital_y = True
        word = 'y' + word[1:]
    else:
        capital_y = False
    parts = word.split('y')
    rewrite_word = ''
    exceptions = {
        'Haye', 'Hoey', 'Meyerye', 'Dey', 'Bey', 'Pays', 'payer', 'Bayreuth', 'Jacoby',
        'York', 'york'
    }
    if word in exceptions:
        return word
    for pi, curr_part in enumerate(parts[:-1]):
        rewrite_word += curr_part
        curr_part = curr_part.lower()
        if len(curr_part) >= 1 and curr_part[-1] in {'u', 'e'}:
            # if 'ey' and 'uy' are not replaced, don't replace 'y' now
            rewrite_word += 'y'
        elif rewrite_word in {'Baronn'}:
            # Baronnye -> Baronnie
            rewrite_word += 'i'
        elif rewrite_word in {'Jul', 'Jun'}:
            # Juny/July -> Juni/Juli
            rewrite_word += 'i'
        elif len(curr_part) >= 3 and curr_part[-3:] in {'hoo', 'koo', 'doo', 'moo', 'noo', 'foo'}:
            # hooy, kooy, dooyen, mooy, nooyt, fooy -> hooi, kooi, dooien, mooi, nooit, fooi
            rewrite_word += 'i'
        elif len(curr_part) >= 4 and curr_part[-4:] in {'troo'}:
            # trooy -> trooi (octrooy -> octrooi)
            rewrite_word += 'i'
        elif pi == 0 and curr_part == '' and len(parts[pi + 1]) > 0 and parts[pi + 1].startswith('e'):
            # ye -> ie (yemand -> iemand)
            rewrite_word += 'i'
        elif pi == 0 and curr_part == '' and len(parts[pi+1]) > 0 and parts[pi+1].startswith('r'):
            # yr -> ier (Yrland -> Ierland, Yrssche -> Ierssche)
            rewrite_word += 'ie'
        elif curr_part.endswith('o') or curr_part.endswith('on'):
            rewrite_word += 'y'
        elif len(parts) > pi + 1 and len(parts[pi + 1]) > 0:
            next_part = parts[pi + 1]
            # print('rewrite_word:', rewrite_word, 'next_part:', next_part)
            if curr_part.endswith('a') and next_part.startswith('r'):
                # ayr -> air
                rewrite_word += 'i'
            elif next_part.startswith('ork'):
                # york -> york (york, new york, newyork)
                rewrite_word += 'y'
            elif len(curr_part) >= 2 and curr_part[-2:] in {'pl', 'Pl'} and next_part.startswith('m'):
                # Plym -> Plym (Plymouth
                rewrite_word += 'y'
            elif curr_part.endswith('g') and next_part.startswith('p'):
                # gyp -> gyp (Egypten)
                rewrite_word += 'y'
            elif curr_part.endswith('e'):
                if next_part.startswith('er'):
                    # eyer -> eier
                    rewrite_word += 'y'
                else:
                    # ey -> ei
                    # should never be reached as replace_ey already changes
                    rewrite_word += 'i'
            elif curr_part.endswith('r') and next_part.startswith('e'):
                # rye -> rie (artillerye -> artillerie)
                rewrite_word += 'i'
            elif curr_part.endswith('aa'):
                rewrite_word += 'i'
            elif curr_part.endswith('a'):
                rewrite_word += 'y'
            else:
                rewrite_word += 'ij'
        elif len(parts[pi+1]) == 0 and len(curr_part) >= 3 and curr_part[-3:] in {'lar', 'nar', 'tar', 'ist'}:
            rewrite_word += 'ie'
        elif len(parts[pi+1]) == 0 and len(curr_part) >= 3 and curr_part[-3:] in {'uar', 'ust'}:
            rewrite_word += 'y'
        elif len(parts[pi+1]) == 0 and len(curr_part) >= 2 and curr_part[-3:] in {'ar'}:
            rewrite_word += 'y'
        elif curr_part.endswith('er'):
            rewrite_word += 'ij'
        elif curr_part.endswith('nn') or curr_part.endswith('rr') or curr_part.endswith('ic'):
            rewrite_word += 'y'
        elif curr_part.endswith('aa'):
            rewrite_word += 'i'
        elif curr_part.endswith('a'):
            rewrite_word += 'y'
        elif curr_part.endswith('b'):
            rewrite_word += 'ij'
        elif rewrite_word in {'h', 's', 'z', 'H', 'S', 'Z'}:
            # hy, sy, zy, Hy, Sy, Zy -> hij, sij, zij, Hij, Sij, Zij
            rewrite_word += 'ij'
        else:
            rewrite_word += 'y'
    rewrite_word += parts[-1]
    if capital_y is True:
        if rewrite_word.startswith('ij'):
            rewrite_word = 'IJ' + rewrite_word[2:]
        elif rewrite_word.startswith('i'):
            rewrite_word = 'I' + rewrite_word[1:]
        elif rewrite_word.startswith('y'):
            rewrite_word = 'Y' + rewrite_word[1:]
        else:
            raise ValueError(f'original word started with Y but rewrite word {rewrite_word} '
                             f'starts with unexpected character')
    return rewrite_word


def replace_t(word: str) -> str:
    """Rewrite a handful of historical word-final 't' spellings.

    If the word is exactly 'wordt' or 'vindt', it is returned unchanged
    (these are correct modern spellings, not historical variants). The
    literal word 'duisent' is rewritten directly to 'duizend'.

    Otherwise, the following suffix rewrites are applied in order (each
    only if the word ends with that suffix), and more than one may apply
    in sequence since later checks run against the already-modified
    word:

    - '-dt' -> '-d'
    - '-heit' -> '-heid'
    - '-lant' -> '-land'
    - '-landt' -> '-land'

    Args:
        word: The word to rewrite.

    Returns:
        The word with the matching word-final spelling rewritten, or
        unchanged if no rule applies or it is a hardcoded exception.
    """
    exceptions = {'wordt', 'vindt'}
    if word in exceptions:
        return word
    if word == 'duisent':
        return 'duizend'
    if word.endswith('dt'):
        word = word[:-2] + 'd'
    if word.endswith('heit'):
        word = word[:-1] + 'd'
    if word.endswith('lant'):
        word = word[:-1] + 'd'
    if word.endswith('landt'):
        word = word[:-2] + 'd'
    return word


def normalise_spelling(word: str) -> str:
    """Apply all historical-spelling rewrite rules to a single word.

    Runs the per-variant replace_* functions in sequence, each only if
    its triggering n-gram is present (case-insensitively, except for
    'ck'/'gh' which are checked case-sensitively): replace_ck, replace_ae,
    replace_gh, replace_uy, replace_ey, replace_y, and finally replace_t
    (only if the word ends with 't', case-insensitively). Because each
    step's output feeds into the next step's input check, later rules see
    the results of earlier rewrites.

    Args:
        word: The word to rewrite.

    Returns:
        The word with all applicable historical-spelling rules applied.
    """
    replace_word = word
    if 'ck' in replace_word:
        replace_word = replace_ck(replace_word)
    if 'ae' in replace_word.lower():
        replace_word = replace_ae(replace_word)
    if 'gh' in replace_word:
        replace_word = replace_gh(replace_word)
    if 'uy' in replace_word.lower():
        replace_word = replace_uy(replace_word)
    if 'ey' in replace_word.lower():
        replace_word = replace_ey(replace_word)
    if 'y' in replace_word.lower():
        replace_word = replace_y(replace_word)
    if replace_word.lower().endswith('t'):
        replace_word = replace_t(replace_word)
    return replace_word


def normalise_word(orig_word: str, rewrite_dict: Dict[str, any] = None, to_ascii: bool = False) -> str:
    """Normalise a word's historical spelling, optionally via a lookup dict.

    If `to_ascii` is True, diacritics are stripped first via
    unicode_to_ascii. If `rewrite_dict` is given and contains an entry for
    the (possibly ASCII-folded) word in lower case, the word is replaced
    by the dictionary entry's 'most_similar_term' value and that
    replacement is passed through normalise_spelling. Otherwise, the word
    is title-cased (if it started with an uppercase letter) and then
    passed through normalise_spelling directly.

    In both cases, the final casing of the original word is reapplied to
    the result: fully uppercase originals yield fully uppercase output,
    and originals starting with an uppercase letter yield title-cased
    output.

    Args:
        orig_word: The word to normalise.
        rewrite_dict: Optional mapping from lower-cased word forms to a
            dict containing at least a 'most_similar_term' key, used as a
            lookup-based override before falling back to rule-based
            rewriting.
        to_ascii: If True, strip diacritics from the word before applying
            the dictionary lookup or rule-based rewriting.

    Returns:
        The normalised word, recapitalised to match the casing pattern of
        `orig_word`.
    """
    copy_word = orig_word
    if to_ascii:
        copy_word = unicode_to_ascii(copy_word)
    if rewrite_dict is not None and copy_word.lower() in rewrite_dict:
        norm_word = normalise_spelling(rewrite_dict[copy_word.lower()]['most_similar_term'])
        if orig_word.isupper():
            norm_word = norm_word.upper()
        elif orig_word[0].isupper():
            norm_word = norm_word.title()
    else:
        if orig_word[0].isupper():
            copy_word = copy_word.title()
        norm_word = normalise_spelling(copy_word)
    if orig_word.isupper():
        return norm_word.upper()
    elif orig_word[0].isupper():
        return norm_word.title()
    else:
        return norm_word
