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


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def replace_ck(word: str) -> str:
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
