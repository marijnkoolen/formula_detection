import re
from collections import Counter
from collections import defaultdict
from itertools import permutations
from typing import Dict, List

from fuzzy_search.similarity import SkipgramSimilarity
from fuzzy_search.tokenization.token import Tokenizer
from fuzzy_search.tokenization.string import score_levenshtein_similarity_ratio
from Levenshtein import editops as get_editops

from .variation.edit import is_punct, classify_diff
from .variation.edit import get_alignment_change


def is_aligned_whitespace_chunk(chunk: Dict[str, any]) -> bool:
    return chunk['type'] == 'aligned' and ' ' in chunk['source']


def is_aligned_with_next_token_chunk(group: List[Dict[str, any]],
                                     debug: int = 0) -> bool:
    source_string = ''.join([group_chunk['source'] for group_chunk in group])
    dest_string = ''.join([group_chunk['dest'] for group_chunk in group])
    # if chunk['type'] == 'aligned':
    #     return False
    # if chunk['source'] != chunk['dest']:
    #     return False
    if len(group) == 0:
        return False
    if not is_whitespace_change_chunk(group[-1]):
        return False
    if len(source_string) > 0 and len(dest_string) > 0:
        return False
    if source_string.endswith(' '):
        if len(source_string.strip()) >= 3:
            return True
        elif source_string.strip() in {'de', 'in', 'op', 'om', 'te'}:
            return True
    if dest_string.endswith(' '):
        if len(dest_string.strip()) >= 3:
            return True
        elif dest_string.strip() in {'de', 'in', 'op', 'om', 'te'}:
            return True
    return False


def is_whitespace_change_chunk(chunk: Dict[str, any]) -> bool:
    if chunk['source'] == '' and chunk['dest'].isspace():
        return True
    if chunk['source'].isspace() and chunk['dest'] == '':
        return True
    else:
        return False


def reduce_changes(aligned_chunks: List[Dict[str, any]], debug: int = 0):
    reduced_chunks = []
    prev_type = 'aligned'
    for ci, curr_chunk in enumerate(aligned_chunks):
        if debug > 0:
            print('reduce_changes - curr_chunk:', curr_chunk)
        if curr_chunk['type'] == 'aligned':
            reduced_chunks.append(curr_chunk)
            if debug > 0:
                print('reduce_changes - adding aligned chunk:')
            prev_type = curr_chunk['type']
            continue
        elif prev_type == 'aligned':
            if debug > 0:
                print('reduce_changes - adding post-aligned chunk:')
            reduced_chunks.append(curr_chunk)
            reduced_chunks[-1]['type'] = 'replaced'
        elif is_whitespace_change_chunk(curr_chunk):
            reduced_chunks.append(curr_chunk)
            reduced_chunks[-1]['type'] = 'replaced'
        else:
            reduced_chunks[-1]['source'] += curr_chunk['source']
            reduced_chunks[-1]['dest'] += curr_chunk['dest']
            reduced_chunks[-1]['type'] = 'replaced'
            if debug > 0:
                print('reduce_changes - merging current chunk with previous replacement:', reduced_chunks[-1])
        prev_type = curr_chunk['type']
    return reduced_chunks


def tokenize_aligned_chunks(aligned_chunks: List[Dict[str, any]], debug: int = 0):
    tokenized_chunks = []
    for chunk in aligned_chunks:
        if debug > 0:
            print('tokenize_aligned_chunks - chunk:', chunk)
        if chunk['type'] != 'aligned':
            if ' ' in chunk['source']:
                pass
                # assert ' ' not in chunk['dest'], f"whitespace in both source '{chunk['source']}'" \
                #                                  f" and dest '{chunk['dest']}'"
        if chunk['type'] == 'aligned' and ' ' in chunk['source']:
            if debug > 0:
                print('tokenize_aligned_chunks - aligned with whitespace:')
            tokens = re.split(r'(\S+)', chunk['source'])
            for token in tokens:
                if token == '':
                    continue
                token_chunk = {'source': token, 'dest':  token, 'type': 'aligned', 'char_diff': 0}
                tokenized_chunks.append(token_chunk)
        elif chunk['type'] == 'replace' and (' ' in chunk['source'] or ' ' in chunk['dest']):
            if debug > 0:
                print('tokenize_aligned_chunks - replaced with whitespace:')
            if ' ' in chunk['source']:
                tokens = re.split(r'(\S+)', chunk['source'])
                split_side = 'source'
            else:
                tokens = re.split(r'(\S+)', chunk['dest'])
                split_side = 'dest'
            # print('tokens:', tokens)
            other_used = False
            for token in tokens:
                if token == '':
                    continue
                if ' ' in token or other_used is True:
                    if split_side == 'source':
                        source_token, dest_token = token, ''
                    else:
                        source_token, dest_token = '', token
                    token_chunk = {'source': source_token, 'dest':  dest_token,
                                   'type': 'replaced', 'char_diff': len(source_token) - len(dest_token)}
                else:
                    other_used = True
                    if split_side == 'source':
                        source_token, dest_token = token, chunk['dest']
                    else:
                        source_token, dest_token = chunk['source'], token
                    token_chunk = {'source': source_token, 'dest': dest_token,
                                   'type': 'replaced', 'char_diff': len(source_token) - len(dest_token)}
                tokenized_chunks.append(token_chunk)
        else:
            if debug > 0:
                print(f"tokenize_aligned_chunks - {chunk['type']} without whitespace:")
            chunk['char_diff'] = len(chunk['source']) - len(chunk['dest'])
            tokenized_chunks.append(chunk)
    return tokenized_chunks


def combine_tokenized_chunks(chunks: List[Dict[str, any]], debug: int = 0):
    combined_chunks = []
    chunk_groups = []
    group = []
    for chunk in chunks:
        if debug > 0:
            print('combine_tokenized_chunks - chunk', chunk)
        is_singleton_group = False
        new_group = False
        if is_aligned_whitespace_chunk(chunk):
            is_singleton_group = True
        elif is_aligned_with_next_token_chunk(group, debug=debug):
            new_group = True
        if new_group or is_singleton_group:
            if len(group) > 0:
                chunk_groups.append(group)
            if is_singleton_group:
                if debug > 0:
                    print('\tsingleton group')
                chunk_groups.append([chunk])
                group = []
            else:
                if debug > 0:
                    print('\tnew group')
                group = [chunk]
        else:
            if debug > 0:
                print('\tadd to group')
            group.append(chunk)
    if len(group) > 0:
        chunk_groups.append(group)
    for group in chunk_groups:
        source = ''.join([chunk['source'] for chunk in group])
        dest = ''.join([chunk['dest'] for chunk in group])
        source_tokens = source.strip().split()
        dest_tokens = dest.strip().split()
        char_diff = len(source) - len(dest)
        replace_chunks = [chunk for chunk in group if chunk['type'] == 'replaced']
        align_chunks = [chunk for chunk in group if chunk['type'] == 'aligned']
        max_align = max([chunk['char_diff'] for chunk in align_chunks]) if len(align_chunks) > 0 else 0
        max_replace = max([chunk['char_diff'] for chunk in replace_chunks]) if len(replace_chunks) > 0 else 0
        min_align = min([chunk['char_diff'] for chunk in align_chunks]) if len(align_chunks) > 0 else 0
        min_replace = min([chunk['char_diff'] for chunk in replace_chunks]) if len(replace_chunks) > 0 else 0
        source_punct_tokens = [token for token in source_tokens if is_punct(token) is True]
        dest_punct_tokens = [token for token in dest_tokens if is_punct(token) is True]
        source_word_tokens = [token for token in source_tokens if is_punct(token) is False]
        dest_word_tokens = [token for token in dest_tokens if is_punct(token) is False]
        if debug > 1:
            print(replace_chunks)
            print(align_chunks)
            print(source_word_tokens)
            print(dest_word_tokens)
            print(max_align, max_replace)
            print(min_align, min_replace)
        if len(group) == 1:
            change_type = ''
        elif ' '.join(source_word_tokens) == ' '.join(dest_word_tokens):
            change_type = ''
        elif len(source_word_tokens) == 0 and len(dest_word_tokens) > 0:
            change_type = 'remove_tokens'
        elif len(source_word_tokens) > 0 and len(dest_word_tokens) == 0:
            change_type = 'add_tokens'
        elif len(source_word_tokens) > len(dest_word_tokens) and max_replace <= 2:
            change_type = 'split_tokens'
        elif len(source_word_tokens) > len(dest_word_tokens) and max_replace > 2:
            change_type = 'add_tokens'
        elif len(source_word_tokens) < len(dest_word_tokens) and min_replace > -2:
            change_type = 'merge_tokens'
        elif len(source_word_tokens) < len(dest_word_tokens) and min_replace <= -2:
            change_type = 'remove_tokens'
        else:
            change_type = 'replace_tokens'
        combined_chunk = {
            'source': source,
            'dest': dest,
            'align_type': 'aligned' if len(group) == 1 else 'replaced',
            'change_type': change_type,
            'chunks': group,
        }
        combined_chunk = solve_whitespace_punctuation(combined_chunk)
        combined_chunks.append(combined_chunk)
    return combined_chunks


def classify_changes(chunk_group: List[Dict[str, str]]):
    return None


def align_phrase_chunks(source: str, dest: str):
    aligned_chunks = get_alignments_changes(source, dest)
    reduced_chunks = reduce_changes(aligned_chunks)
    try:
        tokenized_chunks = tokenize_aligned_chunks(reduced_chunks)
    except AssertionError:
        print(f"source: {source}\ndest: {dest}\n")
        raise
    combined_chunks = combine_tokenized_chunks(tokenized_chunks)
    # for cchunk in combined_chunks:
    #     print(cchunk)
    return combined_chunks


def solve_whitespace_punctuation(combined_group: Dict[str, any]) -> Dict[str, any]:
    new_group = {
        'source': combined_group['source'],
        'dest': combined_group['dest'],
        'align_type': combined_group['align_type'],
        'change_type': combined_group['change_type'],
        'chunks': combined_group['chunks']
    }
    punct_start_pattern = re.compile(r'^[.,:;\'"-+=()&*\[\]{}]+ +')
    punct_end_pattern = re.compile(r' +[.,:;\'"-+=()&*\[\]{}]+$')
    if new_group['source'] == '' or new_group['dest'] == '':
        return new_group
    if re.match(punct_start_pattern, new_group['source']) and new_group['dest'][0].isalnum():
        new_group['dest'] = ' ' + new_group['dest']
        # assert is_whitespace_change_chunk(new_group['chunks'][1])
        # new_group['chunks'][1]['dest'] = ' '
        # new_group['chunks'][1]['type'] = 'aligned'
        if new_group['change_type'] == 'replace_tokens':
            new_group['change_type'] += '_add_punctuation'
    elif re.match(punct_start_pattern, new_group['dest']) and new_group['source'][0].isalnum():
        new_group['source'] = ' ' + new_group['source']
        # if is_whitespace_change_chunk(new_group['chunks'][1]) is False:
        #     print(new_group['chunks'])
        #     raise ValueError('second chunks is not a whitespace change')
        # new_group['chunks'][1]['source'] = ' '
        # new_group['chunks'][1]['type'] = 'aligned'
        new_group['change_type'] += '_remove_punctuation'
    if re.search(punct_end_pattern, new_group['source']) and new_group['dest'][-1].isalnum():
        new_group['dest'] = new_group['dest'] + ' '
        new_group['change_type'] += '_add_punctuation'
    elif re.search(punct_end_pattern, new_group['dest']) and new_group['source'][-1].isalnum():
        new_group['source'] = new_group['source'] + ' '
        new_group['change_type'] += '_remove_punctuation'
    if new_group['change_type'].startswith('_'):
        new_group['change_type'] = new_group['change_type'][1:]
    return new_group


def detect_word_swaps(phrases: Counter, tokenizer: Tokenizer, ngram_size: int = 2, debug: int = 0):
    ngram_freq = Counter()
    candidate_swap_freq = Counter()
    in_doc = defaultdict(set)
    for phrase, freq in phrases.most_common():
        doc = tokenizer.tokenize(phrase)
        for ti in range(len(doc.tokens) - (ngram_size - 1)):
            ngram_tokens = [token.n for token in doc.tokens[ti:ti+ngram_size]]
            if is_punct(ngram_tokens[0]) or is_punct(ngram_tokens[-1]):
                # ignore word swap when the first or last word is punctuation
                continue
            ngram = tuple(ngram_tokens)
            in_doc[ngram].add(doc)
            ngram_freq.update([ngram])
            for perm_tokens in permutations(ngram_tokens, len(ngram_tokens)):
                perm_ngram = tuple(perm_tokens)
                if perm_ngram == ngram:
                    continue
                if perm_ngram in ngram_freq:
                    if debug > 1:
                        ngram_string = ' '.join(ngram)
                        perm_ngram_string = ' '.join(perm_ngram)
                        print(f'detect_word_swaps - ngram "{ngram_string}"\tperm_ngram "{perm_ngram_string}"')
                    candidate_swap = (ngram, perm_ngram)
                    candidate_swap_freq.update([candidate_swap])
    swaps = Counter()
    for candidate_swap in candidate_swap_freq:
        ngram1, ngram2 = candidate_swap
        if debug > 0:
            print('comparing phrases of candidates', ngram1, ngram2)
        for doc1 in in_doc[ngram1]:
            ngram_string1 = ' '.join(ngram1)
            assert ngram_string1 in doc1.text, f"ngram_string '{ngram_string1}' not in doc text '{doc1.text}'"
            try:
                head1, tail1 = doc1.text.split(ngram_string1, 1)
            except ValueError:
                print(f"ngram_string: #{ngram_string1}#\tdoc1.text: {doc1.text}")
                raise
            if head1.endswith(' ') and tail1.startswith(' '):
                rest_doc1 = f"{head1}{tail1[1:]}"
            else:
                rest_doc1 = doc1.text.replace(ngram_string1, '')
            if debug > 0:
                print('\tdoc.text1:', doc1.text)
                print('\tngram_string1:', ngram_string1)
            for doc2 in in_doc[ngram2]:
                ngram_string2 = ' '.join(ngram2)
                assert ngram_string2 in doc2.text, f"ngram_string '{ngram_string2}' not in doc text '{doc2.text}'"
                if ngram_string1 in doc2.text:
                    # word repetition like ever and ever (with ngrams ever_and, and_ever)
                    # in resolutions e.g.: woorde te woorde
                    if debug > 0:
                        print(f"'skipping word repetition: {ngram_string1}' in '{doc2.text}'")
                    continue
                else:
                    if debug > 0:
                        print(f"'{ngram_string1}' not in '{doc2.text}'")
                    pass
                head2, tail2 = doc2.text.split(ngram_string2)
                if head2.endswith(' ') and tail2.startswith(' '):
                    rest_doc2 = f"{head2}{tail2[1:]}"
                else:
                    rest_doc2 = doc2.text.replace(ngram_string2, '')
                if abs(len(rest_doc1) - len(rest_doc2)) > 4:
                    continue
                sim = score_levenshtein_similarity_ratio(rest_doc1, rest_doc2)
                if sim < 0.7:
                    continue
                if debug > 0:
                    print('\tdoc.text2:', doc2.text)
                    print('\tngram_string2:', ngram_string2)
                if sim < 0.6:
                    continue
                if debug > 0:
                    print('\trest_doc1:', rest_doc1)
                    print('\trest_doc2:', rest_doc2)
                    print('\tsim:', sim)
                swaps.update([(ngram_string1, ngram_string2)])
    return swaps


def get_alignments_changes(source: str, dest: str, debug: int = 0):
    edits = get_editops(source, dest)
    aligned_chunks = []
    source_shift = 0
    dest_shift = 0
    source_dummy, dest_dummy = source, dest
    for edit in edits:
        op, source_i, dest_i = edit
        if debug > 0:
            print('get_alignments_changes - source_dummy:', source_dummy)
            print('get_alignments_changes - dest_dummy:', dest_dummy)
            print('get_alignments_changes - edit:', edit)
        source_i -= source_shift
        dest_i -= dest_shift
        if debug > 0:
            print('get_alignments_changes - source_i, dest_i:', source_i, dest_i)
            print('get_alignments_changes - source_shift, dest_shift:', source_shift, dest_shift)
        chunks, source_tail, dest_tail = get_alignment_change(source_dummy, dest_dummy, op, source_i, dest_i)
        if debug > 0:
            print('get_alignments_changes - chunks:', chunks)
            print('get_alignments_changes - source_tail:', source_tail)
            print('get_alignments_changes - dest_tail:', dest_tail)
        aligned_chunks.extend(chunks)
        source_shift += len(source_dummy) - len(source_tail)
        dest_shift += len(dest_dummy) - len(dest_tail)
        if debug > 0:
            print('get_alignments_changes - source_shift:', source_shift)
            print('get_alignments_changes - dest_shift:', dest_shift)
        source_dummy = source_tail
        dest_dummy = dest_tail
        if debug > 0:
            print('get_alignments_changes - source_dummy:', source_dummy)
            print('get_alignments_changes - dest_dummy:', dest_dummy)
            print('\n')
    if len(source_dummy) > 0 or len(dest_dummy) > 0:
        aligned_chunks.append({'source': source_dummy, 'dest': dest_dummy, 'type': 'aligned'})
    return aligned_chunks


def get_aligned_token_freq(sub_phrase_freq: Counter, word_swap_freq: Counter,
                           lev_score_threshold: float = 0.6, debug: int = 0):
    skip_sim = SkipgramSimilarity(ngram_length=2, skip_length=2,
                                  terms=list(sub_phrase_freq.keys()))

    aligned_tokens_freq = Counter()
    phrase_pair_checked = set()
    token_freq = Counter()

    for sub_phrase in sorted(sub_phrase_freq, key=lambda x: sub_phrase_freq[x], reverse=True):
        if sub_phrase in phrase_pair_checked:
            continue
        for sim_phrase, sim_score in skip_sim.rank_similar(sub_phrase, score_cutoff=0.75):
            if sub_phrase == sim_phrase:
                continue
            if (sim_phrase, sub_phrase) in phrase_pair_checked:
                continue
            for word_swap1, word_swap2 in word_swap_freq:
                if word_swap1 in sub_phrase and word_swap2 in sim_phrase:
                    if debug > 0:
                        print(f'SWAPPING "{word_swap1}" for "{word_swap2}" in sim_phrase "{sim_phrase}"')
                    sim_phrase = sim_phrase.replace(word_swap2, word_swap1)
            phrase_pair_checked.add((sub_phrase, sim_phrase))
            lev_score = score_levenshtein_similarity_ratio(sub_phrase, sim_phrase)
            if lev_score < lev_score_threshold:
                continue
            if debug > 0:
                diff_class = classify_diff(sub_phrase, sim_phrase)
                print(f"{sub_phrase: <40}{sim_phrase: <40}{sim_score: >6.2f}\t{lev_score: >6.2f}\t{diff_class}")
            aligned_phrase_chunks = align_phrase_chunks(sub_phrase, sim_phrase)
            for apc in aligned_phrase_chunks:
                if apc['align_type'] == 'aligned':
                    continue
                if debug > 0:
                    for chunk in apc['chunks']:
                        # if chunk['type'] != 'replaced':
                        #    continue
                        print('\t', chunk)
                    print([chunk for chunk in apc['chunks'] if chunk['align_type'] == 'replaced'])
                source = apc['source']
                dest = apc['dest']
                lev_score = score_levenshtein_similarity_ratio(source, dest)
                if lev_score < 0.5:
                    continue
                if debug > 0:
                    print(f"\t{lev_score: >6.2f}\t{source: <20}\t\t{dest: <20}")
                # aligned_tokens_freq[apc] += sub_phrase_freq[sub_phrase]
                aligned_tokens_freq[(source, dest, apc['align_type'], apc['change_type'])] += sub_phrase_freq[sub_phrase]
                token_freq[source] += sub_phrase_freq[sub_phrase]

            if debug > 0:
                print()
    return aligned_tokens_freq, token_freq


class MapVariants:

    def __init__(self, aligned_tokens_freq: Counter, token_freq: Counter,
                 min_freq: int = 0, debug: int = 0):
        self.atf = aligned_tokens_freq
        self.tf = token_freq
        self.has_variant = defaultdict(set)
        self.is_variant_of = {}
        self.tokens = set()
        self.min_freq = min_freq
        self.debug = debug
        self._get_variant_mapping()

    def _get_best_source(self, source: str):
        if source in self.is_variant_of:
            source = self.is_variant_of[source]
            if self.debug > 0:
                print('\tswapping to preferred source:', source)
        elif source.strip() in self.is_variant_of:
            # source has whitespace prefix or suffix -> check if source has a preferred
            # if so, use that and prefix/suffix it in the same way
            new_source = self.is_variant_of[source.strip()]
            if source[0] == ' ':
                new_source = ' ' + new_source
            elif source[-1] == ' ':
                new_source = new_source + ' '
            if self.debug > 0:
                print(f'\tswapping prefixed source "{source}" for more common prefixed source "{new_source}"')
            self.is_variant_of[source] = new_source
            self.has_variant[new_source].add(source)
            self.tokens.add(source)
            self.tokens.add(new_source)
            source = new_source
        rewritten_source = rewrite_context_phrase(source, self.is_variant_of)
        if rewritten_source == source:
            pass
        elif rewritten_source in self.is_variant_of:
            if self.debug > 0:
                print(f'source "{source}" is rewritten to "{rewritten_source}" which '
                      f'is a variant of "{self.is_variant_of[rewritten_source]}"')
            source = self.is_variant_of[rewritten_source]
        else:
            if self.debug > 0:
                print(f'source "{source}" is rewritten to "{rewritten_source}"')
            source = rewritten_source
        return source

    def _sort_aligned_tokens(self):
        done = set()
        sorted_aligned_tokens_freq = defaultdict(Counter)
        for aligned_tokens, freq in self.atf.most_common():
            if freq < self.min_freq:
                continue
            source, dest, align_type, change_type = aligned_tokens
            if self.tf[dest] > self.tf[source]:
                source, dest = dest, source
            if (is_punct(source[0]) and source[1] == ' ') or \
                    (is_punct(source[-1]) and source[-2] == ' '):
                if self.debug > 0:
                    print('\tswapping source and dest to avoid punctuation issues')
                source, dest = dest, source
            if source in self.is_variant_of and self.is_variant_of[source] == dest:
                if self.debug > 0:
                    print('\tskipping because reverse is already registered\n')
                continue
            if dest in self.is_variant_of and self.is_variant_of[dest] == source:
                if self.debug > 0:
                    print('\tskipping because mapping is already registered\n')
                continue
            if (source, dest) in done:
                continue
            done.add((source, dest))
            num_tokens = len(source.split(' '))
            sorted_aligned_tokens_freq[num_tokens][(source, dest, change_type)] = self.atf[aligned_tokens]
        return sorted_aligned_tokens_freq

    def iterate_tokens(self):
        sorted_aligned_tokens_freq = self._sort_aligned_tokens()
        for num_tokens in sorted(sorted_aligned_tokens_freq):
            for aligned_tokens, freq in sorted_aligned_tokens_freq[num_tokens].most_common():
                yield aligned_tokens, freq
        return None

    def _replace_preferred_variant(self, old_pref: str, new_pref: str):
        if old_pref == new_pref:
            return None
        if self.debug > 0:
            print(f'\tremoving dest "{old_pref}" from preferred variants to be replaced by "{new_pref}"')
        if old_pref in self.has_variant:
            for old_pref_variant in self.has_variant[old_pref]:
                if self.debug > 0:
                    print(f'\t\tmoving dest variant: "{old_pref_variant}" to be variant of source "{new_pref}"')
                self.has_variant[new_pref].add(old_pref_variant)
                self.is_variant_of[old_pref_variant] = new_pref
            if self.debug > 0:
                print('\tremoving old_pref from preferred variants')
            del self.has_variant[old_pref]
        self.has_variant[new_pref].add(old_pref)
        self.is_variant_of[old_pref] = new_pref

    def _map_variant(self, source: str, dest: str):
        source = self._get_best_source(source)
        if self.debug > 0:
            print(f'_map_variant - best_source: "{source}"')
        rewritten_dest = rewrite_context_phrase(dest, self.is_variant_of)
        if self.debug > 0:
            print(f'_map_variant - rewritten_dest: "{rewritten_dest}"')
        if rewritten_dest != dest:
            if self.debug > 0:
                print(f'_map_variant - before rewriting, current dest is "{dest}"')
            best_source = self._get_best_source(rewritten_dest)
            if source == best_source:
                if self.debug > 0:
                    print(f'_map_variant - during rewriting (source == best_source), current dest is "{dest}"')
                pass
            elif best_source in self.tf and source in self.tf:
                if self.debug > 0:
                    print(f'_map_variant - during rewriting (source and best_source in tf), current dest is "{dest}"')
                if self.tf[source] > self.tf[best_source]:
                    self._map_variant(source, best_source)
                else:
                    self._map_variant(best_source, source)
                if self.debug > 0:
                    print(f'_map_variant - during rewriting (after mapping source and best_source), current dest is "{dest}"')
            elif source in best_source:
                if self.debug > 0:
                    print(f'_map_variant - during rewriting (source in best_source), current dest is "{dest}"')
                self._replace_preferred_variant(best_source, source)
                self.tokens.add(rewritten_dest)
            else:
                if self.debug > 0:
                    print(f'_map_variant - during rewriting (no evidence for best_source), current dest is "{dest}"')
                # the rewrite has no evidence of being better in context
                if self.debug > 0:
                    print(f'no evidence to accept "{rewritten_dest}" as a preferred variant of "{dest}"')
                pass
                # if source in self.has_variant:
                #     self._replace_preferred_variant(source, best_source)
                # self.is_variant_of[source] = best_source
                # source = best_source
            if self.debug > 0:
                print(f'_map_variant - after rewriting, current dest is "{dest}"')
        elif dest in self.has_variant:
            self._replace_preferred_variant(dest, source)
        elif dest in self.is_variant_of:
            if self.tf[self.is_variant_of[dest]] > self.tf[source]:
                new_source = self.is_variant_of[dest]
                self._replace_preferred_variant(source, new_source)
                source = new_source
            else:
                old_source = self.is_variant_of[dest]
                self._replace_preferred_variant(old_source, source)
        else:
            pass
        if self.debug > 0:
            print(f'\tadding dest "{dest}" as variant of "{source}"')
        self.has_variant[source].add(dest)
        self.is_variant_of[dest] = source
        self.tokens.add(source)
        self.tokens.add(dest)
        self._check_missing()

    def _get_variant_mapping(self) -> Dict[str, str]:
        count = 0
        for aligned_tokens, freq in self.iterate_tokens():
            source, dest, change_type = aligned_tokens
            count += 1
            # token_freq[source] += aligned_tokens_freq[aligned_tokens]
            if self.debug > 0:
                print(f'{count}\t"{source}" <- "{dest}", {self.tf[source]}, {self.tf[dest]}')
            self._map_variant(source, dest)
        return self.is_variant_of

    def _check_missing(self):
        dict_tokens = set([token for token in list(self.has_variant.keys()) + list(self.is_variant_of.keys())])
        for token in self.tokens:
            if token not in self.has_variant and token not in self.is_variant_of:
                if self.debug > 0:
                    print(f'missing token in variant map: "{token}"')
                pass
            assert token in self.has_variant or token in self.is_variant_of
        for token in dict_tokens:
            if token not in self.tokens:
                if self.debug > 0:
                    print(f'missing token in token list: "{token}"')
            assert token in self.tokens
        if len(self.tokens) != len(dict_tokens):
            print('self.tokens:', sorted(self.tokens))
            print('dict_tokens:', sorted(dict_tokens))
            raise ValueError
        for variant in self.is_variant_of:
            source = self.is_variant_of[variant]
            assert variant in self.has_variant[source], f"variant '{variant}' not in has_variant of source '{source}'"    # print(tokens)
        if self.debug > 0:
            print()


def rewrite_context_phrases(context_phrases: List[str], is_variant_of: Dict[str, str],
                            debug: int = 0) -> Dict[str, str]:
    rewritten_context_phrases = {}
    for phrase in context_phrases:
        rewritten_context_phrases[phrase] = rewrite_context_phrase(phrase, is_variant_of, debug=debug)
    return rewritten_context_phrases


def rewrite_context_phrase(phrase: str, is_variant_of: Dict[str, str], debug: int = 0) -> str:
    rewritten_phrase = phrase
    for variant in is_variant_of:
        if re.search(fr"\b{variant}\b", rewritten_phrase):
            if debug > 0:
                print(f"\trewriting phrase '{phrase}' with '{variant}' to '{is_variant_of[variant]}'")
            rewritten_phrase = re.sub(fr"\b{variant}\b", is_variant_of[variant], rewritten_phrase)
    if debug > 0:
        if rewritten_phrase != phrase:
            print(f"\noriginal phrase: {phrase}\n")
            print(f"\nrewritten phrase: {rewritten_phrase}\n")
    return rewritten_phrase


def merge_phrase_context_freqs(context_freq):
    merged_context_freq = {
        'pre': Counter(),
        'post': Counter(),
        'phrase': Counter()
    }
    for direction in {'pre', 'post'}:
        for phrase in context_freq[direction]:
            for context_phrase in context_freq[direction][phrase]:
                freq = context_freq[direction][phrase][context_phrase]
                merged_context_freq[direction][context_phrase] += freq
    for phrase in context_freq['phrase']:
        merged_context_freq['phrase'][phrase] += context_freq['phrase'][phrase]
    return merged_context_freq


def merge_period_context_freqs(context_freq, periods):
    merged_context_freq = init_context_freq()

    for period in periods:
        for direction in ['pre', 'post']:
            for phrase in context_freq[period][direction]:
                for phrase_context in context_freq[period][direction][phrase]:
                    merged_context_freq[direction][phrase][phrase_context] += context_freq[period][direction][phrase][
                        phrase_context]
    return merged_context_freq


def get_sub_phrase_freq(context_freq, tokenizer: Tokenizer, direction: str, debug: int = 0):
    sub_phrase_freq = Counter()
    prefix_phrase_freq = Counter()

    for context_phrase in context_freq[direction]:
        if context_freq[direction][context_phrase] <= 1:
            continue
        prefix_phrase_freq[context_phrase] += context_freq[direction][context_phrase]
        if debug > 0:
            print('\t', context_phrase)
        doc = tokenizer.tokenize(context_phrase)
        tokens = [token.n for token in doc]
        if direction == 'pre':
            tokens = tokens[::-1]
        for i in range(len(tokens)):
            sub_phrase = tokens[:i+1]
            if debug > 0:
                print('\t\t', sub_phrase, context_freq[direction][context_phrase])
            if direction == 'pre':
                sub_phrase = sub_phrase[::-1]
            sub_phrase_freq[' '.join(sub_phrase)] += context_freq[direction][context_phrase]
            if debug > 0:
                print('\t\t', ' '.join(sub_phrase))
    return sub_phrase_freq


def init_context_freq():
    return {
        'pre': defaultdict(Counter),
        'post': defaultdict(Counter),
        'phrase': Counter()
    }


def rewrite_context_variation(context_freq, tokenizer: Tokenizer, min_freq: int = 0):
    merged_context_freq = merge_phrase_context_freqs(context_freq)
    rewritten_context_freq = init_context_freq()
    rewritten_context_freq['word_swap'] = {
        'pre': defaultdict(Counter),
        'post': defaultdict(Counter)
    }
    for direction in ['pre', 'post']:
        print(f"rewrite_context_variation - direction: {direction} - num phrases: {len(merged_context_freq[direction])}")
        sub_phrase_freq = get_sub_phrase_freq(merged_context_freq, tokenizer, direction)
        word_swap_freq = detect_word_swaps(merged_context_freq[direction], tokenizer, debug=0)
        rewritten_context_freq['word_swap'][direction] = word_swap_freq
        aligned_tokens_freq, token_freq = get_aligned_token_freq(sub_phrase_freq, word_swap_freq)
        variant_mapper = MapVariants(aligned_tokens_freq, token_freq, min_freq=min_freq, debug=0)
        is_variant_of = variant_mapper.is_variant_of
        # is_variant_of = get_variant_mapping(aligned_tokens_freq, token_freq)
        for phrase in context_freq[direction]:
            # phrase_context_freq = context_freq[direction][phrase]
            # rpf_freq = rewrite_context_phrases(phrase_context_freq, is_variant_of)
            for context_phrase, freq in context_freq[direction][phrase].most_common():
                rewritten_context = rewrite_context_phrase(context_phrase, is_variant_of)
                rewritten_context_freq[direction][phrase][rewritten_context] += freq
            # rewritten_context_freq[direction][phrase] = rpf_freq
    return rewritten_context_freq



