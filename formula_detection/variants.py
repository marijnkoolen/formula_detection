"""
variants.py — Spelling-variant and word-order-variant detection for recurring phrases.

Given the contexts (preceding/following phrases) collected around a set of
candidate formulas, this module aligns pairs of similar sub-phrases character
by character (using Levenshtein edit operations), reduces and re-groups the
resulting edit chunks into token-level changes, and classifies those changes
(e.g. token split/merge/add/remove/replace, punctuation addition/removal).

It also detects word-order swaps between near-duplicate n-grams (e.g. "and
ever" vs "ever and") and aligns tokens across many sub-phrase pairs to build
a frequency-weighted map of which spelling/word-order variant should be
treated as canonical. The MapVariants class consumes that aligned-token
frequency information and produces a `is_variant_of` mapping plus its
inverse `has_variant` mapping, used to rewrite raw context phrases to a
canonical form so that recurring formulas are not fragmented by minor
historical spelling variation.
"""
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
    """Check whether an aligned chunk is purely whitespace within an otherwise aligned span.

    Args:
        chunk: An alignment chunk dict with 'type' and 'source' keys, as
            produced by get_alignments_changes.

    Returns:
        True if the chunk's type is 'aligned' and its source text contains
        a space character, False otherwise.
    """
    return chunk['type'] == 'aligned' and ' ' in chunk['source']


def is_aligned_with_next_token_chunk(group: List[Dict[str, any]],
                                     debug: int = 0) -> bool:
    """Decide whether the chunk group should be closed off before adding the next token.

    Used while grouping tokenized alignment chunks: a group should end here
    (and a new group should start with the next chunk) when the group's
    last chunk is a pure whitespace insertion/deletion and one side of the
    concatenated group text is empty while the other ends in a deleted or
    inserted whitespace-trailing token of at least 3 characters, or is one
    of a small set of short Dutch function words ('de', 'in', 'op', 'om',
    'te') that are still considered meaningful word boundaries.

    Args:
        group: The list of alignment chunks accumulated so far for the
            current group.
        debug: Verbosity level for debug printing (0 = silent).

    Returns:
        True if the current group is aligned with (should be split before)
        the next token, False otherwise.
    """
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
    """Check whether a chunk represents an insertion or deletion of pure whitespace.

    Args:
        chunk: An alignment chunk dict with 'source' and 'dest' keys.

    Returns:
        True if one side of the chunk is empty and the other side is
        non-empty whitespace (i.e. a whitespace insertion or deletion),
        False otherwise.
    """
    if chunk['source'] == '' and chunk['dest'].isspace():
        return True
    if chunk['source'].isspace() and chunk['dest'] == '':
        return True
    else:
        return False


def reduce_changes(aligned_chunks: List[Dict[str, any]], debug: int = 0):
    """Merge consecutive non-aligned (changed) chunks into single 'replaced' chunks.

    Walks the raw character-level alignment chunks produced by
    get_alignments_changes and collapses runs of consecutive change
    operations (insert/delete/replace) that directly follow each other or
    follow an aligned chunk into a single 'replaced' chunk, so that
    adjacent edits on the same word/span are treated as one combined
    change rather than many tiny ones. Whitespace-change chunks are kept
    as their own (also relabelled 'replaced') chunk rather than merged
    into the preceding one.

    Args:
        aligned_chunks: List of alignment chunk dicts (each with 'source',
            'dest', 'type') as produced by get_alignments_changes.
        debug: Verbosity level for debug printing (0 = silent).

    Returns:
        A new list of chunk dicts where consecutive change chunks have
        been merged into single 'replaced' chunks, interleaved with the
        original 'aligned' chunks.
    """
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
    """Split reduced alignment chunks that span whitespace into per-token chunks.

    For 'aligned' chunks containing whitespace, splits the chunk into
    separate aligned chunks per whitespace-delimited token (each given
    'char_diff': 0). For 'replace' chunks where either side contains
    whitespace, splits on whitespace and distributes the source/dest text
    across new 'replaced' chunks, attaching the non-whitespace-containing
    side's untouched text to the first token-aligned sub-chunk it
    encounters (tracked via `other_used`). All other chunks are passed
    through unchanged except for an added 'char_diff' field (the
    difference in length between source and dest).

    Args:
        aligned_chunks: List of reduced alignment chunk dicts (as produced
            by reduce_changes).
        debug: Verbosity level for debug printing (0 = silent).

    Returns:
        A new list of chunk dicts, each with a 'char_diff' field, where
        chunks spanning whitespace have been split into per-token chunks.
    """
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
    """Group tokenized alignment chunks into combined chunks and classify each group's change type.

    Iterates the per-token chunks produced by tokenize_aligned_chunks and
    groups them using is_aligned_whitespace_chunk (which forces a
    singleton group) and is_aligned_with_next_token_chunk (which forces a
    group boundary before the next token). For each resulting group, joins
    the source/dest text, tokenizes both into words, separates punctuation
    tokens from word tokens, and compares the word-token sequences to
    classify the change as one of: '' (no real change), 'remove_tokens',
    'add_tokens', 'split_tokens', 'merge_tokens', or 'replace_tokens',
    based on how the token counts differ and the magnitude of the largest
    per-chunk character-length difference (max_replace/min_replace). Each
    combined chunk is then passed through solve_whitespace_punctuation to
    fix up surrounding punctuation/whitespace handling.

    Args:
        chunks: List of tokenized alignment chunk dicts (as produced by
            tokenize_aligned_chunks).
        debug: Verbosity level for debug printing (0 = silent; >1 prints
            extra grouping detail).

    Returns:
        A list of combined chunk dicts, each with keys 'source', 'dest',
        'align_type' ('aligned' or 'replaced'), 'change_type' (one of the
        classifications above, possibly suffixed by
        solve_whitespace_punctuation), and 'chunks' (the underlying list
        of per-token chunks in that group).
    """
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
    """Placeholder for change classification (currently unimplemented; always returns None).

    Args:
        chunk_group: A list of alignment chunk dicts.

    Returns:
        None, always.
    """
    return None


def align_phrase_chunks(source: str, dest: str):
    """Align two phrase strings character-by-character and combine the result into token-level change chunks.

    Runs the full alignment pipeline: get_alignments_changes (raw
    character-level edit alignment) -> reduce_changes (merge consecutive
    edits) -> tokenize_aligned_chunks (split on whitespace) ->
    combine_tokenized_chunks (group into token-level chunks with
    classified change types).

    Args:
        source: The source phrase string.
        dest: The destination phrase string to align source against.

    Returns:
        The list of combined chunk dicts produced by
        combine_tokenized_chunks, describing the token-level changes
        between source and dest.

    Raises:
        AssertionError: Propagated from tokenize_aligned_chunks if an
            internal whitespace-alignment invariant is violated; the
            source and dest strings are printed before re-raising.
    """
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
    """Pad source/dest text with whitespace so leading/trailing punctuation changes align cleanly.

    When one side of a combined change chunk starts (or ends) with
    punctuation followed (or preceded) by whitespace while the
    corresponding position on the other side is an alphanumeric
    character, this inserts a matching space on the alphanumeric side so
    the two strings line up positionally, and tags the change type with
    '_add_punctuation' or '_remove_punctuation' accordingly (stripping a
    leading underscore if the change_type was empty).

    Args:
        combined_group: A combined chunk dict with 'source', 'dest',
            'align_type', 'change_type', and 'chunks' keys, as produced by
            combine_tokenized_chunks.

    Returns:
        A new combined chunk dict with the same keys, where 'source'
        and/or 'dest' may have an extra leading/trailing space inserted,
        and 'change_type' may have an '_add_punctuation' or
        '_remove_punctuation' suffix appended.
    """
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
    """Detect pairs of n-grams that are word-order permutations of each other and likely interchangeable.

    For each phrase, tokenizes it and slides a window of `ngram_size`
    tokens across it (skipping windows whose first or last token is pure
    punctuation), recording each n-gram and which documents it occurs in.
    For every n-gram, generates all its token permutations and checks
    which other observed n-grams match a permutation, registering those as
    candidate swaps. For each candidate swap, compares the surrounding
    context (the rest of the phrase with the n-gram removed) between
    documents containing each ngram variant: candidates whose surrounding
    context lengths differ by no more than 4 characters and whose
    Levenshtein similarity ratio is at least 0.7 (and not below a stricter
    0.6 check) are confirmed as real word-order swaps. Exact word
    repetitions (e.g. "ever and ever") are skipped to avoid spurious swap
    detection.

    Args:
        phrases: Counter mapping phrase strings to their frequency.
        tokenizer: Tokenizer used to split phrases into tokens.
        ngram_size: Size of the token n-gram window to compare (default 2).
        debug: Verbosity level for debug printing (0 = silent).

    Returns:
        A Counter mapping (ngram_string1, ngram_string2) tuples of
        space-joined token strings to the number of times that swap
        pattern was confirmed across phrase pairs.
    """
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
    """Compute the raw character-level alignment chunks between two strings.

    Uses Levenshtein editops (insert/delete/replace) between source and
    dest, and for each edit operation calls get_alignment_change to
    produce the aligned ('unchanged') chunk preceding it plus the change
    chunk itself, tracking index shifts as the matched/changed prefix is
    consumed from both strings. Any remaining unmatched suffix of source
    and dest after all edits are applied is appended as a final 'aligned'
    chunk.

    Args:
        source: The source string to align.
        dest: The destination string to align source against.
        debug: Verbosity level for debug printing (0 = silent).

    Returns:
        A list of alignment chunk dicts, each with 'source', 'dest', and
        'type' keys ('aligned', 'insert', 'delete', or 'replace'),
        describing the full alignment between source and dest.
    """
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
    """Find similar sub-phrase pairs and aggregate their token-level alignment changes by frequency.

    Builds a SkipgramSimilarity index over all sub-phrases and, for each
    sub-phrase (processed most-frequent first, skipping ones already
    checked), finds skipgram-similar sub-phrases (score_cutoff 0.75).
    Before comparing, any known word swaps from word_swap_freq are applied
    to the candidate similar phrase so word-order variants are normalized
    before alignment. Pairs below `lev_score_threshold` overall Levenshtein
    similarity are discarded. Remaining pairs are aligned with
    align_phrase_chunks, and for each non-trivial ('replaced') aligned
    chunk whose own source/dest similarity is at least 0.5, the
    (source, dest, align_type, change_type) tuple's frequency is
    incremented by the original sub-phrase's frequency.

    Args:
        sub_phrase_freq: Counter mapping sub-phrase strings to their
            frequency.
        word_swap_freq: Counter of (word_swap1, word_swap2) string pairs
            considered interchangeable word-order variants, as produced by
            detect_word_swaps.
        lev_score_threshold: Minimum overall Levenshtein similarity ratio
            required between a sub-phrase and a candidate similar phrase
            for them to be aligned at all (default 0.6).
        debug: Verbosity level for debug printing (0 = silent).

    Returns:
        A tuple (aligned_tokens_freq, token_freq):
            aligned_tokens_freq: Counter mapping
                (source, dest, align_type, change_type) tuples to their
                aggregated frequency.
            token_freq: Counter mapping each chunk's source text to the
                total frequency it was observed with.
    """
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
    """Builds a canonical-spelling/variant mapping from aligned-token frequency data.

    Consumes the (source, dest, align_type, change_type) -> frequency
    Counter produced by get_aligned_token_freq, and for each aligned token
    pair (processed in order of increasing token count, then descending
    frequency, via iterate_tokens) decides which of source/dest should be
    treated as the preferred (canonical) form and which as its variant,
    favouring the more frequent token and chaining/rewriting through
    already-known mappings as new pairs are added. Skips pairs whose
    reverse mapping is already registered, and resolves the rare cases
    where the mapping would create both directions or conflict with prior
    decisions.

    Attributes:
        atf: The input aligned-tokens-frequency Counter.
        tf: The input per-token frequency Counter.
        has_variant: Dict mapping each canonical (preferred) token to the
            set of tokens that are variants of it.
        is_variant_of: Dict mapping each variant token to its canonical
            (preferred) token.
        tokens: Set of all tokens (canonical and variant) seen so far.
        min_freq: Minimum frequency an aligned token pair must have to be
            considered.
        debug: Verbosity level for debug printing (0 = silent).
    """

    def __init__(self, aligned_tokens_freq: Counter, token_freq: Counter,
                 min_freq: int = 0, debug: int = 0):
        """Initialize the mapper and immediately build the variant mapping.

        Args:
            aligned_tokens_freq: Counter mapping
                (source, dest, align_type, change_type) tuples to their
                frequency, as produced by get_aligned_token_freq.
            token_freq: Counter mapping token strings to their frequency.
            min_freq: Minimum frequency an aligned token pair must have to
                be included in the mapping (default 0, i.e. no filtering).
            debug: Verbosity level for debug printing (0 = silent).
        """
        self.atf = aligned_tokens_freq
        self.tf = token_freq
        self.has_variant = defaultdict(set)
        self.is_variant_of = {}
        self.tokens = set()
        self.min_freq = min_freq
        self.debug = debug
        self._get_variant_mapping()

    def _get_best_source(self, source: str):
        """Resolve a candidate source token to its currently preferred canonical form.

        Follows an existing is_variant_of mapping for the exact source
        string, or (if the source has a leading/trailing whitespace
        artifact) for its stripped form, re-applying the stripped
        whitespace to the resolved preferred form and registering it as a
        new variant mapping. Then rewrites any sub-phrases inside the
        source using the current is_variant_of mapping via
        rewrite_context_phrase, and resolves to that rewritten form's
        preferred mapping if one exists.

        Args:
            source: The candidate source token or phrase string.

        Returns:
            The resolved, currently preferred form of source.
        """
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
        """Order aligned token pairs for stable, frequency-aware variant mapping.

        Filters out pairs below self.min_freq, swaps source/dest so the
        more frequent token (by self.tf) is the source, swaps again to
        avoid leading/trailing punctuation ending up on the source side,
        skips pairs whose mapping (in either direction) is already
        registered in self.is_variant_of, and deduplicates. Groups the
        remaining pairs by the source's token count.

        Returns:
            A defaultdict mapping token-count (int) to a Counter of
            (source, dest, change_type) tuples to their frequency, ready
            to be iterated in order of increasing token count.
        """
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
        """Iterate aligned token pairs in processing order (shortest source first, most frequent first).

        Yields:
            Tuples of ((source, dest, change_type), freq), ordered first
            by ascending number of tokens in source, then by descending
            frequency within each token-count group.
        """
        sorted_aligned_tokens_freq = self._sort_aligned_tokens()
        for num_tokens in sorted(sorted_aligned_tokens_freq):
            for aligned_tokens, freq in sorted_aligned_tokens_freq[num_tokens].most_common():
                yield aligned_tokens, freq
        return None

    def _replace_preferred_variant(self, old_pref: str, new_pref: str):
        """Re-point all variants of old_pref to new_pref and demote old_pref to being a variant of new_pref.

        Args:
            old_pref: The currently preferred (canonical) token to demote.
            new_pref: The new preferred (canonical) token to promote.
        """
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
        """Register dest as a variant of source, resolving conflicts with any existing mappings.

        Resolves source to its current best/preferred form, then checks
        whether rewriting dest through the existing is_variant_of mapping
        changes it: if so, compares the frequency of the resolved source
        against the best-known source for the rewritten dest and
        recursively re-maps whichever is less frequent to the other (or,
        if the original source is a substring of the rewritten best
        source, demotes the rewritten best source in favour of the
        original source). If dest is itself already a preferred form with
        its own variants, those variants are folded under source via
        _replace_preferred_variant. If dest already has a registered
        mapping, compares frequencies to decide whether to keep dest's
        existing preferred form or switch to source. In all cases, dest
        is finally recorded as a variant of source (after the above
        adjustments), and self.tokens/self.has_variant/self.is_variant_of
        are updated, followed by an internal consistency check.

        Args:
            source: The (candidate) preferred/canonical token.
            dest: The token to register as a variant of source.
        """
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
        """Build the full is_variant_of mapping by processing all aligned token pairs in order.

        Iterates iterate_tokens (shortest/most-frequent first) and calls
        _map_variant on each (source, dest) pair to incrementally build
        up self.is_variant_of and self.has_variant.

        Returns:
            The completed self.is_variant_of mapping (variant token ->
            its canonical/preferred token).
        """
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
        """Assert internal consistency between self.tokens, self.has_variant, and self.is_variant_of.

        Verifies every token in self.tokens appears as either a key in
        has_variant or is_variant_of, that every key appearing in either
        dict is also in self.tokens, that the two token sets are the same
        size, and that every variant's recorded source actually lists
        that variant in has_variant[source].

        Raises:
            AssertionError: If a token is missing from the expected
                dict(s), or if a variant/source consistency check fails.
            ValueError: If the size of self.tokens differs from the
                combined set of dict keys (after printing both sets for
                debugging).
        """
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
    """Rewrite a list of context phrases to their canonical form using a variant mapping.

    Args:
        context_phrases: List of phrase strings to rewrite.
        is_variant_of: Mapping from variant token/phrase to its canonical
            (preferred) form, as produced by MapVariants.
        debug: Verbosity level for debug printing (0 = silent).

    Returns:
        A dict mapping each original phrase in context_phrases to its
        rewritten (canonicalized) form.
    """
    rewritten_context_phrases = {}
    for phrase in context_phrases:
        rewritten_context_phrases[phrase] = rewrite_context_phrase(phrase, is_variant_of, debug=debug)
    return rewritten_context_phrases


def rewrite_context_phrase(phrase: str, is_variant_of: Dict[str, str], debug: int = 0) -> str:
    """Rewrite a single phrase by substituting any contained variant tokens with their canonical form.

    For every variant in is_variant_of, performs a whole-word regex
    substitution of the variant for its preferred form wherever the
    variant occurs as a word boundary-delimited substring of the phrase.
    Note that variants are applied in dict iteration order, so a phrase
    containing multiple variants may be rewritten through several
    substitutions in sequence.

    Args:
        phrase: The phrase string to rewrite.
        is_variant_of: Mapping from variant token/phrase to its canonical
            (preferred) form.
        debug: Verbosity level for debug printing (0 = silent).

    Returns:
        The rewritten phrase string, with any matched variants replaced
        by their canonical form. Unchanged if no variant was found.
    """
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
    """Merge per-formula-phrase context frequency counters into single pre/post/phrase counters.

    Args:
        context_freq: Dict with 'pre', 'post', and 'phrase' keys, where
            'pre' and 'post' map each formula phrase to a Counter of its
            preceding/following context phrases and their frequencies,
            and 'phrase' is a Counter of formula phrase frequencies, as
            produced by init_context_freq and populated during context
            collection.

    Returns:
        A dict with the same 'pre', 'post', 'phrase' keys, where 'pre' and
        'post' are now flat Counters (summed across all formula phrases)
        mapping context phrase to total frequency, and 'phrase' is a
        Counter of formula phrase frequencies summed from the input.
    """
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
    """Merge per-period context frequency dicts into a single combined context frequency dict.

    Args:
        context_freq: Dict mapping a period key to a context frequency
            dict (with 'pre' and 'post' keys, each mapping formula phrase
            to a Counter of context phrase frequencies), as produced by
            init_context_freq per period.
        periods: Iterable of period keys to merge, in the order they
            should be combined.

    Returns:
        A single context frequency dict (from init_context_freq) with
        'pre' and 'post' Counters summed across all given periods, per
        formula phrase.
    """
    merged_context_freq = init_context_freq()

    for period in periods:
        for direction in ['pre', 'post']:
            for phrase in context_freq[period][direction]:
                for phrase_context in context_freq[period][direction][phrase]:
                    merged_context_freq[direction][phrase][phrase_context] += context_freq[period][direction][phrase][
                        phrase_context]
    return merged_context_freq


def get_sub_phrase_freq(context_freq, tokenizer: Tokenizer, direction: str, debug: int = 0):
    """Expand context phrases (occurring more than once) into all their token-prefix sub-phrases.

    For each context phrase in the given direction occurring more than
    once, tokenizes it and (for 'pre' direction, reversing the token order
    first so growth proceeds outward from the formula) builds every
    prefix sub-phrase (1 token, 2 tokens, ... up to the full phrase),
    re-reversing 'pre' sub-phrases back to natural order, and accumulates
    each sub-phrase's frequency (weighted by the full phrase's frequency).

    Args:
        context_freq: Dict with 'pre'/'post' keys mapping context phrase
            to its frequency (e.g. the merged frequencies from
            merge_phrase_context_freqs).
        tokenizer: Tokenizer used to split context phrases into tokens.
        direction: Either 'pre' or 'post', selecting which side's context
            phrases to expand.
        debug: Verbosity level for debug printing (0 = silent).

    Returns:
        A Counter mapping each sub-phrase string (space-joined tokens) to
        its aggregated frequency.
    """
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
    """Create an empty context frequency structure.

    Returns:
        A dict with 'pre' and 'post' keys mapping to empty
        defaultdict(Counter) (per formula phrase, a Counter of context
        phrase frequencies), and a 'phrase' key mapping to an empty
        Counter (formula phrase frequencies).
    """
    return {
        'pre': defaultdict(Counter),
        'post': defaultdict(Counter),
        'phrase': Counter()
    }


def rewrite_context_variation(context_freq, tokenizer: Tokenizer, min_freq: int = 0):
    """Detect spelling and word-order variation in formula context phrases and rewrite them to canonical forms.

    This is the top-level entry point of the module's variant-detection
    pipeline. For each direction ('pre' and 'post'): merges all per-phrase
    context frequencies, expands context phrases into sub-phrases
    (get_sub_phrase_freq), detects word-order swaps (detect_word_swaps),
    aligns similar sub-phrases and aggregates their token-level changes
    (get_aligned_token_freq), builds a canonical variant mapping
    (MapVariants), and then rewrites every original context phrase
    (per formula phrase) to its canonical form using that mapping.

    Args:
        context_freq: Dict with 'pre' and 'post' keys, each mapping
            formula phrase to a Counter of context phrase frequencies
            (e.g. as produced by init_context_freq and populated during
            context collection).
        tokenizer: Tokenizer used to split phrases into tokens.
        min_freq: Minimum frequency an aligned token pair must have to be
            included when building the variant mapping (default 0).

    Returns:
        A context frequency dict (from init_context_freq) with the
        original 'pre'/'post' phrase-context frequencies rewritten to use
        canonical spelling/word-order forms, plus an additional
        'word_swap' key holding the detected word-swap frequencies per
        direction.
    """
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



