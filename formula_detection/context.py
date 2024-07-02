import copy
from collections import defaultdict
from collections import Counter
from typing import Dict, Iterable, List, Set, Union

from fuzzy_search.similarity import SkipgramSimilarity

from formula_detection.variation.edit import compute_variant_similarity
from formula_detection.transitions import compute_transition_probs


def compute_context_word_freq(phrases: Union[str, List[str]], context_count: Dict[str, Dict[str, Counter]]) -> Dict[str, Counter]:
    if isinstance(phrases, str):
        phrases = [phrases]
    context_word_freq = defaultdict(Counter)
    for direction in {'pre', 'post'}:
        for phrase in phrases:
            for context_phrase in context_count[direction][phrase]:
                for context_word in context_phrase.strip().split(' '):
                    if len(context_word) == 0:
                        continue
                    context_word_freq[direction][context_word] += context_count[direction][phrase][context_phrase]
    return context_word_freq


def compute_context_skip_sim(phrases: Union[str, List[str]], context_count: Dict[str, Dict[str, Counter]],
                             include_boundaries: bool = True) -> SkipgramSimilarity:
    if isinstance(phrases, str):
        phrases = [phrases]
    terms = set()
    for direction in {'pre', 'post'}:
        for phrase in phrases:
            if include_boundaries and not phrase.startswith('<START>'):
                phrase = f"<START> {phrase} <END>"
            for pw in context_count[direction][phrase]:
                for w in pw.split(' '):
                    if len(w) > 0:
                        terms.add(w)
    return SkipgramSimilarity(ngram_length=2, skip_length=2, terms=list(terms))


def map_word_variants(context_freq: Dict[str, Counter], skip_sim: SkipgramSimilarity,
                      term_freq: Counter = None, w2v_model = None, sim_threshold: float = 0.75,
                      known_variants: Dict[str, str] = None) -> Dict[str, str]:
    variant_of = {}
    mapped = set()
    if known_variants is not None:
        for variant_word in known_variants:
            main_word = known_variants[variant_word]
            variant_of[variant_word] = main_word
            mapped.add(variant_word)
            mapped.add(main_word)

    for direction in context_freq:
        for variant_word, freq in context_freq[direction].most_common():
            if variant_word in mapped:
                continue
            mapped.add(variant_word)
            for sim_word, skip_sim_score in skip_sim.rank_similar(variant_word, top_n=1000):
                if skip_sim_score < 0.5:
                    continue
                scores = [skip_sim_score]
                if sim_word == variant_word or sim_word in mapped:
                    continue
                if term_freq is not None:
                    if term_freq[sim_word] > term_freq[variant_word]:
                        # print(f'skipping higher frequency variant #{sim_word}# ({term_freq[sim_word]}) of suggestion '
                        #       f'#{variant_word}# ({term_freq[variant_word]})')
                        continue
                    elif term_freq[sim_word] > 1000:
                        continue
                    elif term_freq[sim_word] > 100 and term_freq[sim_word] / term_freq[variant_word] > 10:
                        # print(f'skipping common variant #{sim_word}# ({term_freq[sim_word]}) of suggestion '
                        #       f'#{variant_word}# ({term_freq[variant_word]}, {skip_sim_score})')
                        continue
                    elif term_freq[sim_word] > 100:
                        # print(f'common variant #{sim_word}# ({term_freq[sim_word]}#, {skip_sim_score}')
                        pass
                if w2v_model is not None:
                    if variant_word not in w2v_model.wv or sim_word not in w2v_model.wv:
                        wv_sim_score = 0
                    else:
                        wv_sim_score = w2v_model.wv.similarity(variant_word, sim_word)
                    scores.append(wv_sim_score)
                try:
                    variant_score = compute_variant_similarity(variant_word, sim_word)
                except ZeroDivisionError:
                    print(f'direction: {direction}\tvariant_word: #{variant_word}#\tsim_word: #{sim_word}#')
                    raise
                scores.append(variant_score)
                score = sum(scores) / len(scores)
                if score >= sim_threshold:
                    mapped.add(sim_word)
                    variant_of[sim_word] = variant_word
                    # print('\t', sim_word, skip_sim_score, wv_sim_score, variant_score)
    return variant_of


def map_variable_word_variants(variant_freq: Counter, w2v_model = None,
                               sim_threshold: float = 0.5) -> Dict[str, str]:
    skip_sim = SkipgramSimilarity(ngram_length=2, skip_length=2, terms=list(variant_freq.keys()))
    variant_of = map_word_variants(variant_freq, skip_sim,
                                   w2v_model=w2v_model, sim_threshold=sim_threshold)
    return variant_of


def map_context_word_variants(phrases: Union[str, List[str]], context_count: Dict[str, Dict[str, Counter]],
                              term_freq: Counter = None, w2v_model=None, sim_threshold: float = 0.75,
                              known_variants: Dict[str, str] = None,
                              include_boundaries: bool = True) -> Dict[str, str]:
    variant_of = copy.deepcopy(known_variants)
    phrases = phrases if isinstance(phrases, (list,set)) else [phrases]
    # print('creating skip sim')
    skip_sim = compute_context_skip_sim(phrases, context_count, include_boundaries=include_boundaries)
    # print('\tvocabulary:', len(skip_sim.vocabulary))
    # print('counting context word freq')
    context_word_freq = compute_context_word_freq(phrases, context_count)
    tokens = sum(sum(context_word_freq[direction].values()) for direction in {'pre', 'post'})
    types = sum([len(context_word_freq[direction]) for direction in {'pre', 'post'}])
    # print(f'\tnum context word types: {types}, tokens: {tokens}')
    # print('creating variant_map')
    variant_map = map_word_variants(context_word_freq, skip_sim, term_freq=term_freq,
                                    w2v_model=w2v_model, sim_threshold=sim_threshold,
                                    known_variants=known_variants)
    # print('\tnum variants:', len(variant_map))
    return variant_of


def find_dominant_terms(variant_freq: Counter, variant_of: Dict[str, str],
                        min_frac: float = 0.1) -> List[str]:
    dominant_terms = []
    mapped_freq = Counter()
    for variant_word in variant_of:
        main_word = variant_of[variant_word]
        mapped_freq[main_word] += variant_freq[variant_word]
    for word in variant_freq:
        if word in variant_of:
            continue
        mapped_freq[word] += variant_freq[word]
    total = sum(mapped_freq.values())
    for main_word in mapped_freq:
        # print(f'{main_word: <20}{mapped_freq[main_word]: >8}{mapped_freq[main_word] / total: >6.2f}')
        if mapped_freq[main_word] / total >= min_frac:
            dominant_terms.append(main_word)
    return dominant_terms


def construct_dominant_phrases(phrase: str, dominant_terms: List[str]) -> List[str]:
    dominant_phrases = []
    for dominant_term in dominant_terms:
        variable_terms = dominant_term.split(' ')
        dominant_phrase = phrase
        for variable_term in variable_terms:
            dominant_phrase = dominant_phrase.replace('<VAR>', variable_term, 1)
        dominant_phrases.append(dominant_phrase)
    return dominant_phrases


def make_main_phrase_map(phrases: Union[List[str], Dict[str, Set[str]]]):
    main_phrase_map = {}
    if isinstance(phrases, dict):
        for main_phrase in phrases:
            main_phrase_map[main_phrase] = main_phrase
            for variant_phrase in phrases[main_phrase]:
                if variant_phrase in main_phrase_map:
                    # if variant maps to multiple mains,
                    # assume the earlier one is the better one
                    continue
                main_phrase_map[variant_phrase] = main_phrase
    elif isinstance(phrases, list):
        for main_phrase in phrases:
            main_phrase_map[main_phrase] = main_phrase
    return main_phrase_map


def count_pre_post_phrase_context(phrases: Union[List[str], Dict[str, Set[str]]],
                                  sent_iterator: Iterable, context_size: int = 5):
    pre_context_count = defaultdict(Counter)
    post_context_count = defaultdict(Counter)
    phrase_count = Counter()
    main_phrase_map = make_main_phrase_map(phrases)

    phrase_tuple_map = {}
    tuple_lengths = set()
    for phrase in main_phrase_map:
        phrase_tuple = tuple(phrase.split(' '))
        phrase_tuple_map[phrase_tuple] = phrase
        tuple_lengths.add(len(phrase_tuple))
    for si, sent in enumerate(sent_iterator):
        for wi in range(len(sent['words'])):
            for tup_len in tuple_lengths:
                word_tuple = tuple(sent['words'][wi:wi+tup_len])
                if len(word_tuple) != tup_len:
                    continue
                if word_tuple in phrase_tuple_map:
                    main_phrase = main_phrase_map[phrase_tuple_map[word_tuple]]
                    start = wi - context_size if wi >= context_size else 0
                    pre_words = sent['words'][start:wi]
                    post_words = sent['words'][wi+tup_len:wi+tup_len+context_size]
                    if len(pre_words) > 0:
                        pre_context_count[main_phrase].update([' '.join(pre_words)])
                    if len(post_words) > 0:
                        post_context_count[main_phrase].update([' '.join(post_words)])
                    phrase_count.update([main_phrase])
    return {
        'phrase': phrase_count,
        'pre': pre_context_count,
        'post': post_context_count
    }


class PhraseContext:

    def __init__(self, phrases: List[str], sentences: Iterable = None,
                 known_variants: Dict[str, Set[str]] = None,
                 w2v_model=None):
        self.phrases = phrases
        self.sentences = sentences
        self.known_variants = known_variants if known_variants else {}
        self.context_count = None
        self.w2v_model = w2v_model if w2v_model is not None else None
        self.trans_probs = {}

    def count_phrase_contexts(self, sentences: Iterable = None):
        if sentences is None:
            sentences = self.sentences
        self.context_count = count_pre_post_phrase_context(self.phrases, sentences)

    def compute_post_context_transitions(self, phrase: str = None, variant_of: Dict[str, str] = None):
        if phrase is not None:
            if phrase not in self.context_count:
                print(f'no context counts for phrase', phrase)
                return None
            else:
                return compute_transition_probs(phrase, self.context_count['post'],
                                                variant_of=variant_of)
        else:
            for phrase in self.phrases:
                transition_probs = compute_transition_probs(phrase, self.context_count['post'],
                                                            variant_of=variant_of)
                self.trans_probs[phrase] = transition_probs
