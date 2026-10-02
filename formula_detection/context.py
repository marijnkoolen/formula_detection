"""
context.py — Utilities for collecting, normalising and analysing the
words and phrases that surround a set of target phrases in a corpus.

This module supports building pre/post context word and phrase counts
for a set of (possibly variant) phrases, mapping variant word and phrase
forms to canonical forms using skip-gram similarity, identifying
dominant terms that fill variable slots in templated phrases, and
computing transition probabilities over post-phrase context.
"""
import copy
from collections import defaultdict
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Union

from fuzzy_search.analysis.similarity import SkipgramSimilarity
from fuzzy_search.tokenization.token import Doc
from fuzzy_search.tokenization.token import Token

from formula_detection.variation.edit import compute_variant_similarity
from formula_detection.transitions import compute_transition_probs


def compute_context_word_freq(phrases: Union[str, List[str]], context_count: Dict[str, Dict[str, Counter]],
                              function_words: Optional[Set[str]] = None) -> Dict[str, Counter]:
    """Compute per-direction context word frequencies for a list of phrases.

    Args:
        phrases: One or more phrase strings to collect context for.
        context_count: Nested dict {direction: {phrase: Counter of context strings}}.
        function_words: Optional set of words to keep. When provided only words in
            this set are counted, which is useful for context-anchored variant
            detection across spelling periods where content words vary but function
            words are more stable. Pass None to keep all words.
    """
    if isinstance(phrases, str):
        phrases = [phrases]
    context_word_freq = defaultdict(Counter)
    for direction in {'pre', 'post'}:
        for phrase in phrases:
            for context_phrase in context_count[direction][phrase]:
                for context_word in context_phrase.strip().split(' '):
                    if len(context_word) == 0:
                        continue
                    if function_words is not None and context_word not in function_words:
                        continue
                    context_word_freq[direction][context_word] += context_count[direction][phrase][context_phrase]
    return context_word_freq


def compute_context_skip_sim(phrases: Union[str, List[str]], context_count: Dict[str, Dict[str, Counter]],
                             include_boundaries: bool = True,
                             function_words: Optional[Set[str]] = None) -> SkipgramSimilarity:
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
                        if function_words is not None and w not in function_words:
                            continue
                        terms.add(w)
    return SkipgramSimilarity(ngram_length=2, skip_length=2, terms=list(terms))


def map_word_variants(context_freq: Dict[str, Counter], skip_sim: SkipgramSimilarity,
                      term_freq: Counter = None, w2v_model=None, sim_threshold: float = 0.75,
                      known_variants: Dict[str, str] = None,
                      function_words: Optional[Set[str]] = None) -> Dict[str, str]:
    """Map variant word forms to their canonical counterparts using skip-gram similarity.

    Args:
        context_freq: Per-direction Counter of word frequencies {direction: Counter}.
        skip_sim: Pre-built SkipgramSimilarity index over the context vocabulary.
        term_freq: Optional overall term frequencies used to prefer higher-frequency forms
            as canonicals and to skip very common words.
        w2v_model: Optional word2vec model for additional similarity evidence.
        sim_threshold: Minimum combined similarity score to accept a variant mapping.
        known_variants: Pre-known {variant: canonical} mappings applied first.
        function_words: Optional set of words to restrict evidence to. When provided,
            only these words participate in similarity comparisons. Useful when context
            windows contain many spelling-variable content words whose skipgram profiles
            are unreliable across time periods.
    """
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
            if function_words is not None and variant_word not in function_words:
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
    """Map variant forms of words that fill a variable slot to a canonical form.

    Builds a skip-gram similarity index directly from the keys of
    variant_freq and delegates to map_word_variants to find and score
    variant mappings. Useful for normalising the set of words observed
    in a `<VAR>` slot of a templated phrase.

    Args:
        variant_freq: Counter of candidate variable-word frequencies.
        w2v_model: Optional word2vec model providing additional similarity
            evidence alongside skip-gram and edit-based similarity.
        sim_threshold: Minimum combined similarity score required to accept
            a variant mapping.

    Returns:
        Dict mapping each variant word to its canonical word.
    """
    skip_sim = SkipgramSimilarity(ngram_length=2, skip_length=2, terms=list(variant_freq.keys()))
    variant_of = map_word_variants(variant_freq, skip_sim,
                                   w2v_model=w2v_model, sim_threshold=sim_threshold)
    return variant_of


def map_context_word_variants(phrases: Union[str, List[str]], context_count: Dict[str, Dict[str, Counter]],
                              term_freq: Counter = None, w2v_model=None, sim_threshold: float = 0.75,
                              known_variants: Dict[str, str] = None,
                              include_boundaries: bool = True,
                              function_words: Optional[Set[str]] = None) -> Dict[str, str]:
    """Build a variant map for context words around the given phrases.

    The function_words parameter restricts evidence to stable function words,
    which is recommended for corpora with significant spelling variation across
    time periods or HTR errors — content words in the context window may vary
    too much to be reliable similarity anchors.
    """
    variant_of = copy.deepcopy(known_variants)
    phrases = phrases if isinstance(phrases, (list, set)) else [phrases]
    skip_sim = compute_context_skip_sim(phrases, context_count, include_boundaries=include_boundaries,
                                        function_words=function_words)
    context_word_freq = compute_context_word_freq(phrases, context_count, function_words=function_words)
    variant_map = map_word_variants(context_word_freq, skip_sim, term_freq=term_freq,
                                    w2v_model=w2v_model, sim_threshold=sim_threshold,
                                    known_variants=known_variants, function_words=function_words)
    return variant_map


def find_dominant_terms(variant_freq: Counter, variant_of: Dict[str, str],
                        min_frac: float = 0.1) -> List[str]:
    """Identify the most frequent canonical terms filling a variable slot.

    Frequencies of variant forms are first folded into their canonical
    (main) word via variant_of, then each canonical word's share of the
    total frequency is compared against min_frac to decide whether it is
    "dominant" enough to be reported.

    Args:
        variant_freq: Counter of observed (variant) word frequencies.
        variant_of: Dict mapping variant words to their canonical word.
            Words not present as a key are treated as already canonical.
        min_frac: Minimum fraction of the total frequency a canonical word
            must reach to be considered dominant.

    Returns:
        List of canonical words whose mapped frequency share is at least
        min_frac.
    """
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
    """Instantiate a templated phrase's `<VAR>` slot(s) with dominant terms.

    For each dominant term, the term's space-separated words are substituted
    one-by-one, in order, for successive `<VAR>` placeholders in phrase.

    Args:
        phrase: A phrase string containing one or more `<VAR>` placeholders.
        dominant_terms: List of (possibly multi-word) terms to substitute
            into phrase, one fully instantiated phrase per term.

    Returns:
        List of phrases with `<VAR>` placeholders replaced by the words of
        each dominant term, in the same order as dominant_terms.
    """
    dominant_phrases = []
    for dominant_term in dominant_terms:
        variable_terms = dominant_term.split(' ')
        dominant_phrase = phrase
        for variable_term in variable_terms:
            dominant_phrase = dominant_phrase.replace('<VAR>', variable_term, 1)
        dominant_phrases.append(dominant_phrase)
    return dominant_phrases


def make_main_phrase_map(phrases: Union[List[str], Dict[str, Set[str]]]):
    """Build a mapping from every phrase (and its variants) to a main phrase.

    Args:
        phrases: Either a list of phrases (each phrase maps to itself), or a
            dict mapping a main phrase to a set of its variant phrases (each
            variant maps to that main phrase, in addition to the main phrase
            mapping to itself).

    Returns:
        Dict mapping each phrase string (main or variant) to its main phrase.
        If a variant phrase appears under multiple main phrases, the mapping
        from the first main phrase encountered is kept.
    """
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
    """Count words/phrases occurring just before and after target phrases.

    Iterates over sentences (or other word-bearing units), locates each
    occurrence of any of the target phrases (or their variants), and
    accumulates frequency counts of the pre- and post-context word windows
    (joined as strings), as well as overall occurrence counts, keyed by the
    phrase's main phrase form.

    Args:
        phrases: Either a list of phrases, or a dict mapping a main phrase
            to a set of its variant phrases. Used to build the main phrase
            mapping via make_main_phrase_map.
        sent_iterator: Iterable of sentence-like items. Each item may be a
            fuzzy_search Doc, a dict with a 'words' key, or a list of
            strings/Tokens.
        context_size: Number of words to take immediately before and after
            each phrase occurrence as its context window.

    Returns:
        Dict with keys 'phrase', 'pre' and 'post'. 'phrase' maps to a
        Counter of occurrence counts per main phrase. 'pre' and 'post' map
        to a dict of {main_phrase: Counter} of context-string frequencies
        for the words immediately before/after each occurrence.

    Raises:
        TypeError: If an item from sent_iterator is not a Doc, a dict with
            a 'words' key, or a list of strings/Tokens.
    """
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
        if isinstance(sent, Doc):
            words = [token.n for token in sent]
        elif isinstance(sent, dict) and 'words' in sent:
            words = sent['words']
        elif isinstance(sent, list):
            words = [token.n if isinstance(token, Token) else token for token in sent]
        else:
            raise TypeError(f"sent must be a Doc, a dict with a 'words' key, or a list of "
                            f"strings/Tokens, not {type(sent)}")
        for wi in range(len(words)):
            for tup_len in tuple_lengths:
                word_tuple = tuple(words[wi:wi+tup_len])
                if len(word_tuple) != tup_len:
                    continue
                if word_tuple in phrase_tuple_map:
                    main_phrase = main_phrase_map[phrase_tuple_map[word_tuple]]
                    start = wi - context_size if wi >= context_size else 0
                    pre_words = words[start:wi]
                    post_words = words[wi+tup_len:wi+tup_len+context_size]
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
    """Collects and analyses the context surrounding a set of phrases.

    Wraps count_pre_post_phrase_context and compute_transition_probs to
    provide a stateful interface for building pre/post context word counts
    for a set of target phrases and deriving transition probabilities over
    the post-phrase context.

    Attributes:
        phrases: List of target phrases to track context for.
        sentences: Default iterable of sentence-like units to count context
            from, used when count_phrase_contexts is called without an
            explicit sentences argument.
        known_variants: Dict mapping known variant phrases/words to their
            canonical form.
        context_count: Result of count_pre_post_phrase_context, populated
            by count_phrase_contexts. None until then.
        w2v_model: Optional word2vec model usable for downstream variant
            similarity computations.
        trans_probs: Dict mapping each phrase to its computed post-context
            transition probabilities, populated by
            compute_post_context_transitions.
    """

    def __init__(self, phrases: List[str], sentences: Iterable = None,
                 known_variants: Dict[str, Set[str]] = None,
                 w2v_model=None):
        """Initialise a PhraseContext for a set of target phrases.

        Args:
            phrases: List of target phrases to track context for.
            sentences: Optional default iterable of sentence-like units
                used by count_phrase_contexts when no sentences are passed
                explicitly.
            known_variants: Optional dict mapping known variant phrases or
                words to their canonical form.
            w2v_model: Optional word2vec model for use in variant similarity
                computations.
        """
        self.phrases = phrases
        self.sentences = sentences
        self.known_variants = known_variants if known_variants else {}
        self.context_count = None
        self.w2v_model = w2v_model if w2v_model is not None else None
        self.trans_probs = {}

    def count_phrase_contexts(self, sentences: Iterable = None):
        """Count pre/post context words for self.phrases and store the result.

        Delegates to count_pre_post_phrase_context and stores its output in
        self.context_count.

        Args:
            sentences: Iterable of sentence-like units to scan for phrase
                occurrences. If None, falls back to self.sentences.
        """
        if sentences is None:
            sentences = self.sentences
        self.context_count = count_pre_post_phrase_context(self.phrases, sentences)

    def compute_post_context_transitions(self, phrase: str = None, variant_of: Dict[str, str] = None):
        """Compute post-context transition probabilities for one or all phrases.

        Requires count_phrase_contexts to have been called first, so that
        self.context_count is populated.

        Args:
            phrase: If given, compute and return transition probabilities
                for just this phrase. If None, compute transition
                probabilities for every phrase in self.phrases and store
                them in self.trans_probs.
            variant_of: Optional dict mapping variant words to their
                canonical form, passed through to compute_transition_probs.

        Returns:
            When phrase is given: the computed transition probabilities for
            that phrase, or None if phrase is not present in
            self.context_count (a message is printed in that case). When
            phrase is None: nothing is returned (results are stored in
            self.trans_probs).
        """
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
