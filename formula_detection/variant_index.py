"""
variant_index.py — VariantIndex and supporting utilities.

Maps surface word forms to canonical forms using a cascade of up to three
evidence passes.  The class is fully language-agnostic: every linguistic
resource (spelling normaliser, function-word list) is optional and must be
supplied by the caller.  Omitting all three passes leaves the index empty
and has no effect on downstream processing.
"""
from collections import Counter
from typing import Callable, Dict, List, Optional, Set

from fuzzy_search.analysis.similarity import SkipgramSimilarity

from formula_detection.variation.edit import compute_variant_similarity
from formula_detection.vocabulary import Vocabulary


class VariantIndex:
    """Maps surface word forms to canonical forms using a cascade of evidence.

    Three passes are available; each is skipped when its prerequisite is absent:

    Pass 1 — Rule-based spelling normalisation
        Requires: a ``spelling_normaliser`` callable (e.g.
        ``normalise_spelling`` from
        ``formula_detection.normalisation.rewrite_historic_dutch``).
        Language-specific; omit for language-agnostic use.

    Pass 2 — Edit-distance grouping
        Requires: a populated ``Vocabulary`` and ``term_freq`` Counter.
        Groups low-frequency types onto their most similar high-frequency
        neighbour, using skipgram pre-filtering followed by
        ``compute_variant_similarity``.  Always available but may be
        slow for very large vocabularies; use ``min_freq_for_edit_dist``
        to restrict it to terms above a frequency floor.

    Pass 3 — Context-anchored grouping
        Requires: a ``PhraseContext`` instance passed to ``build()``.
        Uses function-word context slots as evidence, which is more robust
        than content-word context across spelling periods or languages with
        high orthographic variation.  Pass a ``function_words`` set to
        restrict the context evidence to stable words.

    All mappings record the original surface form so no information is lost.
    """

    def __init__(self,
                 term_freq: Counter,
                 vocab: Vocabulary,
                 spelling_normaliser: Optional[Callable[[str], str]] = None,
                 sim_threshold: float = 0.82,
                 function_words: Optional[Set[str]] = None,
                 min_freq_for_edit_dist: int = 2):
        """
        Args:
            term_freq: Counter mapping vocabulary term IDs to frequencies.
            vocab: The Vocabulary that was used to build term_freq.
            spelling_normaliser: Optional callable mapping a surface string to
                its canonical spelling.  When None, pass 1 is skipped.  Keep
                None for language-agnostic use and supply a language-specific
                function only when needed.
            sim_threshold: Minimum combined similarity score (0–1) for
                accepting a variant mapping in pass 2.
            function_words: Optional set of word strings.  When provided,
                pass 3 uses only these words as context evidence.  Recommended
                for corpora where content-word spelling varies substantially
                across time periods or HTR quality levels.
            min_freq_for_edit_dist: Only terms with corpus frequency >= this
                value are considered as canonical targets in pass 2.  Keeps
                the comparison set manageable for large vocabularies.
        """
        self.term_freq = term_freq
        self.vocab = vocab
        self.spelling_normaliser = spelling_normaliser
        self.sim_threshold = sim_threshold
        self.function_words = function_words
        self.min_freq_for_edit_dist = min_freq_for_edit_dist
        # surface -> canonical
        self._canonical: Dict[str, str] = {}
        # canonical -> set of surface forms
        self._surfaces: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def canonical(self, term: str) -> str:
        """Return the canonical form for *term*, or *term* itself if unmapped."""
        return self._canonical.get(term, term)

    def surface_forms(self, canonical: str) -> Set[str]:
        """Return all surface forms that map to *canonical*."""
        return set(self._surfaces.get(canonical, {canonical}))

    def build(self, phrase_context=None,
              seed_phrases: Optional[List[str]] = None) -> None:
        """Run the variant-detection cascade and populate the index.

        Args:
            phrase_context: Optional ``PhraseContext`` instance (from
                ``formula_detection.context``) providing pre/post context
                counts.  Required for pass 3; pass None to skip it.
            seed_phrases: Optional list of known formula strings used to
                restrict pass 3 to context windows around those phrases.
                When None, all phrases with context counts are used.
        """
        self._pass1_rule_based()
        self._pass2_edit_distance()
        if phrase_context is not None:
            self._pass3_context_anchored(phrase_context, seed_phrases)

    def reindex_counter(self, freq: Counter) -> Counter:
        """Collapse a term-ID Counter onto canonical-form term IDs.

        Terms that have no mapping are kept under their original ID.

        Args:
            freq: Counter mapping term IDs (ints) to frequencies.

        Returns:
            New Counter with all variant IDs merged into their canonical ID.
        """
        result: Counter = Counter()
        for term_id, count in freq.items():
            term = self.vocab.id2term(term_id)
            if term is None:
                continue
            canonical_str = self.canonical(term)
            canonical_id = self.vocab.term2id(canonical_str)
            if canonical_id is None:
                canonical_id = term_id
            result[canonical_id] += count
        return result

    # ------------------------------------------------------------------
    # Private passes
    # ------------------------------------------------------------------

    def _add_mapping(self, surface: str, canonical: str) -> None:
        self._canonical[surface] = canonical
        self._surfaces.setdefault(canonical, set()).add(surface)

    def _pass1_rule_based(self) -> None:
        """Apply the spelling normaliser to every vocabulary term."""
        if self.spelling_normaliser is None:
            return
        for term in list(self.vocab.term_id.keys()):
            canonical = self.spelling_normaliser(term)
            if canonical != term:
                self._add_mapping(term, canonical)

    def _pass2_edit_distance(self) -> None:
        """Group low-frequency types onto high-frequency near-neighbours.

        Uses a SkipgramSimilarity index for fast candidate pre-filtering,
        then accepts pairs whose ``compute_variant_similarity`` score meets
        ``sim_threshold``.  Only terms at or above ``min_freq_for_edit_dist``
        are eligible as canonical targets so that the comparison set stays
        tractable for large vocabularies.
        """
        term_id_freq = {self.vocab.id2term(tid): freq
                        for tid, freq in self.term_freq.items()
                        if self.vocab.id2term(tid) is not None}
        # Restrict candidate pool to terms above the frequency floor.
        candidate_terms = [t for t, f in term_id_freq.items()
                           if f >= self.min_freq_for_edit_dist]
        if not candidate_terms:
            return

        skip_sim = SkipgramSimilarity(ngram_length=2, skip_length=2,
                                      terms=candidate_terms)
        # Process from highest to lowest frequency so high-freq forms
        # are established as canonicals before low-freq ones are mapped.
        sorted_terms = sorted(candidate_terms,
                              key=lambda t: term_id_freq.get(t, 0),
                              reverse=True)
        established: Set[str] = set()
        for term in sorted_terms:
            if term in self._canonical:
                continue  # already mapped by pass 1
            established.add(term)
            term_freq_val = term_id_freq.get(term, 0)
            for sim_term, score in skip_sim.rank_similar(term, top_n=20):
                if score < 0.5:
                    break
                if sim_term == term or sim_term in self._canonical or sim_term in established:
                    continue
                sim_freq = term_id_freq.get(sim_term, 0)
                if sim_freq >= term_freq_val:
                    continue  # only map lower-frequency -> higher-frequency
                # print(f"VariantIndex._pass2_edit_distance - term: {term}\tsim_term: {sim_term}")
                edit_sim = compute_variant_similarity(term, sim_term)
                if edit_sim >= self.sim_threshold:
                    self._add_mapping(sim_term, term)

    def _pass3_context_anchored(self, phrase_context,
                                seed_phrases: Optional[List[str]]) -> None:
        """Use function-word context slots to find additional variant pairs.

        Calls the existing ``map_context_word_variants`` pipeline in
        ``formula_detection.context``, restricted to ``function_words`` when
        provided.
        """
        from formula_detection.context import map_context_word_variants

        context_count = getattr(phrase_context, 'context_count', None)
        if not context_count:
            return

        if seed_phrases is None:
            seed_phrases = list(context_count.get('pre', {}).keys())
        if not seed_phrases:
            return

        # Build a string-keyed frequency counter for the context helper.
        str_term_freq: Counter = Counter({
            self.vocab.id2term(tid): freq
            for tid, freq in self.term_freq.items()
            if self.vocab.id2term(tid) is not None
        })

        variant_map = map_context_word_variants(
            seed_phrases,
            context_count,
            term_freq=str_term_freq,
            function_words=self.function_words,
        )
        for surface, canonical in variant_map.items():
            if surface not in self._canonical:
                self._add_mapping(surface, canonical)
