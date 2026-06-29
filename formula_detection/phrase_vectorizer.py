"""
phrase_vectorizer.py — Corpus-internal feature vectors for candidate phrases.

Builds sparse tf-idf-weighted context-word vectors from the pre/post context
counts stored in a PhraseContext object.  No external linguistic resources are
required, making this fully language-agnostic.

Optionally restrict context evidence to a ``function_words`` set, which
improves stability when content-word spelling varies across time periods or
when HTR quality is uneven.
"""
import math
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse

from formula_detection.context import PhraseContext


class PhraseVectorizer:
    """Builds sparse feature vectors for a list of candidate phrases.

    Features are tf-idf-weighted frequencies of context words appearing
    in the pre and post context windows stored in a ``PhraseContext``
    instance.  The resulting matrix can be fed directly to
    ``formula_detection.clustering.cluster_phrases``.

    Attributes:
        phrase_context: Source of pre/post context word frequencies.
        function_words: When set, only words in this set contribute to
            the vectors.  Recommended for corpora with variable spelling.
        _context_vocab: Mapping from context word string to column index,
            built during the first call to ``vectorize``.
    """

    def __init__(self, phrase_context: PhraseContext,
                 function_words: Optional[Set[str]] = None):
        """
        Args:
            phrase_context: A ``PhraseContext`` instance whose
                ``count_phrase_contexts`` method has already been called so
                that ``context_count`` is populated.
            function_words: Optional set of word strings to use as context
                features.  Pass None to use all context words.  Using a
                curated set of function words that are stable across spelling
                periods tends to produce more reliable clusters for historical
                corpora.
        """
        self.phrase_context = phrase_context
        self.function_words = function_words
        self._context_vocab: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def vectorize(self, phrases: List[str]) -> Tuple[scipy.sparse.csr_matrix, List[str]]:
        """Build a (n_phrases × n_features) sparse feature matrix.

        Each row corresponds to one phrase in *phrases* (same order).
        Each column corresponds to a context word observed across all phrases.
        Values are tf-idf scores: within-phrase tf × log-smoothed idf.

        Args:
            phrases: Ordered list of phrase strings to vectorize.  All
                phrases must be present in ``phrase_context.context_count``;
                phrases with no context evidence produce all-zero rows.

        Returns:
            A tuple ``(matrix, phrases)`` where *matrix* is a
            ``scipy.sparse.csr_matrix`` and *phrases* is the same list
            passed in (returned for convenience when the order matters).
        """
        self._build_context_vocab(phrases)
        global_word_freq = self._global_word_freq(phrases)

        n_phrases = len(phrases)
        n_features = len(self._context_vocab)
        if n_features == 0:
            empty = scipy.sparse.csr_matrix((n_phrases, 0), dtype=np.float32)
            return empty, phrases

        rows, cols, data = [], [], []
        for pi, phrase in enumerate(phrases):
            for col, val in self._tfidf_vector(phrase, global_word_freq).items():
                rows.append(pi)
                cols.append(col)
                data.append(val)

        matrix = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_phrases, n_features),
            dtype=np.float32,
        )
        return matrix, phrases

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _context_word_counts(self, phrase: str) -> Counter:
        """Return per-word context frequency for *phrase*."""
        word_freq: Counter = Counter()
        context_count = getattr(self.phrase_context, 'context_count', {}) or {}
        for direction in ('pre', 'post'):
            for context_phrase, count in context_count.get(direction, {}).get(phrase, {}).items():
                for word in context_phrase.strip().split():
                    if not word:
                        continue
                    if self.function_words is not None and word not in self.function_words:
                        continue
                    word_freq[word] += count
        return word_freq

    def _build_context_vocab(self, phrases: List[str]) -> None:
        all_words: Set[str] = set()
        for phrase in phrases:
            all_words.update(self._context_word_counts(phrase).keys())
        self._context_vocab = {w: i for i, w in enumerate(sorted(all_words))}

    def _global_word_freq(self, phrases: List[str]) -> Counter:
        """Document frequency: number of phrases each word appears near."""
        doc_freq: Counter = Counter()
        for phrase in phrases:
            doc_freq.update(self._context_word_counts(phrase).keys())
        return doc_freq

    def _tfidf_vector(self, phrase: str, global_word_freq: Counter) -> Dict[int, float]:
        word_freq = self._context_word_counts(phrase)
        total = sum(word_freq.values()) or 1
        n_docs = len(self._context_vocab) or 1
        vec: Dict[int, float] = {}
        for word, count in word_freq.items():
            if word not in self._context_vocab:
                continue
            idx = self._context_vocab[word]
            tf = count / total
            idf = math.log((1 + n_docs) / (1 + global_word_freq[word]))
            score = tf * idf
            if score != 0.0:
                vec[idx] = score
        return vec
