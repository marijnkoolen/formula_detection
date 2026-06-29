"""
clustering.py — Semantic clustering of candidate phrases.

Uses agglomerative clustering with Ward linkage over the sparse feature
vectors produced by PhraseVectorizer.  Ward linkage is preferred because
the number of clusters is unknown a priori and cluster sizes are expected
to be very unequal.  The dendrogram can be inspected at different
granularity levels by varying distance_threshold.

The PhraseClusters result class is kept here alongside the algorithm that
produces it.  The related FormulaSignature data class (a community-level
concept derived from the co-occurrence graph) lives in
formula_detection.patterns.pattern.
"""
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from formula_detection.patterns.pattern import PhraseClusters


def cluster_phrases(matrix: scipy.sparse.csr_matrix,
                    phrases: List[str],
                    n_clusters: Optional[int] = None,
                    distance_threshold: float = 0.4) -> PhraseClusters:
    """Cluster *phrases* using Ward agglomerative clustering.

    Exactly one of *n_clusters* or *distance_threshold* controls the
    number of clusters produced:

    - Pass *n_clusters* when you know the expected number of formula
      types.
    - Pass only *distance_threshold* (the default) when the number of
      clusters is unknown — this is the common case.  Lower values
      produce more, finer-grained clusters.

    The input *matrix* is L2-normalised before clustering so that phrase
    vectors of different overall magnitude are treated equally.

    Args:
        matrix: ``(n_phrases, n_features)`` sparse matrix as returned by
            ``PhraseVectorizer.vectorize``.
        phrases: Ordered list of phrase strings matching the rows of
            *matrix*.
        n_clusters: Fixed number of clusters.  When provided,
            *distance_threshold* is ignored.
        distance_threshold: Maximum inter-cluster distance at which
            clusters are merged.  Used only when *n_clusters* is None.

    Returns:
        A ``PhraseClusters`` instance mapping each phrase to a cluster ID.
    """
    if n_clusters is not None:
        # sklearn requires distance_threshold=None when n_clusters is set
        effective_threshold = None
        compute_full_tree = False
    else:
        effective_threshold = distance_threshold
        compute_full_tree = True

    dense = normalize(matrix.toarray(), norm='l2')

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=effective_threshold,
        metric='euclidean',
        linkage='ward',
        compute_full_tree=compute_full_tree,
    )
    labels: np.ndarray = model.fit_predict(dense)

    phrase_to_cluster: Dict[str, int] = {
        phrase: int(label) for phrase, label in zip(phrases, labels)
    }
    return PhraseClusters(phrase_to_cluster)
