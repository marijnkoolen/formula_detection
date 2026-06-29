# Document Pattern Detection: Design Notes and Findings

This document tracks what we've learned while extending `formula_detection`
with three related capabilities beyond the original frequency-based phrase
search:

1. **Orthographic variant detection** (`variant_index.py`)
2. **Semantic clustering of formulaic phrases** (`phrase_vectorizer.py`, `clustering.py`)
3. **Phrase co-occurrence patterns and document/element boundary detection**
   (`phrase_cooccurrence.py`, `motif.py`, `window_estimator.py`, `boundary.py`)

The goal throughout has been to support corpora where document boundaries
are *not* known in advance — large streams of running text (e.g. scanned
books containing many resolutions, deeds, or letters) where formulaic
phrase patterns themselves are the evidence for where one document or
document element ends and the next begins.

The notes below record design decisions, what worked, what didn't, and why
— particularly the gap between clean synthetic test cases and a real,
messy corpus.

## Module overview

| Module | Purpose |
|---|---|
| `variant_index.py` | `VariantIndex` — three-pass, language-agnostic cascade mapping surface word forms to canonical forms (rule-based normaliser, edit-distance grouping, context-anchored grouping) |
| `phrase_vectorizer.py` | `PhraseVectorizer` — tf-idf context-word feature vectors for candidate phrases, optionally restricted to a function-word set |
| `clustering.py` | `cluster_phrases()` — Ward agglomerative clustering over phrase vectors |
| `phrase_cooccurrence.py` | `PhraseCooccurrenceIndex` — multi-scale pairwise phrase co-occurrence (LLR-scored) from `phrase_positions`, single pass over a token stream |
| `motif.py` | `PhraseMotifIndex` — ordered, multi-phrase ("motif") co-occurrence mining; `classify_relation()` for nested/sequential/overlapping relationships between motifs |
| `window_estimator.py` | `WindowEstimator` — data-driven window-size recommendation from gap-distribution knee detection |
| `boundary.py` | `BoundaryDetector` — scores token positions as document/element boundary candidates, from either community-transition evidence (`score_stream`) or mined motif instances (`score_from_motifs`) |
| `phrase_typing.py` | `PhraseTypeRegistry`, `TypeSuggester`, `concordance()` — human-in-the-loop, multi-label categorisation of phrases into user-defined types, with co-occurrence-based suggestions for uncategorised phrases |
| `phrase_typing_ui.py` | `PhraseTypingApp` — ipywidgets UI over `phrase_typing.py` (browse/concordance, suggestions, registry overview) |

`window_estimator.py` also exposes `WindowEstimator.group_by_scale`, which groups a list of phrases by the characteristic scale of their own recurrence (median gap between successive occurrences), useful for separating phrases that operate at different levels of nested document structure (e.g. day-level vs. resolution-level vs. sub-decision-level formulas) before or alongside manual typing.

All of this is built on top of the existing `FormulaSearch` phrase
extraction pipeline and operates on `phrase_positions` (per-phrase sorted
token offsets) rather than rescanning the raw token stream wherever
possible.

## What worked well on clean synthetic corpora

Synthetic corpora with 2–3 sharply distinct document types, well-separated
by filler text, validated the core mechanics end to end:

- **Community detection** (`score_stream` + Louvain on the phrase
  co-occurrence graph) correctly separated phrases belonging to different
  document types into distinct communities, *provided the co-occurrence
  window was smaller than the gap between documents* — window size
  relative to document spacing is the single most important tuning
  parameter for this approach.
- **Motif mining** (`PhraseMotifIndex`) recovered all document boundaries
  exactly (60/60, including both start and end) when wired into
  `BoundaryDetector.score_from_motifs`, using the canonical/maximal motif
  for each document type. `order_entropy` (near 0 for a strict template)
  and `lift` (very high for a true formula) were both clean, decisive
  signals in this setting.
- **`classify_relation()`** correctly distinguished nested sub-elements,
  sequential halves of the same formula (candidates for merging into one
  larger motif), and genuinely unrelated document types — each producing
  a 100%-dominant relation label with no ambiguity.
- **Automatic window estimation** (`WindowEstimator.cluster_based_window`)
  — clustering the merged stream of all candidate phrases at the knee of
  its own gap distribution — recovered the correct window size
  automatically, matching hand-tuned values, *as long as document lengths
  were fairly uniform*.

## What broke down on a real corpus, and why

Testing against a real corpus of resolution paragraphs (see
`data/resolution_paragraphs/`) — historical Dutch, ~450k tokens, ~2300
resolutions, much of it densely formulaic — surfaced limitations that the
synthetic corpora didn't:

### 1. Automatic window estimation assumes fairly uniform document lengths

`cluster_based_window()` works by finding the knee in the *combined*
candidate-phrase event stream and using it as a single-link clustering
threshold; cluster spans then estimate the window. This assumes a
reasonably clean separation between "within one document" and "between
documents." On the real corpus, paragraph lengths ranged from 1 token to
over 7000 tokens (one extreme outlier), and the formula-internal point
process is not bursty enough relative to natural variance for the knee
detector to find a clean, useful threshold. The estimator should not be
trusted blindly on corpora with high length variance; a manual window
informed by summary statistics (e.g. median document length) is a
reasonable fallback, and `recommended_global_window`'s self-gap approach
has the same problem for a different reason (see below).

### 2. A phrase's own recurrence interval is not its within-unit span

`WindowEstimator.recommended_window()` (the per-phrase, self-gap-based
estimate intended for `window='auto-phrase'`) measures the gap to a
phrase's *own next occurrence*. In a stream where several
document/element types are interleaved, that's the distance to the next
document of the *same type* — not the within-document span you actually
want to bound. On the real corpus this was reliably large and flagged
`multimodal=True`, exactly as the diagnostic is designed to warn. Prefer
`cluster_based_window` for a global estimate; treat per-phrase self-gap
windows as informative mainly when a phrase is reliably type-specific
**and** document types aren't densely interleaved.

### 3. Ranking candidate motifs by lift is the wrong objective for boundary detection

This was the most consequential finding. Lift (observed / expected
co-occurrence under independence) systematically favours **rare,
highly-specific** co-occurring phrase combinations over **common,
reliably-positioned single phrases**. On the real corpus, the
highest-lift motifs were administrative formulas about provincial
deputies and resolution-resumption boilerplate — real, statistically
significant patterns, but poor boundary markers, because they recur
*within* documents about specific topics rather than marking the
start/end of *every* document.

Meanwhile, the single most useful boundary signal — common opening
formulas like *"ontfangen een missive van"* and *"is ter vergaderinge
geleesen de requeste van"* — were correctly surfaced by frequency-based
candidate discovery (`FormulaSearch.extract_phrases`), but got crowded out
once motifs were *ranked by lift* for boundary scoring.

**Empirical comparison on the real corpus:**

| Approach | Resolution-boundary F1 |
|---|---|
| Full pipeline (`FormulaSearch` → dedup → `PhraseMotifIndex` → top-40 motifs by lift → `score_from_motifs`) | ≈ 0.00–0.25 |
| Four manually-identified high-frequency opener phrases, used directly as boundary positions (no motif mining) | **0.57** (P=0.49, R=0.69) |

The takeaway: for boundary detection specifically, candidate-phrase
*selection* should weight **regularity of a phrase's own recurrence**
(does it recur roughly once per document, with low variance?) far more
heavily than co-occurrence lift with other phrases. This is not yet
implemented as a scoring function in the library — `WindowEstimator`
computes the necessary ingredient (self-gap distributions) but doesn't
yet expose an "opener quality" score derived from it. This is a clear,
concrete next step if returning to this problem.

### Bugs found only because of real, messy input

Two genuine pre-existing bugs only surfaced when run against real data
with plain `List[str]` documents and large candidate/score counts —
synthetic test corpora were too small or too clean to trigger them:

- `make_candidate_phrase_match` (`candidate.py`) assumed `doc` was always a
  fuzzy-search `Doc` object and crashed (`AttributeError`) on plain
  token-list documents, despite that being a documented, supported input
  type. Fixed to branch on `isinstance(doc, Doc)`, consistent with how the
  rest of `search.py` already handles this.
- `BoundaryDetector.detect_boundaries` (`boundary.py`) used an O(n·k)
  nested scan for non-maximum suppression, which is fine for the dozens of
  candidates a synthetic corpus produces but hangs in practice once a real
  corpus's motif mining produces tens or hundreds of thousands of
  candidate scores. Fixed with a sorted-insert/bisect approach.
- `count_pre_post_phrase_context` (`context.py`) used
  `hasattr(sent, 'words')` to detect dict-shaped sentences, which is
  always `False` for a plain dict (dicts don't have attributes), causing
  an `UnboundLocalError` any time `PhraseContext` was given dict-style
  sentences. Fixed to check `'words' in sent`, and added support for
  plain token-list sentences too.

## A caveat about the test corpus

The resolution-paragraphs corpus used for this evaluation is **unusually,
extremely formulaic** — fixed opening phrases, highly templated structure
— which makes it a poor corpus for *steering* further development of
these methods: a method that overfits to "just match the one dominant
opening formula" would score deceptively well here without generalizing.
Findings 1–3 above are being kept as documented lessons, but further
tuning decisions should be validated against corpora with different
characteristics (varying degrees of formulaicity, different boundary
marker reliability, different document-length distributions) before being
treated as general guidance.

## Practical recommendations (current state)

- Don't trust `window='auto'` blindly on corpora with high document-length
  variance; sanity-check the resolved window against a known summary
  statistic (e.g. median document length) and fall back to a manual value
  if they disagree wildly.
- For boundary detection, don't rank motifs by lift alone. Consider
  combining lift with a regularity-of-recurrence measure (low variance in
  a phrase's own gap distribution, relative to corpus-wide document
  count) before feeding motifs into `score_from_motifs`.
- Deduplicate near-identical sliding-window n-gram candidates (e.g. via
  substring containment, processed in frequency-descending order) before
  motif mining — otherwise mining produces a large number of redundant,
  highly-overlapping motifs that are expensive to score and don't add
  information.
- `TypeSuggester.suggest_for_type`'s `window` parameter is deliberately
  manual for the same reason `window='auto'` is unreliable elsewhere:
  once a type's confirmed members are individually very frequent, no
  automatic estimator we've tried (including
  `WindowEstimator.cluster_based_window` restricted to just that type's
  members) produces a usable window. Pick a window matching the scale
  you intend to capture for that specific type, and expect very
  different types (day-scale vs. decision-scale) to need very different
  windows.
- Validate any boundary-detection configuration against a corpus that
  isn't extremely formulaic before generalizing conclusions from it.
