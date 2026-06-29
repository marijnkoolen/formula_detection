"""
phrase_typing_ui.py — ipywidgets UI for the human-in-the-loop phrase
typing workflow in phrase_typing.py.

Kept in a separate module from phrase_typing.py so that the core
registry/suggester/concordance logic has no dependency on ipywidgets or
IPython and can be used (and tested) headlessly; only this UI layer
requires a notebook environment.

Usage in a notebook::

    from formula_detection.phrase_typing import PhraseTypeRegistry, TypeSuggester
    from formula_detection.phrase_typing_ui import PhraseTypingApp

    registry = PhraseTypeRegistry()
    suggester = TypeSuggester(phrase_positions, flat_stream)
    app = PhraseTypingApp(registry, suggester, candidate_phrases, phrase_freq)
    app.display()
"""
from typing import Dict, List, Optional, Sequence

import ipywidgets as widgets
from IPython.display import display

from formula_detection.phrase_typing import PhraseTypeRegistry, TypeSuggester, concordance


class PhraseTypingApp:
    """Interactive widget for browsing, assigning, and suggesting phrase types.

    Three sections, each independently usable:

    1. **Browse & assign** — pick a phrase, read concordance examples, and
       assign it to one or more types (existing or new).
    2. **Suggestions** — pick an existing type and a window size, and get
       a ranked list of still-uncategorised phrases with quick-assign
       buttons, each with its own mini-concordance preview.
    3. **Registry overview** — a live summary of all types and their members.

    Args:
        registry: The ``PhraseTypeRegistry`` to read from and write to.
        suggester: A ``TypeSuggester`` built over the same phrase_positions
            and flat token stream the candidate phrases come from.
        candidate_phrases: The full pool of phrases available to browse,
            assign, and suggest from.
        phrase_freq: Optional Dict mapping phrase to a frequency count,
            used only to order the browse dropdown (most frequent first)
            and display it alongside each phrase.
        concordance_window: Tokens of context to show on each side in
            concordance views.
    """

    def __init__(self, registry: PhraseTypeRegistry, suggester: TypeSuggester,
                 candidate_phrases: Sequence[str],
                 phrase_freq: Optional[Dict[str, int]] = None,
                 concordance_window: int = 10):
        self.registry = registry
        self.suggester = suggester
        self.candidate_phrases = list(candidate_phrases)
        self.phrase_freq = phrase_freq or {}
        self.concordance_window = concordance_window

        self._build_browse_section()
        self._build_suggestion_section()
        self._build_overview_section()

        self.widget = widgets.Tab(children=[
            self.browse_box, self.suggestion_box, self.overview_box,
        ])
        self.widget.set_title(0, 'Browse & assign')
        self.widget.set_title(1, 'Suggestions')
        self.widget.set_title(2, 'Registry overview')

    def display(self) -> None:
        """Render the app. Call this in a notebook cell."""
        display(self.widget)

    # ------------------------------------------------------------------
    # Section 1: Browse & assign
    # ------------------------------------------------------------------

    def _phrase_label(self, phrase: str) -> str:
        freq = self.phrase_freq.get(phrase)
        types = self.registry.types_of(phrase)
        freq_str = f' ({freq})' if freq is not None else ''
        types_str = f' [{", ".join(types)}]' if types else ' [uncategorised]'
        return f'{phrase}{freq_str}{types_str}'

    def _sorted_phrases(self) -> List[str]:
        if self.phrase_freq:
            return sorted(self.candidate_phrases, key=lambda p: -self.phrase_freq.get(p, 0))
        return list(self.candidate_phrases)

    def _build_browse_section(self) -> None:
        phrases = self._sorted_phrases()
        self.browse_dropdown = widgets.Dropdown(
            options=[(self._phrase_label(p), p) for p in phrases],
            description='Phrase:',
            layout=widgets.Layout(width='600px'),
        )
        self.browse_concordance_out = widgets.Output()
        self.browse_existing_types = widgets.SelectMultiple(
            options=sorted(self.registry.types.keys()),
            description='Existing types:',
            layout=widgets.Layout(width='400px'),
        )
        self.browse_new_type = widgets.Text(
            description='New type:', placeholder='e.g. opening formula',
        )
        self.browse_assign_button = widgets.Button(description='Assign', button_style='success')
        self.browse_status_out = widgets.Output()

        self.browse_dropdown.observe(self._on_browse_phrase_change, names='value')
        self.browse_assign_button.on_click(self._on_browse_assign_click)

        self.browse_box = widgets.VBox([
            self.browse_dropdown,
            self.browse_concordance_out,
            widgets.HBox([self.browse_existing_types, self.browse_new_type]),
            self.browse_assign_button,
            self.browse_status_out,
        ])
        self._refresh_browse_concordance()

    def _refresh_browse_concordance(self) -> None:
        self.browse_concordance_out.clear_output()
        phrase = self.browse_dropdown.value
        if phrase is None:
            return
        with self.browse_concordance_out:
            lines = concordance(
                phrase, self.suggester.flat_stream,
                phrase_positions=self.suggester.phrase_positions,
                window=self.concordance_window, max_examples=8,
            )
            print(f'{len(self.suggester.phrase_positions.get(phrase, []))} total occurrences; '
                  f'showing {len(lines)} examples spread across the corpus:\n')
            for line in lines:
                print(' ', line)

    def _on_browse_phrase_change(self, change) -> None:
        self._refresh_browse_concordance()

    def _on_browse_assign_click(self, _button) -> None:
        phrase = self.browse_dropdown.value
        type_names = list(self.browse_existing_types.value)
        new_type = self.browse_new_type.value.strip()
        if new_type:
            type_names.append(new_type)
        self.browse_status_out.clear_output()
        with self.browse_status_out:
            if not type_names:
                print('Select at least one existing type or enter a new type name.')
                return
            self.registry.assign(phrase, type_names)
            print(f'Assigned {phrase!r} to: {type_names}')
        self._refresh_all_type_choices()
        self._refresh_browse_concordance()  # updates the [types] label via dropdown rebuild

    # ------------------------------------------------------------------
    # Section 2: Suggestions
    # ------------------------------------------------------------------

    def _build_suggestion_section(self) -> None:
        self.suggest_type_dropdown = widgets.Dropdown(
            options=sorted(self.registry.types.keys()), description='Type:',
        )
        self.suggest_window = widgets.FloatText(value=100, description='Window:')
        self.suggest_top_n = widgets.IntSlider(value=10, min=1, max=50, description='Top N:')
        self.suggest_button = widgets.Button(description='Get suggestions', button_style='info')
        self.suggest_results_box = widgets.VBox([])

        self.suggest_button.on_click(self._on_suggest_click)

        self.suggestion_box = widgets.VBox([
            widgets.HBox([self.suggest_type_dropdown, self.suggest_window, self.suggest_top_n]),
            self.suggest_button,
            self.suggest_results_box,
        ])

    def _on_suggest_click(self, _button) -> None:
        type_name = self.suggest_type_dropdown.value
        if type_name is None or type_name not in self.registry.types:
            self.suggest_results_box.children = [widgets.HTML('<i>No type selected.</i>')]
            return
        members = self.registry.types[type_name].members
        uncategorised = self.registry.uncategorised(self.candidate_phrases)
        results = self.suggester.suggest_for_type(
            members, uncategorised,
            window=self.suggest_window.value, top_n=self.suggest_top_n.value,
        )
        rows = []
        for phrase, affinity, similarity in results:
            rows.append(self._make_suggestion_row(phrase, affinity, similarity, type_name))
        if not rows:
            rows = [widgets.HTML('<i>No suggestions (no uncategorised candidates, or none nearby).</i>')]
        self.suggest_results_box.children = rows

    def _make_suggestion_row(self, phrase: str, affinity: float, similarity: float,
                             type_name: str) -> widgets.Widget:
        label = widgets.HTML(
            f'<b>{phrase}</b> &mdash; affinity={affinity:.3f}, context_sim={similarity:.3f}',
            layout=widgets.Layout(width='420px'),
        )
        preview_lines = concordance(
            phrase, self.suggester.flat_stream,
            phrase_positions=self.suggester.phrase_positions,
            window=self.concordance_window, max_examples=2,
        )
        preview = widgets.HTML(
            '<br>'.join(f'<small>{line}</small>' for line in preview_lines),
            layout=widgets.Layout(width='600px'),
        )
        assign_button = widgets.Button(description=f'Assign to {type_name}', button_style='success',
                                       layout=widgets.Layout(width='160px'))

        def _on_click(_button, phrase=phrase, type_name=type_name):
            self.registry.assign(phrase, type_name)
            assign_button.description = 'Assigned'
            assign_button.disabled = True
            self._refresh_all_type_choices()

        assign_button.on_click(_on_click)
        return widgets.VBox([widgets.HBox([label, assign_button]), preview])

    # ------------------------------------------------------------------
    # Section 3: Registry overview
    # ------------------------------------------------------------------

    def _build_overview_section(self) -> None:
        self.overview_out = widgets.Output()
        self.overview_refresh_button = widgets.Button(description='Refresh')
        self.overview_refresh_button.on_click(lambda _b: self._refresh_overview())
        self.overview_box = widgets.VBox([self.overview_refresh_button, self.overview_out])
        self._refresh_overview()

    def _refresh_overview(self) -> None:
        self.overview_out.clear_output()
        with self.overview_out:
            categorised = self.registry.all_categorised()
            uncategorised = self.registry.uncategorised(self.candidate_phrases)
            print(f'{len(self.registry.types)} types, '
                  f'{len(categorised)} categorised, {len(uncategorised)} uncategorised\n')
            print(self.registry.summary() or '(no types yet)')

    # ------------------------------------------------------------------
    # Shared refresh
    # ------------------------------------------------------------------

    def _refresh_all_type_choices(self) -> None:
        """Refresh dropdowns/selectors that list type names or phrase labels."""
        type_names = sorted(self.registry.types.keys())
        self.browse_existing_types.options = type_names
        self.suggest_type_dropdown.options = type_names
        current = self.browse_dropdown.value
        phrases = self._sorted_phrases()
        self.browse_dropdown.options = [(self._phrase_label(p), p) for p in phrases]
        self.browse_dropdown.value = current
        self._refresh_overview()
