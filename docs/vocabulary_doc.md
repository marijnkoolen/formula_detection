## Detailed Docstring Explanations:

`token_to_string` **function**:
- Converts a term (either a string or a Token object) to its string representation. This function handles both cases where the input is a raw string or a Token.

`Vocabulary` **class**:
- Manages a collection of terms, providing functionality for indexing terms and retrieving term IDs and their string representations.
It includes methods for adding, indexing, and retrieving terms by their ID, as well as resetting the vocabulary index.

`make_selected_vocab` **function**:
- Creates a new vocabulary containing a selected subset of terms from the full vocabulary. The selection can be based on specific terms, term IDs, or term frequencies (with an optional frequency threshold).

`calculate_term_freq` **function**:
- Iterates through a collection of documents, calculates the frequency of each term, and returns the frequencies as a Counter object, which maps term IDs to their respective counts.
These docstrings help explain the purpose, arguments, and return values of each function and class, offering clarity on how the components of the code work together.