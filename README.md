# Formula Detection

Python tooling to detect formulaic language use in historic documents. 


# Formula Detection Usage

The main class in the [formula_detection](https://pypi.org/project/formula_detection/) package is `FormulaSearch`. This instantiates a searcher that iterates over a list of tokenized texts and identifies candidate formulaic phrases.

It builds up frequency lists of unigram and bigram tokens (with skips) and uses these to efficiently filter the search space.

## Phrases, tokens and variable terms

Phrases are sequences of tokens, which can be contiguous (the tokens appear in a continuous sequence in the source text), but can also contain variable elements (where variable tokens can represent any surface term).

## Input texts

The `FormulaSearch` class takes as input a set of tokenized texts. In the simplest form, these are lists of tokenized text strings -- that is, a list of a list of token-level strings. For this, you can use any tokenizer to preprocess the texts.

Because this package builds on the [fuzzy-search](https://pypi.org/project/fuzzy-search/) pacakge, they can also be texts tokenized by a `fuzzy-search` tokenizer (in that case, the input is a list of `Document` elements or a list of lists of `Token` elements).

## Streaming text files as inputs

If you have a large amount of text, it can be more (memory) efficient to read the input texts directly from one or more files. This is possible because the `FormulaSearch` class takes any Iterable as input, as long as it iterates over lists of tokens.

This notebook shows how to create a simple `Iterable` class that allows you to stream the text from one or more files, apply a tokenizer and pass the resulting lists of tokens to `FormulaSearch` document iterator.



# Passing Texts as Tokenized Strings

We start with the simplest form: texts as lists of tokens.



To demonstrate the need for tokenization, we pass a single, untokenized sentence string to the searcher. Because the `FormulaSearch` document iterator iterates over the tokens in each document, and due to nature of Python strings, in this case, it iterates over each character in the sentence and treats it as a token.


```python
from formula_detection.search import FormulaSearch

texts = ["Yeah, well, you know, that's just like, uh, your opinion, man."]

formula_search = FormulaSearch(texts)
```

    1. Iterating over sentences to calculate term frequencies
        full collection size (tokens): 62
        full lexicon size (types): 22
        minimum term frequency: 1
        minimum frequency lexicon size: 22
    WARNING: No value passed for min_cooc_freq, skipping co-occurrence calculations.


The `FormulaSearch` instance indexes each token (mapping it to a numeric identifier for efficiency) and keeps track of the full vocabulary and the frequency of each token type in the vocabulary.

We can see the result by inspecting the term-to-identifier dictionary of the full vocabulary:


```python
formula_search.full_vocab.term_id
```




    {'Y': 0,
     'e': 1,
     'a': 2,
     'h': 3,
     ',': 4,
     ' ': 5,
     'w': 6,
     'l': 7,
     'y': 8,
     'o': 9,
     'u': 10,
     'k': 11,
     'n': 12,
     't': 13,
     "'": 14,
     's': 15,
     'j': 16,
     'i': 17,
     'r': 18,
     'p': 19,
     'm': 20,
     '.': 21}



If we instead tokenize the sentence by splitting on whitespace, the index consists of a list of words (with puncuation attached, because we using a simplistic tokenizer):


```python
tokenized_texts = [text.split(' ') for text in texts]

formula_search = FormulaSearch(tokenized_texts)

formula_search.full_vocab.term_id
```

    1. Iterating over sentences to calculate term frequencies
        full collection size (tokens): 11
        full lexicon size (types): 11
        minimum term frequency: 1
        minimum frequency lexicon size: 11
    WARNING: No value passed for min_cooc_freq, skipping co-occurrence calculations.





    {'Yeah,': 0,
     'well,': 1,
     'you': 2,
     'know,': 3,
     "that's": 4,
     'just': 5,
     'like,': 6,
     'uh,': 7,
     'your': 8,
     'opinion,': 9,
     'man.': 10}



As you can see, the tokenized terms are indexed as is, case-sensitive and including punctuation. You can use any tokenizer you want to process the texts as you see fit.

## Streaming Texts from Disk

In cases where you have large amounts of text that you can't or don't want to completely read in memory, you can create a Python `Iterable` that can stream the texts from desk while the formula searcher is iterating over the texts. A good explanation and example is given in the [Gensim documentation](https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-streaming-one-document-at-a-time). 

You can pass the filenames of one or more text files to a class that can read their contents and has an `__iter__` method that yields tokenized documents. 


```python
import os

class TextIterable:

    def __init__(self, temp_files):
        self.temp_files = temp_files if isinstance(temp_files, list) else [temp_files]

    def __iter__(self):
        for temp_file in self.temp_files:
            with open(temp_file, 'rt') as fh:
                for line in fh:
                    tokens = line.strip('\n').split(' ')
                    yield tokens


# First, write the texts (containing only the example sentence) to file
temp_file = 'temp.txt'
with open(temp_file, 'wt') as fh:
    for text in texts:
        fh.write(f"{text}\n")

# Create an iterable and pass the text filename
text_iter = TextIterable(temp_file)

# iterate over the documents in the file and pass each document as a list of tokens
for tokens in text_iter:
    print(tokens)


```

    ['Yeah,', 'well,', 'you', 'know,', "that's", 'just', 'like,', 'uh,', 'your', 'opinion,', 'man.']


Now you can pass the iterable to the formula searcher:


```python
formula_search = FormulaSearch(text_iter)
```

    1. Iterating over sentences to calculate term frequencies
        full collection size (tokens): 11
        full lexicon size (types): 11
        minimum term frequency: 1
        minimum frequency lexicon size: 11
    WARNING: No value passed for min_cooc_freq, skipping co-occurrence calculations.


Because we've created a temporary file, it's good to clean up after ourselves


```python
os.unlink(temp_file)
```

## A more serious example

Let's use a longer text with some repetition to see the formula searcher in action. The code below downloads a few texts from the excellent [Project Gutenberg](https://gutenberg.org/).


```python
import requests

text_files = [
    {
        'name': 'Church of Scotland. General Assembly', 
        'url': 'https://www.gutenberg.org/cache/epub/28957/pg28957.txt',
        'file': 'church_acts.txt'
    },
    {
        'name': 'The Bible, Douay-Rheims, Complete', 
        'url': 'https://www.gutenberg.org/cache/epub/1581/pg1581.txt',
        'file': 'bible.txt'
    },
    {
        'name': 'The Gilgamesh Epic', 
        'url': 'https://www.gutenberg.org/cache/epub/11000/pg11000.txt',
        'file': 'gilgamesh.txt'
    },
]


for text_file in text_files:
    if os.path.exists(text_file['file']) is False:
        response = requests.get(text_file['url'])
        if response.status_code == 200:
            with open(text_file['file'], 'wt') as fh:
                fh.write(response.text)

```


```python
name = 'Abridgment of the Debates of Congress, from 1789 to 1856, Vol. 1-4 (of 16)'
text_urls = [
    'https://www.gutenberg.org/cache/epub/40499/pg40499.txt',
    'https://www.gutenberg.org/cache/epub/40851/pg40851.txt',
    'https://www.gutenberg.org/cache/epub/54345/pg54345.txt',
    'https://www.gutenberg.org/cache/epub/47289/pg47289.txt',
]

for ui, url in enumerate(text_urls):
    text_file = f"debates_of_congress-vol_{ui+1}.txt"
    if os.path.exists(text_file) is False:
        response = requests.get(url)
        if response.status_code == 200:
            with open(text_file, 'wt') as fh:
                fh.write(response.text)
    
    

```


```python
import glob

text_files = glob.glob('debates_of_congress-vol_*.txt')
text_files
```




    ['debates_of_congress-vol_1.txt',
     'debates_of_congress-vol_3.txt',
     'debates_of_congress-vol_2.txt',
     'debates_of_congress-vol_4.txt']




```python
text_iter = TextIterable(text_files)
docs = [tokens for tokens in text_iter]
print(f"number of documents: {len(docs):,}\nnumber of tokens: {sum([len(doc) for doc in docs]):,}")
```

    number of documents: 325,831
    number of tokens: 3,465,969


These four volumes together contain 3.5 million words across 325,831 lines.

If we pass these to the formula searcher, we get a larger vocabulary and term frequency index:


```python
formula_search = FormulaSearch(text_iter)
```

    1. Iterating over sentences to calculate term frequencies
        full collection size (tokens): 3,465,969
        full lexicon size (types): 76,530
        minimum term frequency: 1
        minimum frequency lexicon size: 76,530
    WARNING: No value passed for min_cooc_freq, skipping co-occurrence calculations.


The vocabulary consists of 76,530 distinct terms (token types), and of course, the least frequency types occur only once. For finding formulaic phrases, we're interested in words combinations that occur frequently, so low-frequency terms can safely be removed, both to make the term frequency and co-occurrence frequency indexes smaller and to make the search space smaller. With the `min_term_freq` parameter you can control the minimum frequency for including terms in the vocabulary. 


```python
formula_search = FormulaSearch(text_iter, min_term_freq=10)
```

    1. Iterating over sentences to calculate term frequencies
        full collection size (tokens): 3,465,969
        full lexicon size (types): 76,530
        minimum term frequency: 10
        minimum frequency lexicon size: 15,196
    WARNING: No value passed for min_cooc_freq, skipping co-occurrence calculations.


The full lexicon (`full_vocab`) contains 76,530 distinct terms while the minimum frequency lexicon (`min_freq_vocab`) contains only 15,196 terms. Only the terms in the minimum frequency lexicon are considered as elements of formulaic phrases.

You can check the terms contained in the frequency index (a Python `Counter` object) by mapping the term IDs to their respective terms.


```python
for term_id, freq in formula_search.term_freq.most_common(10):
    print(formula_search.id2term(term_id), freq)
```

     385093
    the 244216
    of 146227
    to 113052
    and 73445
    a 50936
    in 49720
    that 45814
    be 37605
    it 29161


The most common terms tend to be stopwords, which do not get removed from the vocabulary, because they are part of formulaic phrases.

As you can see, the most common term is the empty string (''). This simplistic tokenizer has a few issues that are can be solved by using a more sophisicated tokenizer (see below).

Next you can calculate the co-occurrence frequency of a term and it's neighbouring terms using the `skip_size` parameter (which defaults to `skip_size=4`), meaning, the co-occurrence of the term and up to `skip_size+1` neighbouring terms (only if both terms are in the minimum frequency lexicon).


```python
formula_search.calculate_co_occurrence_frequencies(skip_size=4)
```

    2. Iterating over sentences to calculate the co-occurrence frequencies
    docs: 325,831	num_words: 3,465,969	num_coocs: 12,891,052	num distinct coocs: 2,938,969
        co-occurrence index size: 2,938,969


The co-occurrence index contains almost 3 million bigrams. Again, the most common ones tend to be combinations of stopwords:


```python
for term_ids, freq in formula_search.cooc_freq.most_common(10):
    terms = [formula_search.id2term(term_id) for term_id in term_ids]
    print(terms, freq)
```

    ['', ''] 850403
    ['the', 'of'] 77174
    ['of', 'the'] 72750
    ['', 'the'] 59062
    ['the', 'the'] 54084
    ['to', 'the'] 48665
    ['', 'of'] 40906
    ['', 'to'] 25126
    ['in', 'the'] 24017
    ['and', 'the'] 22086


Finally, we get to the point where we can ask the formula searcher to find candidate formulaic phrases using `extract_phrases`:


```python
from collections import Counter

extractor = (formula_search
             .extract_phrases(phrase_type='sub_phrases', max_phrase_length=5, min_cooc_freq=5))

for ti, candidate_pm in enumerate(extractor):
    print('phrase:', candidate_pm.phrase)
    if (ti+1) == 10:
        break

```

    Minimum co-occurrence frequency: 5
    phrase: Project Gutenberg eBook of Abridgment
    phrase: Gutenberg eBook of Abridgment of
    phrase: eBook of Abridgment of the
    phrase: of Abridgment of the Debates
    phrase: Abridgment of the Debates of
    phrase: of the Debates of Congress,
    phrase: the Debates of Congress, from
    phrase: Debates of Congress, from 1789
    phrase: of Congress, from 1789 to
    phrase: Congress, from 1789 to 1856,


Here you see phrases from the standard Project Gutenberg preamble. This is not particularly interesting. You can get more insight in the most common phrases by counting their frequency:


```python
extractor = (formula_search
             .extract_phrases(phrase_type='sub_phrases', max_phrase_length=5, min_cooc_freq=10))

sub_phrase_freq = Counter()
for ti, candidate_pm in enumerate(extractor):
    sub_phrase_freq.update([candidate_pm.phrase])
```

    Minimum co-occurrence frequency: 10



```python
sub_phrase_freq.most_common(20)
```




    [('    ', 103719),
     ('    the', 3012),
     ('    of', 1521),
     ('    and', 1014),
     ('    to', 962),
     ('    on', 710),
     ('   of the', 540),
     ('the President of the United', 512),
     ('    The', 498),
     ('    in', 487),
     ('  |  ', 457),
     (' |   ', 431),
     ('   | ', 428),
     ('the PRESIDENT OF THE UNITED', 370),
     ('    that', 351),
     ('into a Committee of the', 350),
     ('President of the United States', 343),
     ('    it', 337),
     ('Committee of the Whole on', 321),
     ('   on the', 314)]



The majority of frequent phrases consist of one or more empty strings. It's time to improve the tokenizer, so you get a better idea of what candidate formulaic phrases can be identified with a more informed tokenizer.

## Tokenizers and the impact of normalisation

We've seen a number of issues when using a simplistic tokenizer:

1. The tokenized strings contain many empty strings ('') as tokens.
2. Many tokens have puncuation attached to words, which makes them distinct terms (token types) from the same words without punctuation.
3. The tokenizer is case-sensitive, which may be a hurdle if formulaic phrases have (unhelpful) variation in the use of case.

A slightly more sophisticated tokenizer is provided by the [fuzzy-search]() package, which is one of the dependencies of the `formula_detection` package, so is already installed if you've used a package manager to install `formula_detection`. 


```python
from fuzzy_search.tokenization.token import Tokenizer

# ignorecase transforms everything to lowercase
# remove_punctuation uses re.split(r'\W+', token) to strip non-alphanumeric chars
tokenizer = Tokenizer(ignorecase=True, remove_punctuation=True)

with open(text_files[0], 'rt') as fh:
    lines = [line for line in fh]
    for line in lines[:5]:
        tokens = tokenizer.tokenize(line)
        print(line)
        print(tokens)
        print()
```

    ﻿The Project Gutenberg eBook of Abridgment of the Debates of Congress, from 1789 to 1856, Vol. 1 (of 16)
    
    ['the', 'project', 'gutenberg', 'ebook', 'of', 'abridgment', 'of', 'the', 'debates', 'of', 'congress', 'from', '1789', 'to', '1856', 'vol', '1', 'of', '16']
    
        
    
    []
    
    This ebook is for the use of anyone anywhere in the United States and
    
    ['this', 'ebook', 'is', 'for', 'the', 'use', 'of', 'anyone', 'anywhere', 'in', 'the', 'united', 'states', 'and']
    
    most other parts of the world at no cost and with almost no restrictions
    
    ['most', 'other', 'parts', 'of', 'the', 'world', 'at', 'no', 'cost', 'and', 'with', 'almost', 'no', 'restrictions']
    
    whatsoever. You may copy it, give it away or re-use it under the terms
    
    ['whatsoever', 'you', 'may', 'copy', 'it', 'give', 'it', 'away', 'or', 're', 'use', 'it', 'under', 'the', 'terms']
    


We add one more improvement to the text iteratable. The Gutenberg Project files use a maximum line width for ease of reading and break sentences over these line widths. The boundaries between paragraphs are indicated by a double newline. 

So by treating each line as a document, the words in two consecutive lines may belong to the same sentence and paragraph, but are treated as belonging to different documents. In other words, the line as document treatment misses many term co-occurrences.

We update the text iterable by first generating paragraphs from the lines, then tokenizing the paragraphs. 

(**Note** if you care about not crossing sentence boundaries, you could of course use a sentence tokenizer before using a word tokenizer.)


```python
from demo_helper import lines_to_paras


class TokenizedTextIterable:

    def __init__(self, text_files, tokenizer = None):
        self.text_files = text_files if isinstance(text_files, list) else [text_files]
        self.tokenize = tokenizer.tokenize if tokenizer is not None else lambda line: line.strip('\n').split(' ')

    def __iter__(self):
        for text_file in self.text_files:
            with open(text_file, 'rt') as fh:
                lines = [line for line in fh]
                for para in lines_to_paras(lines):
                    yield self.tokenize(para)
            
    
text_iter = TokenizedTextIterable(text_files, tokenizer)
#text_iter = TokenizedTextIterable('gutenberg_intro.txt', tokenizer)
#text_iter = TokenizedTextIterable('gilgamesh.txt', tokenizer)

for tokens in text_iter:
    print(tokens)
    break
```

    ['the', 'project', 'gutenberg', 'ebook', 'of', 'abridgment', 'of', 'the', 'debates', 'of', 'congress', 'from', '1789', 'to', '1856', 'vol', '1', 'of', '16', 'this', 'ebook', 'is', 'for', 'the', 'use', 'of', 'anyone', 'anywhere', 'in', 'the', 'united', 'states', 'and', 'most', 'other', 'parts', 'of', 'the', 'world', 'at', 'no', 'cost', 'and', 'with', 'almost', 'no', 'restrictions', 'whatsoever', 'you', 'may', 'copy', 'it', 'give', 'it', 'away', 'or', 're', 'use', 'it', 'under', 'the', 'terms', 'of', 'the', 'project', 'gutenberg', 'license', 'included', 'with', 'this', 'ebook', 'or', 'online', 'at', 'www', 'gutenberg', 'org', 'if', 'you', 'are', 'not', 'located', 'in', 'the', 'united', 'states', 'you', 'will', 'have', 'to', 'check', 'the', 'laws', 'of', 'the', 'country', 'where', 'you', 'are', 'located', 'before', 'using', 'this', 'ebook']


Now the first document corresponds to the first paragraph in the file. 


```python
formula_search = FormulaSearch(text_iter, min_term_freq=10, min_cooc_freq=10)
```

    1. Iterating over sentences to calculate term frequencies
        full collection size (tokens): 3,102,812
        full lexicon size (types): 26,514
        minimum term frequency: 10
        minimum frequency lexicon size: 9,862
    2. Iterating over sentences to calculate the co-occurrence frequencies
    docs: 39,245	num_words: 3,102,812	num_coocs: 14,566,150	num distinct coocs: 2,227,284
        co-occurrence index size: 2,227,284


The removal of punctuation and lowercasing of all tokens has resulted in a much smaller full lexicon (26,514 instead of 76,530 terms) and also a smaller minimum frequency lexicon (9,862 instead of 15,196 terms). 

However, by concatenating lines to paragraphs, the total number of bigram co-occurrences has gone up from 12,891,052 to 14,566,150. But because of the smaller, normalised lexicon, the distinct number of co-occurring bigrams dropped from 2,938,969 to 2,227,284.

Next, we can extract candidate formulaic phrases again:


```python
extractor = (formula_search
             .extract_phrases(phrase_type='sub_phrases', max_phrase_length=5, min_cooc_freq=10))

sub_phrase_freq = Counter()
for ti, candidate_pm in enumerate(extractor):
    sub_phrase_freq.update([candidate_pm.phrase])
```

    Minimum co-occurrence frequency: 10



```python
sub_phrase_freq.most_common(20)
```




    [('president of the united states', 1572),
     ('the president of the united', 1325),
     ('of the united states and', 729),
     ('a committee of the whole', 698),
     ('committee of the whole on', 464),
     ('into a committee of the', 420),
     ('of the whole on the', 391),
     ('of the united states to', 386),
     ('on the part of the', 343),
     ('itself into a committee of', 317),
     ('of the house of representatives', 311),
     ('resolved itself into a committee', 304),
     ('the secretary of the treasury', 274),
     ('the committee of the whole', 272),
     ('the report of the committee', 256),
     ('committee of the whole house', 254),
     ('government of the united states', 252),
     ('on the president of the', 250),
     ('constitution of the united states', 246),
     ('the constitution of the united', 238)]



## Allowing variable terms

So far, we've only extracted continuous phrases (only terms that are directly adjacent to each other), but formulaic phrases can have contain variable elements. These are typically the names of persons, locations, dates, or words that optional and only occur in a fraction of the uses of a formulaic phrase. The use of skips allows the searcher to identify frequently occurring phrases with such variable elements. To show that a phrase contains a variable element, the variable token is replaced by `<VAR>`.


```python
extractor = (formula_search
             .extract_phrases(phrase_type='sub_phrases', max_phrase_length=8,
                              max_variables=2, min_cooc_freq=10))

sub_phrase_freq = Counter()
for ti, candidate_pm in enumerate(extractor):
    sub_phrase_freq.update([candidate_pm.phrase])
```

    Minimum co-occurrence frequency: 10



```python
for phrase, freq in sub_phrase_freq.most_common(5000):
    if '<VAR>' not in phrase:
        continue
    print(phrase, freq)
```

    on the execution of the british treaty <VAR> 16
    rights of the house relative to treaties <VAR> 16
    <VAR> appointed a senator by the legislature of 15
    on an additional military force <VAR> _see index_ 9
    of the house relative to treaties <VAR> on 8
    the house relative to treaties <VAR> on the 8
    <VAR> not protected by u s copyright law 8
    prosecution on the trial of judge chase <VAR> 8
    defence on the trial of judge chase <VAR> 8
    on jurisdiction over the district of columbia <VAR> 8
    form a more perfect union establish justice <VAR> 7
    a more perfect union establish justice <VAR> domestic 7
    more perfect union establish justice <VAR> domestic tranquillity 7
    perfect union establish justice <VAR> domestic tranquillity provide 7
    union establish justice <VAR> domestic tranquillity provide for 7
    establish justice <VAR> domestic tranquillity provide for the 7
    justice <VAR> domestic tranquillity provide for the common 7
    <VAR> domestic tranquillity provide for the common defence 7
    <VAR> on the execution of the british treaty 7
    commercial intercourse with france and great britain <VAR> 7
    house relative to treaties <VAR> on the execution 6
    relative to treaties <VAR> on the execution of 6
    to treaties <VAR> on the execution of the 6
    treaties <VAR> on the execution of the british 6
    officers and crew of the united states <VAR> 6
    to such english authorities as they believed <VAR> 6
    inquiry into the conduct of gen wilkinson <VAR> 6
    <VAR> appointed a senator by the state of 6
    on exempting bank notes from stamp duty <VAR> 6
    non intercourse with great britain and france <VAR> 6
    the town of <VAR> in the state of 5
    zephaniah swift silas <VAR> george thatcher uriah tracy 5
    <VAR> on the right to indian lands within 5
    the execution of the british treaty <VAR> on 5
    execution of the british treaty <VAR> on the 5
    <VAR> on a salary for members of congress 5
    and crew of the united states <VAR> <VAR> 5
    the bill <VAR> from postage all letters and 5
    bill <VAR> from postage all letters and packets 5
    <VAR> from postage all letters and packets to 5
    such english authorities as they believed <VAR> or 5
    english authorities as they believed <VAR> or from 5
    authorities as they believed <VAR> or from citing 5
    as they believed <VAR> or from citing certain 5
    they believed <VAR> or from citing certain statutes 5
    believed <VAR> or from citing certain statutes of 5
    <VAR> or from citing certain statutes of the 5
    a dam or <VAR> from mason s island 5
    dam or <VAR> from mason s island to 5
    or <VAR> from mason s island to the 5
    <VAR> from mason s island to the western 5
    president of the united states before he <VAR> 5
    of the united states before he <VAR> on 5
    the united states before he <VAR> on the 5
    united states before he <VAR> on the execution 5
    states before he <VAR> on the execution of 5
    before he <VAR> on the execution of his 5
    he <VAR> on the execution of his office 5
    <VAR> on the execution of his office on 5



```python

```
