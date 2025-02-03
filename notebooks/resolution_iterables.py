import gzip
from typing import Callable, List

from formula_detection.tokenization.tokenizer import Doc
from formula_detection.tokenization.tokenizer import Tokenizer, RegExTokenizer


def generate_para_doc(files, tokenizer: Tokenizer):
    for fname in files:
        opener = gzip.open if fname.endswith('.gz') else open
        with opener(fname, 'rt') as fh:
            for line in fh:
                row = line.strip().split('\t')
                if len(row) == 4:
                    res_id, para_id, sent_num, text = row
                elif len(row) == 3:
                    res_id, para_id, text = row
                else:
                    continue
                para = Doc(text, para_id, tokenizer.tokenize(text), metadata={'resolution_id': res_id})
                yield para


def generate_doc(files, tokenizer: Tokenizer, aggregate_doc: bool = False):
    prev_id = None
    prev_doc = None
    for doc in generate_para_doc(files, tokenizer=tokenizer):
        if aggregate_doc is False:
            yield doc
            continue
        else:
            doc.id = doc.metadata['resolution_id']
        if prev_id is not None and prev_id == doc.id:
            # print(f'MERGING {prev_id} and {doc["doc_id"]}')
            prev_doc = merge_para_docs(prev_doc, doc)
        else:
            if prev_id is not None:
                yield prev_doc
            prev_doc = doc
            prev_id = doc.id
            # print(f'SETTING PREV_ID:', prev_id)


def merge_para_docs(doc1: Doc, doc2: Doc) -> Doc:
    text = '\n'.join([doc1.text, doc2.text])
    return Doc(text=text, doc_id=doc1.id, metadata=doc1.metadata, tokens=doc1.tokens + doc2.tokens)


class ResolutionIterable:

    def __init__(self, files, tokenizer: Tokenizer = None, ignorecase: bool = False, include_boundaries: bool = False,
                 aggregate_doc: bool = False):
        self.files = files
        if not tokenizer:
            tokenizer = RegExTokenizer(ignorecase=ignorecase, include_boundaries=include_boundaries)
        self.tokenizer = tokenizer
        self.ignorecase = ignorecase
        self.include_boundaries = include_boundaries
        self.aggregate_doc = aggregate_doc

    def __iter__(self):
        for doc in generate_doc(self.files, tokenizer=self.tokenizer,
                                aggregate_doc=self.aggregate_doc):
            yield doc


class NormalizedIterable:

    def __init__(self, resolutions: ResolutionIterable, normalize_functions: List[Callable]):
        self.resolutions = resolutions
        self.normalize_functions = normalize_functions

    def __iter__(self):
        for resolution in self.resolutions:
            for normalize_func in self.normalize_functions:
                resolution.tokens = normalize_func(resolution.tokens)
            yield resolution
