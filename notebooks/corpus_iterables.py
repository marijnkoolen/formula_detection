import csv
import gzip
import re
import glob
import json
import zipfile
from typing import Callable, Dict, List, Generator, Union

from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from fuzzy_search.tokenization.token import Tokenizer, CustomTokenizer
from fuzzy_search.tokenization.token import Doc


def read_text_csv(text_file: str, sep: str = '\t',
                  file_has_header: bool = True, headers: List[str] = None,
                  skip_invalid_lines: bool = False,
                  debug: int = 0):
    """Generic paragraph reader that can read from CSV/TSV files.

    :param text_file: the filename of an input text file.
    :param sep: the field separator str (default is '\t').
    :param file_has_header: Boolean flag to indicate whether the input file starts.
    :param headers: an optional list of headers to use. Even if there is a header line in the
    input file, you can use this parameter to override the headers.
    :param skip_invalid_lines: ignore lines with a different number of columns than
    the number of headers
    :param debug: print basic debug information.
    """
    opener = gzip.open if text_file.endswith('.gz') else open
    if debug > 0:
        print('parsing file', text_file)
    with opener(text_file, 'rt') as fh:
        if file_has_header is True:
            header_line = next(fh)
            if headers is None:
                headers = header_line.strip('\n').split(sep)
        elif file_has_header is False and headers is None:
            raise ValueError("If you set 'file_has_header' to False, you must pass headers.")
        for li, line in enumerate(fh):
            line_idx = li + 2 if file_has_header else li + 1
            if len(headers) > 1:
                row = line.strip('\n').split(sep)
            else:
                row = [line.strip('\n')]
            if len(row) != len(headers):
                if skip_invalid_lines is True:
                    continue
                else:
                    raise IndexError(f"Error in line {line_idx}: number of columns {len(row)} "
                                     f"differs from the number of headers {len(headers)}.")
            else:
                yield {header: row[hi] for hi, header in enumerate(headers)}


def make_reader(sep: str = '\t',
                file_has_header: bool = True, headers: List[str] = None,
                skip_invalid_lines: bool = False, debug: int = 0):
    """Closure to generate a reader function that takes a list of text files as input
    and returns lists of tokens per text."""

    def reader(text_file: str):
        return read_text_csv(text_file, sep=sep, file_has_header=file_has_header,
                             headers=headers, skip_invalid_lines=skip_invalid_lines,
                             debug=debug)
    return reader


class CorpusIterable:

    def __init__(self, text_files: List[str], tokenize_func: Callable = None,
                 text_field: str = None, doc_id_field: str = None, sep: str = '\t',
                 file_has_header: bool = True, headers: List[str] = None,
                 skip_invalid_lines: bool = False, debug: int = 0):
        """Instantiation of a CorpusIterable.

        :param text_files: a list of filenames of input text files.
        :param tokenize_func: a text tokenizer function that takes a text string as input
        and returns a list of tokens. Defaults to the tokenize function of the FuzzySearch Tokenizer.
        :param text_field: the name of the field/column that contains the text
        :param doc_id_field: the name of the field/column that contains the document identifier
        :param sep: the field separator str (default is '\t').
        :param file_has_header: Boolean flag to indicate whether the input file starts.
        :param headers: an optional list of headers to use. Even if there is a header line in the
        input file, you can use this parameter to override the headers.
        :param skip_invalid_lines: ignore lines with a different number of columns than
        the number of headers
        :param debug: print basic debug information.
        """
        self.text_files = text_files
        if tokenize_func:
            self.tokenizer = CustomTokenizer(tokenize_func)
        else:
            # load a default tokenizer if non is passed
            self.tokenizer = Tokenizer()
        self.text_field = text_field
        self.doc_id_field = doc_id_field
        self.sep = sep
        self.file_has_header = file_has_header
        self.headers = headers
        self.skip_invalid_lines = skip_invalid_lines
        self.debug = debug
        self.reader = make_reader(sep=sep, file_has_header=file_has_header, headers=headers,
                                  skip_invalid_lines=skip_invalid_lines)

    def __iter__(self):
        for text_file in self.text_files:
            for di, doc_dict in enumerate(self.reader(text_file)):
                file_line_idx = di + 2 if self.file_has_header else di + 1
                if self.doc_id_field:
                    doc_id = doc_dict[self.doc_id_field]
                else:
                    doc_id = f"{text_file}:{file_line_idx}"
                tokenized_doc = self.tokenizer.tokenize_doc(doc_dict[self.text_field],
                                                            doc_id=doc_id)
                yield tokenized_doc


def read_reviews(review_file):
    with gzip.open(review_file, 'rt') as fh:
        reader = csv.reader(fh, delimiter='\t')
        headers = next(reader)
        for row in reader:
            review = {header: row[hi] for hi, header in enumerate(headers)}
            yield review
    return None


def read_charter_files(charter_zipfile):
    with zipfile.ZipFile(charter_zipfile, 'r') as zh:
        filenames = sorted(zh.namelist())
        # print(filenames)
        for fname in filenames:
            if fname.startswith('__MACOSX'):
                continue
            if not fname.endswith('.xml'):
                continue
            with zh.open(fname) as fh:
                yield fname, fh.read()


def extract_charter_text(charter):
    soup = BeautifulSoup(charter, 'lxml')
    transcript = BeautifulSoup(soup.find('transcriptie').text, 'lxml')
    text = '\n'.join([re.sub(r'{.*?}', '', p.text) for p in transcript.find_all('p')])
    return text


class CharterSentences:

    def __init__(self, charter_zipfile):
        self.charter_zipfile = charter_zipfile

    def __iter__(self):
        for fname, charter in read_charter_files(self.charter_zipfile):
            text = extract_charter_text(charter)
            for pi, para in enumerate(text.split('\n')):
                yield {
                    'doc_id': fname,
                    'para_id': f'{fname}-{pi + 1}',
                    'text': para,
                    'words': [w for w in re.split(r'\W+', para) if w != '']
                }


def read_articles(wiki_files: List[str]) -> Generator[Dict[str, any], None, None]:
    for wiki_file in wiki_files:
        with open(wiki_file, 'rt') as fh:
            for line in fh:
                article = json.loads(line)
                yield article
    return None


def read_voc_paras(voc_file, add_boundaries: bool = False):
    with gzip.open(voc_file, 'rt') as fh:
        for line in fh:
            row = line.strip().split('\t')
            if len(row) == 3:
                doc_id, para_id, text = row
                words = [t for t in re.split(r'\W*\s+\W*', re.sub(r'\W+$', '', text)) if t != '']
                if add_boundaries is True:
                    words = ['<PARA_START>'] + words + ['<PARA_END>']
                yield {'doc_id': doc_id, 'para_id': para_id, 'text': text, 'words': words}
    return None


class DocReader:

    def __init__(self, text_files: Union[List[str], str], ignorecase: bool = False,
                 include_boundaries: bool = False, has_headers: bool = False,
                 use_headers: List[str] = None, remove_punctuation: bool = False,
                 tokenizer: Tokenizer = None,
                 id_field: str = None, text_field: str = None):
        self.text_files = text_files if isinstance(text_files, list) else [text_files]
        self.ignorecase = ignorecase
        self.remove_punctuation = remove_punctuation
        self.include_boundaries = include_boundaries
        self.has_headers = has_headers
        self.id_field = id_field if id_field else 'doc_id'
        self.text_field = text_field if text_field else 'text'
        self.use_headers = use_headers
        if not tokenizer:
            self.tokenizer = Tokenizer(ignorecase=ignorecase, include_boundary_tokens=include_boundaries,
                                       remove_punctuation=remove_punctuation)
        else:
            self.tokenizer = tokenizer

    def __iter__(self):
        for text_file in self.text_files:
            with open(text_file, 'rt') as fh:
                if self.has_headers:
                    headers = next(fh).split()
                elif self.use_headers:
                    headers = self.use_headers
                else:
                    headers = ['doc_id', 'text']
                for line in fh:
                    values = line.strip('\n').split('\t')
                    # print(len(values))
                    doc = {header: values[hi] for hi, header in enumerate(headers)}
                    metadata = {header: doc[header] for header in headers
                                if header != self.text_field and header != self.id_field}
                    if self.ignorecase:
                        doc[self.text_field] = doc[self.text_field].lower()
                    doc = self.tokenizer.tokenize(doc[self.text_field], doc_id=doc[self.id_field])
                    doc.metadata = metadata
                    yield doc
        return None


class GoldenAgentsSentences:

    def __init__(self, sent_file: str, word_sep: str = r'\W+', lower: bool = False):
        self.sent_file = sent_file
        self.word_sep = word_sep
        self.lower = lower

    def __iter__(self):
        with gzip.open(self.sent_file, 'rt') as fh:
            for line in fh:
                row = line.strip().split('\t')
                if len(row) != 3:
                    continue
                doc_id, para_id, text = row
                if self.lower is True:
                    text = text.lower()
                words = [w for w in re.split(self.word_sep, text) if w != '']
                yield {'doc_id': doc_id, 'id': para_id, 'text': text, 'words': words}


class VOCSentences:

    def __init__(self, voc_file, add_boundaries: bool = False):
        self.voc_file = voc_file
        self.add_boundaries = add_boundaries

    def __iter__(self):
        for voc_para in read_voc_paras(self.voc_file, add_boundaries=self.add_boundaries):
            yield voc_para


class WikiSentences:

    def __init__(self, wiki_files):
        self.wiki_files = wiki_files

    def __iter__(self):
        for article in read_articles(self.wiki_files):
            sents = sent_tokenize(article['text'])
            for si, sent in enumerate(sents):
                sent_id = f'{article["id"]}-{si + 1}'''
                words = [w for w in re.split(r'\W+', sent.lower()) if w != '']
                words = ['<SENT_START>'] + words + ['<SENT_END>']
                yield {'id': sent_id, 'text': sent.lower(), 'words': words}


class ResolutionSentences:

    def __init__(self, files, ignorecase: bool = False, include_boundaries: bool = False):
        self.files = files
        self.ignorecase = ignorecase
        self.include_boundaries = include_boundaries

    def __iter__(self):
        for fname in self.files:
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
                    para = {'doc_id': res_id, 'id': para_id, 'text': text}
                    para_text = para['text'].lower() if self.ignorecase else para['text']
                    para['words'] = [w for w in re.split(r'\W+', para_text) if w != '']
                    if self.include_boundaries:
                        para['words'] = ['<START>'] + para['words'] + ['<END>']
                    yield para


class NovelSentences:

    def __init__(self, novel_file):
        self.novel_file = novel_file

    def __iter__(self):
        with gzip.open(self.novel_file, 'rt') as fh:
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                doc_id, para_id, text = parts
                for si, sent in enumerate(sent_tokenize(text)):
                    yield {
                        'doc_id': doc_id,
                        'para_id': para_id,
                        'id': f'{para_id}-sent-{si + 1}',
                        'text': sent,
                        'words': [w for w in re.split(r'\W+', sent) if w != '']
                    }


class NewspaperSentences:

    def __init__(self, para_file: str, lowercase: bool = False):
        self.para_file = para_file
        self.lowercase = lowercase

    def __iter__(self):
        with gzip.open(self.para_file, 'rt') as fh:
            for line in fh:
                columns = line.strip().split('\t')
                if len(columns) == 4:
                    art_id, para_id, field, text = columns
                elif len(columns) == 3:
                    art_id, para_id, text = columns
                else:
                    print(line)
                    print(columns)
                    raise ValueError(f'unexpected number of columns')
                para = {
                    'doc_id': art_id,
                    'id': para_id,
                    'text': text
                }
                if self.lowercase:
                    para['words'] = [w for w in re.split(r'\W+', text.lower()) if w != '']
                else:
                    para['words'] = [w for w in re.split(r'\W+', text) if w != '']
                yield para


class ReviewSentences:

    def __init__(self, review_file):
        self.review_file = review_file

    def __iter__(self):
        for review in read_reviews(self.review_file):
            sentences = sent_tokenize(review['review_text'])
            for si, sent in enumerate(sentences):
                words = [w for w in re.split(r'\W+', sent.lower()) if w != '']
                yield {
                    'id': f"{review['review_id']}-{si + 1}",
                    'text': sent,
                    'words': words
                }


class StaBeSentences:

    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for fname in self.files:
            with gzip.open(fname, 'rt') as fh:
                reader = csv.reader(fh, delimiter='\t')
                for row in reader:
                    para = {'id': f'{row[0]}-{row[1]}', 'text': row[2]}
                    para['words'] = [w for w in re.split(r'\W+', para['text']) if w != '']
                    para['words'] = ['<PARA_START>'] + para['words'] + ['<PARA_END>']
                    yield para


def get_charter_sents():
    data_dir = '/Users/marijnkoolen/Data/workshops/2018/Data-Scopes-Developers-2018/Opdracht-Henegouwse-Registers'
    charter_zipfile = f'{data_dir}/data.zip'
    return CharterSentences(charter_zipfile)


def get_newspaper_sents():
    para_file = '/Volumes/Samsung_T5/Data/Delpher/Kranten/newspapers_18th_century-para_format.tsv.gz'
    return NewspaperSentences(para_file, lowercase=True)


def get_notarial_sents():
    ga_para_file = '/Volumes/Samsung_T5/Data/PageXML/GoldenAgents-text_para_format.tsv.gz'
    return GoldenAgentsSentences(ga_para_file)


def get_novel_sents():
    novel_para_file = '/Volumes/Samsung_T5/Data/ImpFic/books/novels-para_format.tsv.gz'
    return NovelSentences(novel_para_file)


def get_review_sents():
    review_file = '/Users/marijnkoolen/Code/impact-of-fiction/data/reviews/stats/review-stats-old.csv.gz'
    return ReviewSentences(review_file)


def get_stabe_sents():
    stabe_files = [
        '/Volumes/Samsung_T5/Data/PageXML/stabe-mandate-para_line_format.tsv.gz',
        '/Volumes/Samsung_T5/Data/PageXML/stabe-policy-para_line_format.tsv.gz'
    ]
    return StaBeSentences(stabe_files)


def get_stabe_mandate_sents():
    stabe_files = [
        '/Volumes/Samsung_T5/Data/PageXML/stabe-mandate-para_line_format.tsv.gz'
    ]
    return StaBeSentences(stabe_files)


def get_stabe_police_sents():
    stabe_files = [
        '/Volumes/Samsung_T5/Data/PageXML/stabe-policy-para_line_format.tsv.gz'
    ]
    return StaBeSentences(stabe_files)


def get_resolutions_sents(resolution_files: Union[str, List[str]] = None,
                          ignorecase: bool = True, include_boundaries: bool = True):
    if resolution_files is None:
        resolution_dir = '/Users/marijnkoolen/Code/Huygens/republic-project/data/paragraphs'
        resolution_files = [f'{resolution_dir}/resolutions-paragraphs-random.tsv.gz']
    elif isinstance(resolution_files, str):
        resolution_files = [resolution_files]
    return ResolutionSentences(resolution_files, ignorecase=ignorecase,
                               include_boundaries=include_boundaries)


def get_voc_sents():
    voc_file = '/Volumes/Samsung_T5/Data/PageXML/NA/HTR-VOC-para_format.tvs.gz'
    return VOCSentences(voc_file, add_boundaries=True)


def get_wiki_sents():
    wiki_dir = '/Volumes/Samsung_T5/Data/Wikipedia/NL-dump/wikinl_txt/'
    wiki_files = glob.glob(wiki_dir + '**/wiki_*')
    return WikiSentences(wiki_files)


def get_all_sents():
    return {
        'charters': get_charter_sents(),
        'deeds': get_notarial_sents(),
        'newspapers': get_newspaper_sents(),
        'novels': get_novel_sents(),
        'resolutions': get_resolutions_sents(),
        'reviews': get_review_sents(),
        'stabe': get_stabe_sents(),
        'stabe_mandate': get_stabe_mandate_sents(),
        'stabe_police': get_stabe_police_sents(),
        'voc': get_voc_sents(),
        'wiki': get_wiki_sents()
    }
