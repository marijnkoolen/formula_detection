from typing import Dict, List, Generator, Union
import csv
import gzip
import re
import glob
import json
import zipfile

from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup


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
    text = '\n'.join([re.sub(r'\{.*?\}', '', p.text) for p in transcript.find_all('p')])
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
    review_file = '/Users/marijnkoolen/Code/impact-of-fiction/data/reviews/review-stats.csv.gz'
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
