import copy
import re
from collections import defaultdict
from typing import Dict, Generator, List, Union

import pagexml.model.physical_document_model as pdm


class StaBeDoc(pdm.StructureDoc):

    def __init__(self, doc_id: str, metadata: Dict[str, any] = None,
                 text_regions: List[pdm.PageXMLTextRegion] = None):
        super().__init__(doc_id=doc_id, metadata=metadata)
        self.text_regions = text_regions

    @property
    def stats(self):
        stats = {
            'text_regions': len(self.text_regions),
            'lines': 0,
            'words': 0
        }
        for tr in self.text_regions:
            tr_stats = tr.stats
            stats['lines'] += tr_stats['lines']
            stats['words'] += tr_stats['words']
        return stats

    @property
    def json(self):
        json_doc = {
            'id': self.id,
            'metadata': copy.deepcopy(self.metadata),
            'text_regions': [tr.json for tr in self.text_regions]
        }
        return json_doc


def read_text_file(text_file: str):
    with open(text_file, 'rt') as fh:
        for line in fh:
            yield line


def parse_text_file(text_file: str) -> Union[Generator[Dict[str, Union[str, int]], None, None], None]:
    for line in read_text_file(text_file):
        if match := re.match(r"^#facs_(\d+)_(\w+) +(.*)", line):
            facs = int(match.group(1))
            line_id = match.group(2)
            line_text = match.group(3)
        else:
            continue
        yield {
            'facs': facs,
            'line_id': line_id,
            'line_text': line_text
        }
    return None


def map_text_lines_to_scans(text_file: str, book_scan: Dict[int, pdm.PageXMLScan], debug: int = 0):
    """Map lines from the text file to lines in the corresponding scan."""
    text_scans = {}
    text_lines = defaultdict(dict)
    if debug > 0:
        print("map_text_lines_to_scans()")
    for line in parse_text_file(text_file):
        scan = book_scan[line['facs']]
        if debug > 0:
            print(f"\tline: {line}")
        text_scans[scan.id] = scan
        text_lines[scan.id][line['line_id']] = line
    return text_lines, text_scans


def align_text_and_tr_lines(text_lines: Dict[str, Dict[str, Dict[str, any]]],
                            text_scan_lines: Dict[str, Dict[str, pdm.PageXMLTextLine]],
                            scan: pdm.PageXMLScan, tr: pdm.PageXMLTextRegion, debug: int = 0):
    """For a given text region in the scan, align its text lines with the lines in the
    text file that belong to the same scan and text region.

    The challenge is that in the text files, some of the line identifiers are incorrect
    because they are shifted from their identifiers in the TEI and PageXML representations.

    This has been traced to an error with handling empty lines that in the PageXML correspond
    to line.text have None values. The alignment handles these shifts and maps the line
    identifier in the text file to line in the PageXML, which retains its original
    identifier.
    """
    num_shifted = 0
    if debug > 0:
        print(f"align_text_and_tr_lines - tr.id: {tr.id}")
    for li, line in enumerate(tr.lines):
        line_id = tr.lines[li - num_shifted].id
        if debug > 1:
            print(f"  scan.id: {scan.id}  line.id: {line.id}\tline_id{line_id}\tline.text: {line.text}")
        if line_id not in text_lines[scan.id]:
            continue
        if line.text is None and text_lines[scan.id][line_id]['line_text'] == '':
            text_scan_lines[scan.id][line_id] = line
            continue
        if line.text != text_lines[scan.id][line_id]['line_text']:
            if debug > 0:
                print(f'  scan.id: {scan.id}  line.id {line.id} text differs from line_id {line_id} text')
            if line.text is None and line != tr.lines[-1]:
                none_shifted = 0
                next_line = line
                while next_line.text is None:
                    none_shifted += 1
                    if next_line != tr.lines[-1]:
                        next_line = tr.lines[li + none_shifted]
                    else:
                        break
                if debug > 0:
                    print('\tline.text is None and line is not last line')
                    print(f"\tnone_shifted: {none_shifted}\tnum_shifted: {num_shifted}")
                    print(f'\t\tnext_line.id {next_line.id} text: {next_line.text}')
                    print(f"\t\ttext_line:      {text_lines[scan.id][line.id]}")
                if next_line.text == text_lines[scan.id][line_id]['line_text']:
                    num_shifted += none_shifted
                    if debug > 0:
                        print(f"num_shifted: {num_shifted}")
                    continue
        text_scan_lines[scan.id][line_id] = line

        if line.text != text_lines[scan.id][line_id]['line_text']:
            print(f"WARNING unaligned texts for scan {scan.id}, line {line.id}:")
            print(f"\ttext_line:      {text_lines[scan.id][line.id]}")
            print(f"\ttext_scan_line: {text_scan_lines[scan.id][line.id].text}")
    return None


def map_text_lines_to_scan_lines(text_lines: Dict[str, Dict[str, Dict[str, any]]],
                                 text_scans: Dict[str, pdm.PageXMLScan],
                                 debug: int = 0) -> Dict[str, Dict[str, pdm.PageXMLTextLine]]:
    """For a set of lines from a StaBe text, find the corresponding lines in the
    scans associated with that text. """
    text_scan_lines = defaultdict(dict)
    for scan_id in text_scans:
        scan = text_scans[scan_id]
        if debug > 0:
            print(f"map_text_lines_to_scan_lines scan.id: {scan.id}")
        for tr in scan.text_regions:
            align_text_and_tr_lines(text_lines, text_scan_lines, scan, tr, debug=debug)
    for scan_id in text_scan_lines:
        if len(text_lines[scan_id]) != len(text_scan_lines[scan_id]):
            print(f"Error mapping text lines to scan lines for scan_id {scan_id}")
            print(f"num text_lines: {len(text_lines[scan_id])}")
            print(f"num text_scan_lines: {len(text_scan_lines[scan_id])}")
            missing = [line_id for line_id in text_lines[scan_id] if line_id not in text_scan_lines[scan_id]]
            raise KeyError(f"line_ids missing from scan {scan_id}: {missing}")
    return text_scan_lines


def make_text_tr(text_scan_lines: Dict[str, Dict[str, pdm.PageXMLTextLine]],
                 scan: pdm.PageXMLScan, tr: pdm.PageXMLTextRegion,
                 debug: int = 0) -> Union[pdm.PageXMLTextRegion, None]:
    """Create a PageXMLTextRegion for the text lines in a text region that
    correspond to the lines in a StaBe text."""
    text_tr_lines = {}
    for line_id in text_scan_lines[scan.id]:
        line = text_scan_lines[scan.id][line_id]
        if line in tr.lines:
            text_tr_lines[line_id] = copy.deepcopy(line)
    if debug > 0:
        print(f"make_text_tr - len(text_tr_lines): {len(text_tr_lines)}")
    for line_id in text_tr_lines:
        line = text_tr_lines[line_id]
        line.metadata['pagexml_line_id'] = line.id
        line.metadata['text_line_id'] = line_id
        if line.id != line_id:
            if debug > 0:
                print(f'tr.id {tr.id} - line id mismatch: line.id {line.id}\tline_id {line_id}')
    if len(text_tr_lines) == 0:
        return None
    text_tr_line_ids = [line_id for line_id in text_tr_lines]
    text_tr_id = text_tr_line_ids[0].split('l')[0]
    # print(f"scan_id: {scan_id}\ttr.id: {tr.id}\ttext_tr_lines: {text_tr_lines}")
    text_tr_lines = [text_tr_lines[line_id] for line_id in text_tr_lines]
    coords = pdm.parse_derived_coords(text_tr_lines)
    metadata = {
        'scan_id': scan.id,
    }
    if debug > 0:
        print(f"make_text_tr - scan.id: {scan.id}")
        print(f"make_text_tr - len(text_tr_lines): {len(text_tr_lines)}")
    return pdm.PageXMLTextRegion(doc_id=text_tr_id, metadata=metadata,
                                 coords=coords, lines=text_tr_lines)


def is_discontinuous(scan_tr: pdm.PageXMLTextRegion, text_tr: pdm.PageXMLTextRegion,
                     debug: int = 0):
    """Check if the lines in a text regions that are associated with a StaBe text are
    not interspersed with other lines in the same text region, but that do not belong
    to the StaBe text. If they are discontinuous, the text region needs to be split."""
    text_line_num_min = int(text_tr.lines[0].id.split('l')[-1])
    text_line_num_max = int(text_tr.lines[-1].id.split('l')[-1])
    text_tr_line_ids = [line.id for line in text_tr.lines]
    for line in scan_tr.lines:
        line_num = int(line.id.split('l')[-1])
        if line.id not in text_tr_line_ids:
            if pdm.is_horizontally_overlapping(line, text_tr, threshold=0.9) is False:
                return False
            if pdm.is_vertically_overlapping(line, text_tr, threshold=0.9) is False:
                return False
            if line.text is None:
                return False
            if line_num < text_line_num_min:
                return False
            if line_num > text_line_num_max:
                return False
            print('discontinuous because line is overlapping but no in text tr')
            print(f"\tline: {line.id} {line.coords.box}\t{line.text}")
            print(f"\ttext_tr: {text_tr.coords.box}")
            return True


def map_text_scan_lines_to_regions(text_scan_lines: Dict[str, Dict[str, pdm.PageXMLTextLine]],
                                   text_scans: Dict[str, pdm.PageXMLScan],
                                   debug: int = 0) -> Dict[str, Dict[str, pdm.PageXMLTextRegion]]:
    """Map the lines from a PageXMLScan associated with a StaBe text to a set of PageXMLTextRegion
    objects."""
    text_scan_trs = defaultdict(dict)
    for scan_id in text_scan_lines:
        scan = text_scans[scan_id]
        if debug > 0:
            print(f'map_text_scan_lines_to_regions - scan.id: {scan.id}\tscan_id: {scan_id}')
        for scan_tr in scan.text_regions:
            text_tr = make_text_tr(text_scan_lines, scan, scan_tr, debug=debug)
            if text_tr is None:
                continue
            if debug > 0:
                print(
                    f"map_text_scan_lines_to_regions - text_tr: {text_tr.id} stats['lines']: {text_tr.stats['lines']}")
            if text_tr.coords is None:
                print(f'WARNING - text_tr without coordinates: {text_tr.id}')
                continue
            if is_discontinuous(scan_tr, text_tr):
                print('map_text_scan_lines_to_regions - DISCONTINUOUS')
                for line in scan_tr.lines:
                    print(f"\tscan tr {scan_tr.id} line {line.id}: {line.text}")
                for line in text_tr.lines:
                    print(f"\ttext tr {text_tr.id} line {line.id}: {line.text}")
            text_scan_trs[scan_id][text_tr.id] = text_tr
        num_scan_tr_lines = sum([text_scan_trs[scan_id][tr_id].stats['lines'] for tr_id in text_scan_trs[scan_id]])
        if num_scan_tr_lines != len(text_scan_lines[scan_id]):
            print(f"map_text_scan_lines_to_regions - unequal number of lines")
            print(f"\tnum_scan_tr_lines: {num_scan_tr_lines}")
            print(f"\tlen(text_scan_lines[scan_id]): {len(text_scan_lines[scan_id])}")
            raise ValueError("map_text_scan_lines_to_regions - unequal number of lines")
    return text_scan_trs


def map_text_scan_trs_to_stabe_doc(text_id: str, file_id: str, text_scan_trs: Dict[str, Dict[str, pdm.PageXMLTextRegion]]) -> StaBeDoc:
    """Create a StaBeDoc for a given text identifier and the corresponding text regions."""
    trs = []
    for scan_id in text_scan_trs:
        for tr_id in text_scan_trs[scan_id]:
            tr = text_scan_trs[scan_id][tr_id]
            tr.metadata['text_id'] = text_id
            tr.metadata['file_id'] = file_id
            if 'text_region_id' in tr.metadata:
                del tr.metadata['text_region_id']
            trs.append(tr)
    metadata = {
        'text_id': text_id,
        'file_id': file_id,
        'scan_ids': [scan_id for scan_id in text_scan_trs]
    }
    return StaBeDoc(doc_id=text_id, metadata=metadata, text_regions=trs)

