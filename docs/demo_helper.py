def lines_to_paras(lines):
    para_lines = []
    for line in lines:
        if line == '\n':
            yield ' '.join(para_lines).strip()
            para_lines = []
        else:
            para_lines.append(line.strip('\n'))
    if len(para_lines) > 0:
        yield ' '.join(para_lines).strip()
