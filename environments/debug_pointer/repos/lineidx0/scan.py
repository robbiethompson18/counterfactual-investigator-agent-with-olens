def find_idx0(lines, needle):
    for i, line in enumerate(lines):
        if needle in line:
            return i
    return -1
