from random import Random
from itertools import islice
from math import ceil

def random_line(iterator):
    index1 = 1
    result = None
    while True:
        try:
            # slice off the newline character at the end of a line
            result = next(iterator)[:-1]
        except StopIteration:
            return result

        r = Random().random()
        offset = max(ceil(r * index1 / (1.0 - r)), 1)
        iterator = islice(iterator, offset - 1, None)
        index1 += offset

