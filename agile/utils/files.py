from random import Random
from itertools import islice
from math import ceil
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

def random_line(iterator):
    index1 = 1
    result = None
    while True:
        try:
            # slice off the last character at the end of a line
            # this is to deal with newlines, but there are some quirks to this
            # the last line in an input file should not end in a newline,
            # contrary to the POSIX standard
            # there should be a junk character at the end of the last line
            # instead, I personally use space in first.txt and last.txt
            result = next(iterator)[:-1]
        except StopIteration:
            return result

        r = Random().random()
        offset = max(ceil(r * index1 / (1.0 - r)), 1)
        iterator = islice(iterator, offset - 1, None)
        index1 += offset

# Dynamically find the path of a file given the file name
# This is needed for containerization
# Otherwise the image and names could not be loaded
def find(relative_path):
    return os.path.join(ROOT_DIR, relative_path)