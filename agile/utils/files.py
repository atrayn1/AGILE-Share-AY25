from random import Random, randrange
import random
from itertools import islice
from math import ceil
import os
import pickle

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

# Save a file to a pickle file, in the saved_files directory
def save(file_name, data):
    with open(os.path.abspath(f'./saved_data/{file_name}'), 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

# Generate a random name
def random_name():
    with open(find('../names/first.txt')) as F, open(find('../names/last.txt')) as L:
        new_name = random_line(F) + '-' + random_line(L)
        return new_name
                
def generate_aliases(df):
    first = []
    last = []
    name_list = []

    
    with open(find('../names/first.txt')) as F, open(find('../names/last.txt')) as L:
        for line_f in F:
            first.append(line_f.strip())
        for line_l in L:
            last.append(line_l.strip())
            
        for f in first:
            for l in last:        
                name_list.append(l + '-' + f) 
           
    
    random.seed(24)

    adid_dict = {}
    
    if len(name_list) >= len(df['advertiser_id'].unique()):
        for id in df['advertiser_id'].unique():
            adid_dict[id] = name_list.pop(random.randrange(len(name_list)))
    else:
        names_left = len(name_list)
        for id in df['advertiser_id'].unique():
            if names_left > 0:
                if names_left % 100000 == 0:
                    print(names_left)
                adid_dict[id] = name_list.pop(random.randrange(names_left))
                names_left -= 1
            else:
                adid_dict[id] = 'Unnamed_Alias'
            
        
    return adid_dict
