# Helper functions for the model

import unicodedata
import string
import time
import math
import torch
from torch.autograd import Variable

all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    file = unicodedata.unicode(open(filename).read())
    return file, len(file)

# turn a string into tensor
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

# time lapsed
def time_since(since):
    s = time.time() - since
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m,s)

