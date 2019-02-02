import csv
import unicodedata
import numpy as np
from constants import ENCODER, SEQLEN, BATCHSIZE, VECLEN

import tensorflow as tf


def asciize(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')


def encode_np(s):
    a = np.frombuffer(s, 'uint8')
    return ENCODER[a]


def preprocess(s):
    return encode_np(asciize(s))

with open('data.csv') as fh:
    data = list(map(lambda x: preprocess(x[0]) if len(x) else '', csv.reader(fh)))


def sequenced(data, seqlen=30):
    for e in data:
        if len(e) > seqlen:
            for i in range(len(e) - seqlen):
                yield e[i : i + seqlen]
        else:
            print('Entry too short for seqlen ({} chars)'.format(seqlen))

x_data = tf.placeholder(dtype='uint8', shape=(None, None))

dataset = tf.data.Dataset.from_tensor_slices((x_data,))

dataset = dataset.shuffle(100).batch(BATCHSIZE)

all_data = np.array(list(sequenced(data, seqlen=SEQLEN)))