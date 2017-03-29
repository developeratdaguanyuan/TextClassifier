from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import os
import tensorflow as tf

def _read_words_label(filename):
    with tf.gfile.GFile(filename, "r") as f:
        cases = []
        for line in f.readlines():
            tokens = line.strip().decode("utf-8").split('\t')
            words = tokens[0].split()
            label = tokens[1].strip()
            cases.append([words, label])
        return cases

def _file_to_cases(filename, word_to_id, label_to_id):
    cases = _read_words_label(filename)
    data = []
    for case in cases:
        data.append([[word_to_id[word] if word in word_to_id else 0 for word in case[0]], label_to_id[case[1]]])
    data = sorted(data, key=lambda x: len(x[0]))
    return data

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        words = []
        for line in f.readlines():
            words.extend(line.strip().decode("utf-8").split('\t')[0].split())
        return words

def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    count_pairs = filter(lambda x: x[1] > 20, count_pairs)
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(1, len(words)+1)))
    return word_to_id

def _read_category(filename):
    with tf.gfile.GFile(filename, "r") as f:
        categories = []
        for line in f.readlines():
            categories.append(line.strip().decode("utf-8").split('\t')[1].strip())
    return set(categories)

def _build_category(filename):
    categories = _read_category(filename)
    counter = collections.Counter(categories)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    labels, _ = list(zip(*count_pairs))
    label_to_id = dict(zip(labels, range(len(labels))))
    return label_to_id

def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "devel.txt")

    word_to_id = _build_vocab(train_path)
    label_to_id = _build_category(train_path)
    
    train_data = _file_to_cases(train_path, word_to_id, label_to_id)
    valid_data = _file_to_cases(valid_path, word_to_id, label_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, vocabulary, word_to_id, label_to_id

class DataProducer(object):

    def __init__(self, data):
        self.data, self.label = zip(*data)
        self.size = len(self.data)
        self.cursor = 0
    
    def next_batch(self, n):
        if (self.cursor + n - 1 >= self.size):
            self.cursor = 0
        curr_data = self.data[self.cursor:self.cursor+n]
        curr_label = self.label[self.cursor:self.cursor+n]
        self.cursor += n

        max_length = max(len(l) for l in curr_data)
        x = np.zeros([n, max_length], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:len(curr_data[i])] = np.array(curr_data[i])
        y = np.array(curr_label)
        seq = np.array([max_length for i in range(n)])

        return x, y, seq
