# Author: Moustafa Alzantot (malzantot@ucla.edu)
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# NOTICE:
# This file has been modified by Riccardo Spolaor in some of its parts.

import os
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf

from google.protobuf import text_format
import licensed_scripts.data_utils as data_utils

class LM(object):
    def __init__(self):
        self.PBTXT_PATH = 'google_language_model\\graph-2016-09-10.pbtxt'
        self.CKPT_PATH = 'google_language_model\\ckpt-*'
        self.VOCAB_PATH = 'google_language_model\\vocab-2016-09-10.txt'

        self.BATCH_SIZE = 1
        self.NUM_TIMESTEPS = 1
        self.MAX_WORD_LEN = 50

        self.vocab = data_utils.CharsVocabulary(self.VOCAB_PATH, self.MAX_WORD_LEN)
        print('LM vocab loading done')

        self.sess, self.t = LM.__LoadModel(self.PBTXT_PATH, self.CKPT_PATH)

    def get_words_probs(self, prefix_words, list_words, suffix=None):
        targets = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        weights = np.ones([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.float32)

        if prefix_words.find('<S>') != 0:
            prefix_words = '<S> ' + prefix_words
        prefix = [self.vocab.word_to_id(w) for w in prefix_words.split()]
        prefix_char_ids = [self.vocab.word_to_char_ids(w) for w in prefix_words.split()]

        inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS, self.vocab.max_word_length], np.int32)

        samples = prefix[:]
        char_ids_samples = prefix_char_ids[:]
        inputs = [ [samples[-1]]]
        char_ids_inputs[0, 0, :] = char_ids_samples[-1]
        softmax = self.sess.run(self.t['softmax_out'],
        feed_dict={
            self.t['char_inputs_in']: char_ids_inputs,
            self.t['inputs_in']: inputs,
            self.t['targets_in']: targets,
            self.t['target_weights_in']: weights
        })
        # print(list_words)
        words_ids = [self.vocab.word_to_id(w) for w in list_words]
        word_probs =[softmax[0][w_id] for w_id in words_ids]
        word_probs = np.array(word_probs)

        if suffix == None:
            suffix_probs = np.ones(word_probs.shape)
        else:
            suffix_id = self.vocab.word_to_id(suffix)
            suffix_probs = []
            for idx, w_id in enumerate(words_ids):
                # print('..', list_words[idx])
                inputs = [[w_id]]
                w_char_ids = self.vocab.word_to_char_ids(list_words[idx])
                char_ids_inputs[0, 0, :] = w_char_ids
                softmax = self.sess.run(self.t['softmax_out'],
                                         feed_dict={
                                             self.t['char_inputs_in']: char_ids_inputs,
                                             self.t['inputs_in']: inputs,
                                             self.t['targets_in']: targets,
                                             self.t['target_weights_in']: weights
                                         })
                suffix_probs.append(softmax[0][suffix_id])
            suffix_probs = np.array(suffix_probs)            
        # print(word_probs, suffix_probs)
        return suffix_probs * word_probs

    @staticmethod
    def __LoadModel(gd_file, ckpt_file):
        """Load the model from GraphDef and Checkpoint.

        Args:
          gd_file: GraphDef proto text file.
          ckpt_file: TensorFlow Checkpoint file.

        Returns:
          TensorFlow session and tensors dict.
        """
        with tf.Graph().as_default():
          sys.stderr.write('Recovering graph.\n')
          with tf.io.gfile.GFile(gd_file, 'r') as f:
            s = f.read()#.decode()
            gd = tf.compat.v1.GraphDef()
            text_format.Merge(s, gd)

          tf.compat.v1.logging.info('Recovering Graph %s', gd_file)
          t = {}
          [t['states_init'], t['lstm/lstm_0/control_dependency'],
          t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
          t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
          t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
          t['all_embs'], t['softmax_weights'], t['global_step']
          ] = tf.import_graph_def(gd, {}, ['states_init',
                                          'lstm/lstm_0/control_dependency:0',
                                          'lstm/lstm_1/control_dependency:0',
                                          'softmax_out:0',
                                          'class_ids_out:0',
                                          'class_weights_out:0',
                                          'log_perplexity_out:0',
                                          'inputs_in:0',
                                          'targets_in:0',
                                          'target_weights_in:0',
                                          'char_inputs_in:0',
                                          'all_embs_out:0',
                                          'Reshape_3:0',
                                          'global_step:0'], name='')

          sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
          sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
          sess.run('save/restore_all', {'save/Const:0': ckpt_file})
          sess.run(t['states_init'])

        return sess, t