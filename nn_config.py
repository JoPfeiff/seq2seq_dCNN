from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_utils as data_utils
from data_utils import get_embedding_data

def data_type():
  return  tf.float32

BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"



class TestConfigToy(object):
    def __init__(self):
        self.init_scale = 0.1
        self.learning_rate = 0.001
        self.num_layers = 2
        # self.num_steps = 10
        self.hidden_size = 200
        self.num_samples = 512
        self.max_epoch = 1
        self.max_max_epoch = 1
        self.keep_prob = 0.35
        self.cell_keep_prob = 0.35
        self.lr_decay = 0.99  # 1 / 1.15 # 0.5
        self.batch_size = 10
        self.test_batch_size = 10
        self.vocab_size = 10000
        self.embedding_dim = 300
        self.rnn_mode = BLOCK
        self.epoch_size = 1
        self.file_count = -1
        self.location = "data/embedding_data/toy_embeddings_epoche_99_lr_1e-05_vectors"
        self.cpu = "/cpu:0"
        self.gpu = "/cpu:0"
        self.min_doc_length = 50
        self.dtype = data_type()
        self.use_lstm = True
        self.buckets = [(5, 5), (10, 10), (15, 15), (20, 20)]
        self.cap = self.buckets[-1][0]
        self.max_gradient_norm = 5.0
        self.train_dir = "data/dCNN/"
        self.steps_per_checkpoint = 100
        self.vocab_size, self.embedding_dim, self.embeddings_mentions_list, self.embeddings_mentions, self.embeddings_np = get_embedding_data(
            self.location, self.vocab_size)
        self.train_set, self.test_set, self.validation_set = data_utils.sentences_train_test_validation_splitting_and_paragraph_generation(
            self.embeddings_mentions, 'data/text_data/raw_sentences.txt', splitting=(70, 20, 10))
        self.train_args = (self.batch_size, self.buckets, self.train_set)
        self.test_args = (self.test_batch_size, self.buckets, self.test_set)
        self.generator = data_utils.toy_text_generator
        self.static = True

        self.teacher_forcing = True
