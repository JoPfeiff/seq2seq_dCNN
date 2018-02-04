from gensim_w2v_toy import GensimW2VToy
from RNN2RNN import train as rnn_train
from RNN2RNN import TestConfigToy as RNN2RNNConfig
from data_utils import LogFileWriter
import tensorflow as tf
from nn_config import TestConfigToy as dCNN2RNNConfig
from dCNN2RNN import train as dcnn_train

raw_sentences_file = 'data/text_data/raw_sentences.txt'
embedding_destination = 'data/embedding_data/toy_embeddings'
gw2v = GensimW2VToy(raw_sentences_file, embedding_destination,  size=50, window=5, min_count=0, workers=24)
gw2v.build_dict(raw_sentences_file)
gw2v.train(epochs=100, start_alpha=0.025, end_alpha=0.00001)
print("Embedding finished")

embeddings_file = 'data/embedding_data/toy_embeddings_epoche_99_lr_1e-05_vectors'

train_log_file = LogFileWriter('data/RNN/train_log.csv')
test_log_file = LogFileWriter('data/RNN/test_log.csv')
train_config = RNN2RNNConfig()
train_config.location = embeddings_file
train_log_file.append_text('RNN LSTM')
test_log_file.append_text('RNN LSTM')
rnn_train(train_config, False, train_log_file, test_log_file)
tf.reset_default_graph()

train_log_file = LogFileWriter('data/dCNN/train_log.csv')
test_log_file = LogFileWriter('data/dCNN/test_log.csv')
train_config = dCNN2RNNConfig()
train_config.location = embeddings_file
train_log_file.append_text('DILATED CNN')
test_log_file.append_text('DILATED CNN')
dcnn_train(train_config, False, train_log_file, test_log_file)






