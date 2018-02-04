import numpy as np
from gensim.models import Word2Vec
import sys
import os.path
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8') #UTF8 #latin-1
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from random import shuffle

class GensimW2VToy:

    def __init__(self,file_name, destination, reload=True, size=300, window=5, min_count=5, workers=4):
        self.file_name = file_name
        self.destination = destination
        self.model = Word2Vec( iter=1, size=size, window=window, min_count=min_count, workers=workers, sg=1, alpha=1, min_alpha= 0.00001, negative=10 , sorted_vocab=1)

    def load_model_from_file(self, file_name=None):
        if file_name is None:
            file_name = self.file_name
        self.model = Word2Vec.load(file_name)

    def build_generator(self, file, yields=1):
        tokens = []
        with open(file, 'r') as f:
            for line in f:
                line_tokens = line.split()
                tokens.append(line_tokens)

        for yield_ in xrange(yields):
            shuffle(tokens)
            for token in tokens:
                yield token

    def build_dict(self, file):
        generator = self.build_generator(file)
        self.model.build_vocab(generator, keep_raw_vocab=False, trim_rule=None, progress_per=100, update=False)


    def train(self, epochs=10, start_alpha=0.025, end_alpha=0.0001):

        for i , new_start_alpha in enumerate(np.linspace(start_alpha,end_alpha,epochs)):
            file_name = self.destination + '_epoche_' + str(i) + '_lr_' + str(new_start_alpha)

            if os.path.isfile(file_name):
                self.load_model_from_file(file_name)
                print file_name + 'loaded'
            else:
                generator = self.build_generator(self.file_name, 1)
                self.model.train(generator, total_examples=self.model.corpus_count, total_words=None, epochs=1, start_alpha=new_start_alpha, end_alpha=end_alpha,
                          word_count=0, queue_factor=2, report_delay=1.0, compute_loss=None)
                self.model.wv.save_word2vec_format(file_name+"_vectors")
                self.model.save(file_name )


