import numpy as np
from tqdm import *
import random
import re
from preprocessing import pickle_call, pickle_dump

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 2
GO_ID = 1
EOS_ID = 0
UNK_ID = 3

def tokenize(text, lower= False, numbers=False):
    punctuation = "!\"#$%&\\'()*+,\-./:;<=>?@[\]\^\`\{\|\}~\"'\xe2\x80\x99"
    mentions = list(re.findall("\s*([" + punctuation + "]|[^" + punctuation + "\s]*)\s*", text))

    if lower or numbers:
        new_mentions = []
        for mention in mentions:

            if lower:
                mention = mention.lower()
            if numbers:
                if mention.isdigit():
                    mention = '0'
            new_mentions.append(mention)

        mentions = new_mentions

    return mentions


def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))


def has_letters(inputString):
    return bool(re.search(r'[a-zA-Z]', inputString))


def has_letters_and_numbers(inputString):
    return has_numbers(inputString) and has_letters(inputString)


def sentences_train_test_validation_splitting_and_paragraph_generation(embeddings_mentions,file_name, splitting=(70,20,10)):
    # embeddings_mentions = model.embeddings_mentions

    paragraphs = pickle_call(file_name + '.pickle')

    if paragraphs is None:

        splitting = np.array(splitting)
        splitting[1] += splitting[0]
        splitting[2] += splitting[1]

        splitting_counter = splitting[2]

        train_sentences = []
        test_sentences = []
        validation_sentences = []

        with open(file_name, 'r') as f:
            for line in f:
                rand_int = random.randint(1, splitting_counter)
                line_tokens = line.split()
                token = []
                for word in line_tokens:
                    try:
                        token.append(embeddings_mentions[word])
                    except:
                        pass

                if rand_int <= splitting[0]:
                    train_sentences.append(token)
                elif rand_int <= splitting[1]:
                    test_sentences.append(token)
                else:
                    validation_sentences.append(token)
        paragraphs = [[],[],[]]
        for i, sentences in enumerate([train_sentences, test_sentences, validation_sentences]):

            sentences_length = len(sentences)

            for j in tqdm(xrange(len(sentences) * 10 )):

                nr_sentences = random.randint(1, 10)
                batch_sentence = []

                for k in xrange(nr_sentences):
                    sentence = random.randint(0, sentences_length - 1)
                    batch_sentence += sentences[sentence]

                paragraphs[i].append(batch_sentence)
        pickle_dump(file_name + '.pickle', paragraphs)

    return paragraphs



def sentence_to_token_ids(text, embeddings_mentions):
    tokens = []
    tok = tokenize(text, True, True)
    for ment in tok:
        try:
            tokens.append(embeddings_mentions[ment])
        except:
            pass
    return tokens


class LogFileWriter:
    def __init__(self, filename):
        self.filename = filename
        self.open = False
        self.file = open(filename, 'a')

    def __open__(self):
        self.file = open(self.filename, 'a')

    def append_text(self, text):
        if not self.open:
            self.__open__()
        self.file.write(text + "\n")
        self.close()

    def close(self):
        self.file.close()

def toy_text_generator(batch_size, buckets, data_set):

    data_sizes = []
    bucket_sizes = []
    bucket_position = []

    max_len = buckets[-1][0]

    batch = []
    data_buckets = [[] for _ in buckets]

    nr_docs = 0

    for data in data_set:
        for i, size in enumerate(buckets):
            if len(data) < buckets[i][0]:
                data_buckets[i].append(data)
                break

    for data_ in data_buckets:
        length = len(data_)
        nr_docs += length
        data_sizes.append(nr_docs)
        bucket_sizes.append(length)
        random.shuffle(data_)
        bucket_position.append(0)

    while True:

        if bucket_sizes == bucket_position:
            yield None, None
            break
        if batch == []:
            while True:
                rand_int = random.randint(1, nr_docs)
                for bucket_nr, size in enumerate(data_sizes):
                    if rand_int < size:
                        break
                if bucket_sizes[bucket_nr] != bucket_position[bucket_nr]:
                    break

        doc_position = bucket_position[bucket_nr]
        token = data_buckets[bucket_nr][doc_position]
        bucket_position[bucket_nr] += 1

        length = len(token)

        if length > max_len:
            length = max_len
            token = token[:length]

        batch.append(token + [EOS_ID])

        if (len(batch) == batch_size) or (bucket_sizes[bucket_nr] == bucket_position[bucket_nr]):
            yield batch, bucket_nr
            batch  = []


def get_embedding_data(location, vocab_size,  all_entities=False):

        pickle_file = 'data/embedding_data/embedding_data4.pickle'

        embedding_data = pickle_call(pickle_file)

        if embedding_data is None:

            embedding_data = {}

            # location = config.location
            embeddings_mentions = {}
            embeddings_mentions_list = []
            embeddings = []


            print("retrieving embeddings from file")

            i = 0

            with open(location, 'r') as f:
                for line in tqdm((f)):

                    add = False

                    if i != 0:

                        chunk = line.split(' ')
                        mention = chunk[0]
                        embedding = np.append(np.array([0.0, 0.0], dtype=np.float64),
                                              np.array([float(j) for j in chunk[1:]], dtype=np.float64))

                        if i == 1:
                            eol = np.zeros(len(embedding), dtype=np.float64)
                            eol[0] = 1.0
                            start = np.zeros(len(embedding), dtype=np.float64)
                            start[1] = 1.0
                            pad = np.zeros(len(embedding), dtype=np.float64)
                            embeddings.append(eol)
                            embeddings.append(start)
                            embeddings.append(pad)
                            embeddings_mentions['!EOL!'] = 0
                            embeddings_mentions['!START!'] = 1
                            embeddings_mentions['!PAD!'] = 2
                            embeddings_mentions_list.append('!EOL!')
                            embeddings_mentions_list.append('!START!')
                            embeddings_mentions_list.append('!PAD!')

                        if i <= vocab_size:
                            embeddings_mentions_list.append(mention)

                            embeddings_mentions[mention] = i + 2
                            embeddings.append(embedding)

                            add = True



                    if i == 0: add = True

                    if add:

                        i += 1

                    if (i == vocab_size and not all_entities):
                        break


            vocab_size = len(embeddings_mentions)
            embedding_dim = len(embeddings[0])
            embeddings_mentions_list = embeddings_mentions_list
            embeddings_mentions = embeddings_mentions
            embeddings_np = np.array(embeddings)

            print(str(vocab_size) + ' embeddings retrieved')

            embedding_data['vocab_size'] = vocab_size
            embedding_data['embedding_dim'] = embedding_dim
            embedding_data['embeddings_mentions_list'] = embeddings_mentions_list
            embedding_data['embeddings_mentions'] = embeddings_mentions
            embedding_data['embeddings_np'] = embeddings_np

            pickle_dump(pickle_file, embedding_data)

        return embedding_data['vocab_size'], embedding_data['embedding_dim'], embedding_data['embeddings_mentions_list'], embedding_data['embeddings_mentions'], embedding_data['embeddings_np']
