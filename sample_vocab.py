
import numpy as np
import re


def get_embedding_data(location, vocab_size):
        embeddings_mentions = {}
        embeddings_mentions_list = []
        embeddings = []

        print("retrieving embeddings from file")

        with open(location, 'r') as f:
            for i, line in (enumerate(f)):

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

                    embeddings_mentions_list.append(mention)

                    embeddings_mentions[mention] = i + 2
                    embeddings.append(embedding)
                if i == vocab_size:
                    break
        vocab_size = len(embeddings_mentions)
        embedding_dim = len(embeddings[0])
        embeddings_mentions_list = embeddings_mentions_list
        embeddings_mentions = embeddings_mentions
        embeddings_np = np.array(embeddings)

        print('embeddings retrieved')

        return vocab_size, embedding_dim, embeddings_mentions_list, embeddings_mentions, embeddings_np



def tokenize(text):
    punctuation = "!\"#$%&\\'()*+,\-./:;<=>?@[\]\^\`\{\|\}~\"'\xe2\x80\x99"
    mentions = list(re.findall("\s*([" + punctuation + "]|[^" + punctuation + "\s]*)\s*", text))
    return mentions


def build_new_vocab_set(new_vocab_list, embeddings_mentions, embeddings_np):

        new_embeddings_mentions = {}
        new_embeddings_mentions_list = []
        new_embeddings = []

        eol = np.zeros(len(embeddings_np[0]), dtype=np.float64)
        eol[0] = 1.0
        start = np.zeros(len(embeddings_np[0]), dtype=np.float64)
        start[1] = 1.0
        pad = np.zeros(len(embeddings_np[0]), dtype=np.float64)
        new_embeddings.append(eol)
        new_embeddings.append(start)
        new_embeddings.append(pad)
        new_embeddings_mentions['!EOL!'] = 0
        new_embeddings_mentions['!START!'] = 1
        new_embeddings_mentions['!PAD!'] = 2
        new_embeddings_mentions_list.append('!EOL!')
        new_embeddings_mentions_list.append('!START!')
        new_embeddings_mentions_list.append('!PAD!')

        i = len(new_embeddings_mentions_list)

        for mention in new_vocab_list:
            position = embeddings_mentions[mention]
            embedding = embeddings_np[position]

            new_embeddings.append(embedding)
            new_embeddings_mentions[mention] = i
            new_embeddings_mentions_list.append(mention)

            i+= 1


        vocab_size = len(new_embeddings_mentions_list)
        embedding_dim = len(new_embeddings[0])
        embeddings_mentions_list = new_embeddings_mentions_list
        embeddings_mentions = new_embeddings_mentions
        embeddings_np = np.array(new_embeddings)

        return vocab_size, embedding_dim, embeddings_mentions_list, embeddings_mentions, embeddings_np



