# Dilated Convolutional Neural Networks as the Encoder for the Sequence2Sequence Model

This code is part of my Master Thesis. I unfortunately have not yet had enough time to
document it for the toy set. 



**General Idea:**

Given text documents we learn word embeddings using gensim.
These word embeddings are subsequently used for autoencoding documents using the 
sequence to sequence model.

The code for RNN2RNN is mostly copied from the original seq2seq model (without attention) 
from tensorflow. The major part that has changed is that we have already learnt
the embeddings previously and use these instead.

The code for dCNN2RNN is a combination of the original seq2seq and the idea of the following paper:
https://arxiv.org/abs/1702.02098 .
The encoder of seq2seq is learnt as a dilated convolutional neural network (dCNN). The output of 
the dCNN is the first cell of the decoder for which currently an LSTM is implemented. 


**Why dCNN?**

RNNs are slow, CNNs are fast... 


**Data:**

The toy data consists of small sentences which are all written in a single file. 
Every line is considered as document. 
We first learn embeddings over all the documents. Because we want to be able to encode documents
and not only sentences, all the sentences are first split up into training, validation and
testing data. Then the sentences in each set are sampled to generate paragraphs in the range
1 - 10 sentences long. 

These new documents are then encoded and decoded using the methods described above.



_I am very sorry for not having documented anything. This is only a small sample of the code
 that I have written for my MT. This is why some functions might seem strange. The retrieval
 of my documents is actually done through mongoDB. The Toy Set is only an implementation
 for my GitHub. 
 I will try to cleanse the code and document it ASAP!!!_
