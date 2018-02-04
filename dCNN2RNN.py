
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
import math
import sys
import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq, sequence_loss_by_example, sequence_loss
import data_utils as data_utils
from data_utils import get_embedding_data, LogFileWriter
from multiprocessing_generator import ParallelGenerator
from tensorflow.python.ops import variable_scope
# from sample_vocab import get_sampled_toy_set
from nn_config import  TestConfigToy
from tqdm import *

def data_type():
  return  tf.float32

BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

class DilatedCNN:


    def __init__(self, config, forward_only, layers):

        self.config = config
        # with tf.device(self.config.gpu):
        self.learning_rate = tf.Variable(
            float(self.config.learning_rate), trainable=False, dtype=self.config.dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * self.config.lr_decay)
        self.global_step = tf.Variable(0, trainable=False)

        self.initializers()

        print( 'building embedding tensor')
        self.build_embedding_tensor()

        # Feeds for inputs.
        self.targets = []
        self.decoder_inputs = []
        # self.encoder_inputs.append([tf.nn.embedding_lookup(self.embeddings, np.full(self.config.batch_size, data_utils.GO_ID))])
        self.target_weights = []
        # self.encoder_inputs_positions = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size,  self.config.buckets[-1][0]])
        self.encoder_inputs_positions = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size,  self.config.cap])
        self.encoder_inputs = [tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs_positions)]

        for i, bucket in enumerate(self.config.buckets):
            # for i in xrange(self.config.buckets[-1][0]):  # Last bucket is the biggest one.
            targets_j = []
            target_weights_j = []
            for j in xrange(bucket[0]):  # Last bucket is the biggest one.
                targets_j.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="targets{0}_{0}".format(i,j)))
                target_weights_j.append(tf.placeholder(self.config.dtype, shape=[None],
                                                          name="weight{0}_{0}".format(i)))
            self.targets.append(targets_j)
            self.target_weights.append(target_weights_j)

        with tf.device(self.config.cpu):

            for targets_j in self.targets:
                decoder_inputs_j = []
                decoder_inputs_j.append(
                    tf.nn.embedding_lookup(self.embeddings, np.full(self.config.batch_size, data_utils.GO_ID)))
                for j, target in enumerate(targets_j):
                    emb_look = tf.nn.embedding_lookup(self.embeddings, target)
                    if j != len(targets_j) - 1 :
                        decoder_inputs_j.append(emb_look)
                self.decoder_inputs.append(decoder_inputs_j)

        # self.filter_ =  tf.get_variable("conv_filter",shape=[1, 3, 1, 1000])

        initial_nr_filter = self.config.hidden_size * self.config.num_layers
        dimension = 3

        initial_shape = [1, dimension, self.config.embedding_dim, initial_nr_filter]

        self.first_w = tf.get_variable("first_w", shape=initial_shape, initializer=tf.contrib.layers.xavier_initializer())
        # self.conv_w = tf_utils.initialize_weights(filter_shape, layer_name + "_w", init_type=initialization,
        #                                 gain=self.nonlinearity, divisor=self.num_classes)
        self.conv0 = tf.nn.conv2d(self.encoder_inputs, self.first_w, strides=[1, 1, 1, 1],
                                            padding='VALID')
        # self.conv = tf.nn.atrous_conv2d(self.encoder_inputs, self.filter_, strides=[1, 1, 1, 1],
        #                                     padding='SAME')


        def r(l):
            return 2 ** (l + 1)

        self.first_b = tf.get_variable( "first_b", initializer=tf.constant(0.01, shape=[initial_nr_filter]))

        self.first_output = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(self.conv0, self.first_b), name="relu"), keep_prob=self.config.keep_prob)


        if self.config.static:
            self.dilated_weight = tf.get_variable("dilated_weight", shape=[1, dimension, initial_nr_filter, initial_nr_filter], initializer=tf.contrib.layers.xavier_initializer())
            self.dilated_biase = tf.get_variable( "dilated_biase", initializer=tf.constant(0.01, shape=[initial_nr_filter]))
            self.dilated_convs = []
            self.encoder_outputs = []
            self.encoder_outputs.append(self.first_output)

            for layer in xrange(0, layers):
                dilation = r(layer)
                # dilation = 1
                # self.dilated_weights.append(tf.get_variable("dilated_weights_" + str(layer), shape=[1, dimension, initial_nr_filter, initial_nr_filter], initializer=tf.contrib.layers.xavier_initializer()))
                self.dilated_convs.append(tf.nn.atrous_conv2d(self.encoder_outputs[-1], self.dilated_weight, rate=dilation, padding="VALID", name='dilated_conv_'+ str(layer)))
                # self.dilated_biases.append(tf.get_variable( "dilated_biases_"+ str(layer), initializer=tf.constant(0.01, shape=[initial_nr_filter])))
                self.encoder_outputs.append(tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(self.dilated_convs[-1], self.dilated_biase), name="dilated_outputs_"+ str(layer)), keep_prob=self.config.keep_prob))
        else:
            self.dilated_weights = []
            self.dilated_convs = []
            self.dilated_biases = []
            self.encoder_outputs = []
            self.encoder_outputs.append(self.first_output)

            for layer in xrange(0, layers):
                dilation = r(layer)
                # dilation = 1
                self.dilated_weights.append(tf.get_variable("dilated_weights_" + str(layer), shape=[1, dimension, initial_nr_filter, initial_nr_filter], initializer=tf.contrib.layers.xavier_initializer()))
                self.dilated_convs.append(tf.nn.atrous_conv2d(self.encoder_outputs[-1], self.dilated_weights[-1], rate=dilation, padding="VALID", name='dilated_conv_'+ str(layer)))
                self.dilated_biases.append(tf.get_variable( "dilated_biases_"+ str(layer), initializer=tf.constant(0.01, shape=[initial_nr_filter])))
                self.encoder_outputs.append(tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(self.dilated_convs[-1], self.dilated_biases[-1]), name="dilated_outputs_"+ str(layer)), keep_prob=self.config.keep_prob))


        softmax_loss_function = None

        self. w_t = tf.get_variable("proj_w", [self.config.vocab_size, self.config.hidden_size], dtype=self.config.dtype)
        self.w = tf.transpose(self.w_t)
        self.b = tf.get_variable("proj_b", [self.config.vocab_size], dtype=self.config.dtype)
        output_projection = (self.w, self.b)

        # self.output_positions = tf.get_variable(name='output_positions', shape=[1, self.config.batch_size],
        #                                         dtype=tf.int64, initializer=self.const_initializer_int_ones,
        #                                         trainable=False)

        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if config.num_samples > 0 and self.config.num_samples < self.config.vocab_size:

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(self.w_t, tf.float32)
                local_b = tf.cast(self.b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=self.config.num_samples,
                        num_classes=self.config.vocab_size),
                    config.dtype)

            softmax_loss_function = sampled_loss

        # Build loop function if for generation
        def _extract_argmax_and_embed(embedding, do_decode, output_projection=None, update_embedding=False, ):
            def loop_function(prev, _):
                if output_projection is not None:
                    prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
                prev_symbol = tf.argmax(prev, 1)
                # Note that gradients will not propagate through the second parameter of
                # embedding_lookup.
                emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
                # self.output_positions = tf.concat([self.output_positions, [prev_symbol]], axis=0)

                if not update_embedding:
                    emb_prev = tf.stop_gradient(emb_prev)
                return emb_prev

            if do_decode:
                return loop_function
            else:
                return None

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size)
        cell = single_cell()

        zero_state = cell.zero_state(config.batch_size, tf.float32)
        self.state_in = tf.identity(zero_state, name='state_in')

        # based on https://medium.com/@erikhallstrm/using-the-tensorflow-multilayered-lstm-api-f6e7da7bbe40#.zhg4zwteg
        # self.state_in = tf.constant(np.zeros((args.num_layers, 2, args.batch_size, args.rnn_size)), dtype=tf.float32, name='state_in')
        state_per_layer_list = tf.unstack(self.state_in, axis=0)

        # self.initial_states = [tuple(
        #     tf.contrib.rnn.LSTMStateTuple( state_per_layer_list[0][idx], self.outputs[-1][0][idx][0])
        #          )for idx in xrange(self.config.batch_size)]


        #
        # self.initial_states = tuple(
        #     [tf.contrib.rnn.LSTMStateTuple(tf.reshape(tf.squeeze(self.encoder_outputs[-1]), [self.config.batch_size, self.config.num_layers, -1])[:,idx,:], state_per_layer_list[idx])
        #     for idx in range(self.config.num_layers)]
        # )

        self.initial_states = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx], tf.reshape(tf.squeeze(self.encoder_outputs[-1]), [self.config.batch_size, self.config.num_layers, -1])[:,idx,:])
            for idx in range(self.config.num_layers)]
        )





        # self.state_in_tuple = tuple(
        #     [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
        #     for idx in range(self.config.num_layers)]
        # )

        # self.initial_states = tf.squeeze(self.outputs[-1])

        if not forward_only and self.config.cell_keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=self.config.cell_keep_prob)

        if self.config.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.config.num_layers)])

        def seq_decoder(initial_state, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.rnn_decoder(
                decoder_inputs,
                initial_state,
                cell,
                loop_function = _extract_argmax_and_embed(self.embeddings, do_decode, output_projection=output_projection),
                # data_type = self.config.dtype
            )

        if forward_only:
            # self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            self.outputs, self.losses = self.decoder_with_buckets(self.initial_states,
                 self.decoder_inputs, self.targets,
                self.target_weights, self.config.buckets, lambda x, y: seq_decoder(x, y, True),
                softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in xrange(len(self.config.buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]

        else:
            # self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            self.outputs, self.losses = self.decoder_with_buckets(self.initial_states,
                 self.decoder_inputs, self.targets,
                self.target_weights, self.config.buckets,
                lambda x, y: seq_decoder(x, y, self.config.teacher_forcing),
                softmax_loss_function=softmax_loss_function)

            # Gradients and SGD update operation for training the model.

        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            opt = tf.train.AdamOptimizer() #self.learning_rate)
            # opt = tf.train.AdagradOptimizer(self.learning_rate, )
            for b in xrange(len(self.config.buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 self.config.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

    def build_embedding_tensor(self):
        # with tf.device(self.cpu):
        with tf.device(self.config.cpu):
            # self.embeddings = tf.Variable(tf.constant(0.0, shape=[self.config.vocab_size, self.config.embedding_dim]),
            #                 trainable=False, name="W")
            with tf.name_scope('embedding'):
                self.embeddings = tf.get_variable(name="Embeddings",
                                                  shape=[self.config.vocab_size, self.config.embedding_dim],
                                                  dtype=data_type(), initializer=self.const_initializer,
                                                  trainable=False)
                self.embedding_placeholder = tf.placeholder(data_type(),
                                                            [self.config.vocab_size, self.config.embedding_dim])
                self.embedding_init = self.embeddings.assign(self.embedding_placeholder)


    def initializers(self):
        self.const_initializer = tf.constant_initializer([0.0], dtype=data_type())
        self.const_initializer_int = tf.constant_initializer([0], dtype=tf.int32)
        self.const_initializer_int_ones = tf.constant_initializer([1], dtype=tf.int32)



    def decoder_with_buckets(self,
                            initial_states, decoder_inputs,
                           targets,
                           weights,
                           buckets,
                           decoder,
                           softmax_loss_function=None,
                           per_example_loss=False,
                           name=None):
        """Create a sequence-to-sequence model with support for bucketing.
        The seq2seq argument is a function that defines a sequence-to-sequence model,
        e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
            x, y, rnn_cell.GRUCell(24))
        Args:
          encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
          decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
          targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
          weights: List of 1D batch-sized float-Tensors to weight the targets.
          buckets: A list of pairs of (input size, output size) for each bucket.
          seq2seq: A sequence-to-sequence model function; it takes 2 input that
            agree with encoder_inputs and decoder_inputs, and returns a pair
            consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
          softmax_loss_function: Function (labels, logits) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
            **Note that to avoid confusion, it is required for the function to accept
            named arguments.**
          per_example_loss: Boolean. If set, the returned loss will be a batch-sized
            tensor of losses for each sequence in the batch. If unset, it will be
            a scalar with the averaged loss from all examples.
          name: Optional name for this operation, defaults to "model_with_buckets".
        Returns:
          A tuple of the form (outputs, losses), where:
            outputs: The outputs for each bucket. Its j'th element consists of a list
              of 2D Tensors. The shape of output tensors can be either
              [batch_size x output_size] or [batch_size x num_decoder_symbols]
              depending on the seq2seq model used.
            losses: List of scalar Tensors, representing losses for each bucket, or,
              if per_example_loss is set, a list of 1D batch-sized float Tensors.
        Raises:
          ValueError: If length of encoder_inputs, targets, or weights is smaller
            than the largest (last) bucket.
        """
        # if len(encoder_inputs) < buckets[-1][0]:
        #     raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
        #                      "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
        # if len(targets) < buckets[-1][1]:
        #     raise ValueError("Length of targets (%d) must be at least that of last "
        #                      "bucket (%d)." % (len(targets), buckets[-1][1]))
        # if len(weights) < buckets[-1][1]:
        #     raise ValueError("Length of weights (%d) must be at least that of last "
        #                      "bucket (%d)." % (len(weights), buckets[-1][1]))

        # with tf.device(self.config.gpu):
        all_inputs = initial_states, decoder_inputs + targets + weights
        losses = []
        outputs = []
        with tf.name_scope(name, "decoder_with_buckets", all_inputs):
            for j, bucket in enumerate(buckets):
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(),  reuse=True if j > 0 else None):
                    bucket_outputs, _ = decoder(initial_states,
                                                decoder_inputs[j])
                    outputs.append(bucket_outputs)
                    if per_example_loss:
                        losses.append(
                            sequence_loss_by_example(
                                outputs[-1],
                                targets[j],
                                weights[j],
                                softmax_loss_function=softmax_loss_function))
                    else:
                        losses.append(
                            sequence_loss(
                                outputs[-1],
                                targets[j],
                                weights[j],
                                softmax_loss_function=softmax_loss_function))

        return outputs, losses



    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):

        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        # print('bucket id = ' + str(bucket_id))
        _, decoder_size = self.config.buckets[bucket_id]

        # if len(encoder_inputs) != encoder_size:
        #     raise ValueError("Encoder length must be equal to the one in bucket,"
        #                      " %d != %d." % (len(encoder_inputs), encoder_size))
        # # if len(decoder_inputs) != decoder_size:
        # #     raise ValueError("Decoder length must be equal to the one in bucket,"
        # #                      " %d != %d." % (len(decoder_inputs), decoder_size))
        # if len(target_weights) != decoder_size:
        #     raise ValueError("Weights length must be equal to the one in bucket,"
        #                      " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.encoder_inputs_positions] = np.array(encoder_inputs).T
        # for l in xrange(encoder_size):
        for l in xrange(decoder_size):
            input_feed[self.targets[bucket_id][l].name] = decoder_inputs[l]
            # input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[bucket_id][l].name] = target_weights[l]




        # pass

        # Since our targets are decoder inputs shifted by one, we need one more.
        # last_target = self.decoder_inputs[decoder_size].name
        # input_feed[last_target] = np.zeros([self.config.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self, data, cap):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        # encoder_size, decoder_size = self.config.buckets[bucket_id]
        encoder_size = cap
        encoder_inputs = []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for d in data:
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(d))
            encoder_inputs.append(list(d + encoder_pad))

        while len(encoder_inputs) < self.config.batch_size:
            encoder_inputs.append([data_utils.PAD_ID] * encoder_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_weights = [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            try:
                batch_encoder_inputs.append(
                    np.array([encoder_inputs[batch_idx][length_idx]
                              for batch_idx in xrange(self.config.batch_size)], dtype=np.int32))
            except:
                pass

            batch_weight = np.ones(self.config.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.config.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if encoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        # batch_weights.append(np.zeros(self.config.batch_size, dtype=np.float32))

        if np.array(batch_encoder_inputs).size != np.array(batch_weights).size:
            pass

        return batch_encoder_inputs, batch_weights




def create_model(session, config, forward_only, layers):
    """Create translation model and initialize or load parameters in session."""

    # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = DilatedCNN(config, forward_only, layers)
    ckpt = tf.train.get_checkpoint_state(config.train_dir)
    # print (ckpt.model_checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        # model.load_embeddings(session, embeddings)

    writer = tf.summary.FileWriter('dil_CNN', session.graph)
    return model

def needed_layers(config):
    tokens = config.buckets[-1][0]
    def tokens_captured(l):
        return 2 ** (l + 1) - 1
    layer = 1
    while True:
        cap = tokens_captured(layer)
        if cap >= tokens:
            return layer - 1 , cap
        layer += 1

def train(config, forward_only, train_log_file, test_log_file):
    """Train a en->fr translation model using WMT data."""
    sess_config = tf.ConfigProto(
        # allow_soft_placement=True,
        # log_device_placement=True
    )
    sess_config.gpu_options.allocator_type = 'BFC'
    # config.gpu_options.per_process_gpu_memory_fraction = 0.99
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        # with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (config.num_layers, config.hidden_size))

        vocab_size = config.vocab_size
        print('getting embeddings')
        # embeddings_mentions, embeddings, embeddings_mentions_list = data_utils.get_embedding_data(location, vocab_size)

    # train_set, test_set = doc_bucket_seperator(mongodb_name, _buckets, embeddings_mentions, train_test_perc=0.80,
    #                                            nr_docs=100)

    # config.vocab_size = len(embeddings)
    # config.embedding_dim = len(embeddings[0])




    layers, cap = needed_layers(config)
    config.cap = cap
    # checkpoint_counter = 0.0

    print('building model')
    model = create_model(sess,config, forward_only, layers)


    sess.run(model.embedding_init, {model.embedding_placeholder: model.config.embeddings_np})

    # train_set, dev_set = data_utils.bucket_generator(mongodb_name, config.buckets, max_len, embeddings_mentions, train_length,
    #                                                  test_length)

    # train_bucket_sizes = [len(train_set[b]) for b in xrange(len(config.buckets))]
    # train_total_size = float(sum(train_bucket_sizes))
    # # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # # the size if i-th training bucket, as used later.
    # train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
    #                        for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, single_step_time, loss, single_step_loss = 0.0, 0.0, 0.0, 0.0
    current_step = 0
    previous_losses = []

    print('starting training')
    for epoch in xrange(config.epoch_size):
        train_log_file.append_text('Epoch: ' + str(epoch))
        test_log_file.append_text('Epoch: ' + str(epoch))

        # generator = new_bucket_generator(mongodb_name, config.buckets, train_set, embeddings_mentions, 4)
        generator = config.generator(*config.train_args)

        # while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.

            # batch, bucket_id = generator.next()

        with ParallelGenerator(generator, max_lookahead=10) as batch_gen_lookahead:
            for batch, bucket_id in batch_gen_lookahead:

                if batch is None: break

                # Get a batch and make a step.
                start_time = time.time()

                # encoder_inputs,  target_weights = model.get_batch(batch, cap)

                encoder_inputs, _ = model.get_batch(batch, cap)
                decoder_inputs, target_weights = model.get_batch(batch, config.buckets[bucket_id][0])

                _, step_loss, _ = model.step(sess, encoder_inputs,decoder_inputs, target_weights, bucket_id, False)
                step_time += (time.time() - start_time) / config.steps_per_checkpoint
                # single_step_time += (time.time() - start_time) / (config.steps_per_checkpoint / 100)
                loss += (step_loss / config.steps_per_checkpoint)
                # checkpoint_counter += 1
                # single_step_loss += step_loss / (config.steps_per_checkpoint / 100)
                current_step += 1


                # Once in a while, we save checkpoint, print statistics, and run evals.
                #
                # if current_step % config.steps_per_checkpoint == 0:
                #     perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                #     print("global step %d learning rate %.4f step-time %.2f perplexity "
                #           "%.2f" % (model.global_step.eval(session=sess), model.learning_rate.eval(session=sess),
                #                     step_time, perplexity))
                #     # Decrease learning rate if no improvement was seen over last 3 times.
                #     if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                #         sess.run(model.learning_rate_decay_op)
                #     previous_losses.append(loss)
                #     # Save checkpoint and zero timer and loss.
                #     checkpoint_path = os.path.join(config.train_dir, "translate.ckpt")
                #     model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                #     # Run evals on development set and print their perplexity.
                #
                #
                #
                # if current_step % (config.steps_per_checkpoint / 100) == 0:
                #     # Print statistics for the previous epoch.
                #     perplexity = math.exp(float(single_step_loss)) if single_step_loss < 300 else float("inf")
                #     print("global step %d learning rate %.4f step-time %.2f perplexity "
                #           "%.2f" % (model.global_step.eval(session=sess), model.learning_rate.eval(session=sess),
                #                     single_step_time, perplexity))
                #     single_step_time, single_step_loss = 0.0, 0.0


                if current_step % config.steps_per_checkpoint == 0:
                    # loss = loss / checkpoint_counter
                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print("global step %d learning rate %.4f step-time %.2f perplexity "
                          "%.2f" % (model.global_step.eval(session=sess), model.learning_rate.eval(session=sess),
                                    step_time, perplexity))
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    #     sess.run(model.learning_rate_decay_op)
                    # previous_losses.append(loss)
                    train_log_file.append_text(str(model.global_step.eval(session=sess)) + ';' + str(perplexity) + ';' + str(loss) + ';' + str(model.learning_rate.eval(session=sess)) + ';' + str(step_time))
                    step_time, loss = 0.0, 0.0


                    # Save checkpoint and zero timer and loss.
                    # checkpoint_path = os.path.join(config.train_dir, "translate.ckpt")
                    # model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    # Run evals on development set and print their perplexity.



                # if current_step % (config.steps_per_checkpoint / 100) == 0:
                #     # Print statistics for the previous epoch.
                #     perplexity = math.exp(float(single_step_loss)) if single_step_loss < 300 else float("inf")
                #     print("global step %d learning rate %.4f step-time %.2f perplexity "
                #           "%.2f" % (model.global_step.eval(session=sess), model.learning_rate.eval(session=sess),
                #                     single_step_time, perplexity))
                #
                #     single_step_time, single_step_loss = 0.0, 0.0

        checkpoint_path = os.path.join(config.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        test_generator = config.generator(*config.test_args)

        print("Testing:")
        with ParallelGenerator(test_generator, max_lookahead=10) as batch_gen_lookahead:
            for batch, bucket_id in batch_gen_lookahead:
                if batch == None: break

                # encoder_inputs, target_weights = model.get_batch(batch, cap)


                encoder_inputs, _ = model.get_batch(batch, cap)
                decoder_inputs, target_weights = model.get_batch(batch, config.buckets[bucket_id][0])


                _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                    "inf")
                print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

                test_log_file.append_text( str(bucket_id)  + ';' + str(eval_ppx) + ';' + str(eval_loss))

        # step_time, loss = 0.0, 0.0
        sys.stdout.flush()



def decode(config):

    layers, cap = needed_layers(config)
    config.cap = cap

    with tf.Session() as sess:
        # Create model and load parameters.

        model = create_model(sess, config, True, layers)
        model.batch_size = 1  # We decode one sentence at a time.

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), config.embeddings_mentions)
            # Which bucket does it belong to?
            bucket_id = len(config.buckets) - 1
            for i, bucket in enumerate(config.buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            # else:
            #   logging.warning("Sentence truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the model.
            # encoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
            encoder_inputs, _ = model.get_batch([token_ids], cap)
            decoder_inputs, target_weights = model.get_batch([token_ids], config.buckets[bucket_id][0])

            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit[0])) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            # print(" ".join([tf.compat.as_str(config.embeddings_mentions_list[output]) for output in outputs]))
            out_str = ""
            for output in outputs:
                try:
                    out_str += config.embeddings_mentions_list[output] + ' '
                except:
                    pass
            print(out_str)

            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()



if __name__ == '__main__':


    path = 'data/dCNN/'

    hidden_sizes = range(10,201,10)

    for hidden_size in hidden_sizes:

        current_path = path + 'hidden_size_' + str(hidden_size) + '/'
        if not os.path.exists(current_path):
            os.makedirs(current_path)

        train_log_file = LogFileWriter(current_path + 'dilated_hidden_size_' + str(hidden_size) + '_train_log.csv')
        test_log_file = LogFileWriter(current_path + 'dilated_hidden_size_' + str(hidden_size) + '_test_log.csv')

        train_config = TestConfigToy()

        train_config.hidden_size = hidden_size
        train_config.train_dir = current_path

        train_log_file.append_text('DILATED CNN')
        train_log_file.append_text('Hidden Size = ' + str(hidden_size))

        test_log_file.append_text('DILATED CNN')
        test_log_file.append_text('Hidden Size = ' + str(hidden_size))



        train(train_config, False, train_log_file, test_log_file)

        tf.reset_default_graph()



        # train_config.batch_size = 1
        # decode(train_config)
        pass




