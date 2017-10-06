import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class model(object):

    def __init__(
            self, max_sequence_length,
                    total_classes,
                    vocab_size,embedding_size,
                    wordVecs, batch_size, lmd = 0
                 ):

        # placeholders
        self.x = tf.placeholder(tf.int32,[batch_size,max_sequence_length],name = "x")
        self.labels = tf.placeholder(tf.int32,[None,total_classes],name="labels")
        self.dropout = tf.placeholder(tf.float32,[None])
        self.hidden_Units = 100
        self.batch_size = batch_size
        # tf.split() might be useful

        with tf.device('/cpu:0'):
            # layer contains trainable weights
            with tf.name_scope("embedding-layer"):
                self.embeddings =  tf.Variable(initial_value=wordVecs,name='embedding',dtype=tf.float32)
                self.chars = tf.nn.embedding_lookup(self.embeddings,self.x)
            # tf.split()
            # convert data to slices
            lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)
            l2_loss = tf.constant(value=0.0,dtype=tf.float32)
            #Initial state of the LSTM memory.
            # hidden_state = tf.zeros([self.batch_size,])
            #
            sequence = tf.split(self.chars[0],num_or_size_splits=max_sequence_length,axis=0)
            outputs,state = rnn.static_rnn(lstm_cell,sequence,dtype=tf.float32)
            # outputs - is a length T list of outputs (one for each input), or a nested tuple of such elements.
            # state represents final state

            with tf.name_scope("last-layer"):
                self.W_f = tf.get_variable(
                                            "W_f",
                                            shape=[self.hidden_Units,total_classes],
                                            initializer=tf.contrib.layers.xavier_initializer()
                                            )
                bias = tf.Variable(tf.constant(value=0.01,shape=[total_classes],name="bias"))
                l2_loss += tf.nn.l2_loss(self.W_f)
                l2_loss += tf.nn.l2_loss(bias)
                self.final_score = tf.nn.xw_plus_b(outputs[-1],weights=self.W_f,biases=bias,name="scores")
                self.pred = tf.argmax(self.final_score,1,name="pred")

            with tf.name_scope("loss"):
                l = tf.nn.softmax_cross_entropy_with_logits(logits=self.final_score,labels=self.labels)
                self.loss = tf.reduce_mean(l) + lmd*l2_loss
model(max_sequence_length=10,total_classes=3,vocab_size=97,embedding_size=300,wordVecs=np.zeros([7,300]),batch_size=1)