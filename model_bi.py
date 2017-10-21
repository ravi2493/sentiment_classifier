import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from swish_activation import swish

class model_bi(object):

    def __init__(
            self, max_sequence_length,
                    total_classes,
                    vocab_size,
                    embedding_size,
                    id2Vecs,
                    batch_size,
                    threshold=0.5,
                    lmd = 0
                 ):

        # placeholders
        self.x = tf.placeholder(tf.int32,[batch_size,max_sequence_length],name = "x")
        self.labels = tf.placeholder(tf.int32,[batch_size,total_classes],name="labels")
        self.dropout = tf.placeholder(tf.float32,name="dropout")
        self.hidden_Units = 100
        self.batch_size = batch_size
        # tf.split() might be useful

        self.threshold = threshold

        with tf.device('/cpu:0'):
            # layer contains trainable weights
            with tf.name_scope("embedding-layer"):
                # self.embeddings =  tf.get_variable(initializer=get_initializer(id2Vecs),
                #                                    name='embedding_lookup',
                #                                           shape=[vocab_size,embedding_size],dtype=tf.float32)
                self.embeddings = tf.Variable(initial_value=id2Vecs,dtype=tf.float32,name='embedding_lookup')
                self.chars = tf.nn.embedding_lookup(self.embeddings,self.x)
            # tf.split()
            # convert data to slices
            right_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units,activation=swish)
            left_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units,activation=swish)

            right_lstm_cell = rnn.DropoutWrapper(right_lstm_cell,output_keep_prob=self.dropout)
            left_lstm_cell = rnn.DropoutWrapper(left_lstm_cell,output_keep_prob=self.dropout)
            l2_loss = tf.constant(value=0.0,dtype=tf.float32)
            #Initial state of the LSTM memory.
            # hidden_state = tf.zeros([self.batch_size,])
            #
            self.sequence = tf.split(self.chars,num_or_size_splits=max_sequence_length,axis=1)

            self.outputs,self.state = tf.nn.bidirectional_dynamic_rnn(left_lstm_cell,right_lstm_cell,self.chars,dtype=tf.float32)
            combined_output = tf.concat(self.outputs, axis=2)

            out = tf.reshape(combined_output,shape=[batch_size,max_sequence_length*self.hidden_Units*2])

            with tf.name_scope("last-layer"):
                self.W_f = tf.get_variable(
                                            "W_f",
                                            shape=[2*max_sequence_length*self.hidden_Units,total_classes],
                                            initializer=tf.contrib.layers.xavier_initializer()
                                            )
                bias = tf.Variable(tf.constant(value=0.01,shape=[total_classes],name="bias"))
                l2_loss += tf.nn.l2_loss(self.W_f)
                l2_loss += tf.nn.l2_loss(bias)
                self.final_score = tf.nn.xw_plus_b(out,weights=self.W_f,biases=bias,name="scores")
                # self.pred = tf.cast(tf.greater(self.final_score,self.threshold*tf.ones_like(self.final_score),name="pred"),tf.int32)
                self.pred = tf.argmax(self.final_score,1,name="pred")
            with tf.name_scope("loss"):
                l = tf.nn.softmax_cross_entropy_with_logits(logits=self.final_score,labels=self.labels)
                self.loss = tf.reduce_mean(l) + lmd*l2_loss

            with tf.name_scope("accuracy"):
                self.corr_pred = tf.equal(self.pred,tf.argmax(self.labels,1))
                self.acc = tf.reduce_mean(tf.cast(self.corr_pred,"float"),name="accuracy")
# model(max_sequence_length=11,total_classes=5,vocab_size=97,embedding_size=300,id2Vecs=np.zeros([7,300]),batch_size=3)