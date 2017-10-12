import tensorflow as tf
import numpy as np
from model import model
from glove import word_embedings
from preprocessor.preprocess import load_data
import random
print("started")
def batch_iter(data, batch_size, epochs, Isshuffle=True):
    ## check inputs
    assert isinstance(batch_size,int)
    assert isinstance(epochs,int)
    assert isinstance(Isshuffle,bool)

    num_batches = int((len(data)-1)/batch_size) + 1
    ## data padded
    data = np.array(data+data[:2*batch_size])
    data_size = len(data)
    print("size of data"+str(data_size)+"---"+str(len(data)))
    for ep in range(epochs):
        if Isshuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            yield shuffled_data[start_index:end_index]


def train(m,data,label,epochs=200,learning_rate=0.001):
    assert isinstance(m,model)
    assert isinstance(epochs,int)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(m.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    sess = tf.Session(config=session_conf)

    ## intialize
    sess.run(tf.global_variables_initializer())
    dt = np.zeros([len(data),len(data[0])],dtype=int)
    for i,d in enumerate(data):
        dt[i,:] = d
    total_data = list(zip(dt,label))
    batches = batch_iter(total_data,batch_size=60,epochs=epochs,Isshuffle=True)
    ## run the graph
    print("\n")
    i = 0
    for batch in batches:
        x,y = zip(*batch)
        x = np.array(x)
        y = np.array(y)
        feed_dict = {
            m.x: x,
            m.labels: y
            # m.dropout : 0.5
        }
        _,loss,accuracy = sess.run([train_op,m.loss,m.acc],feed_dict=feed_dict)
        print("step - "+str(i)+"    loss is " + str(loss)+"and accuracy is "+str(accuracy)+"\n")
        i += 1
    return sess

def test(m,data,label,sess):
    feed_dict ={
        m.x : data,
        m.labels : label
        # m.dropout : 0.5
    }

    loss, accuracy = sess.run([ m.loss, m.acc], feed_dict=feed_dict)
    print("test accuracy:  "+str(accuracy))
    return accuracy
# debug = True  ---> only loads 200 lines of glove file
debug = True
# this will load data from default path
word_vecs = word_embedings(debug=debug)

batch_size = 60
embedding_size = 300
res = load_data()
data = res['data']
label = res['label']
word2Id = res['word2Id']
Id2Word  = res['Id2Word']
max_sequence_length = res['max_sequence_length']
total_classes = res['total_classes']
Id2Vec = np.zeros([len(Id2Word.keys()),embedding_size])
words_list = word_vecs.word2vec.keys()
for i in range(len(Id2Word.keys())):
    word = Id2Word[i]
    if word in words_list:
        Id2Vec[i,:] = word_vecs.word2vec[word]
    else:
        Id2Vec[i, :] = word_vecs.word2vec['unknown']

lstm = model(
        max_sequence_length=res['max_sequence_length'],
        total_classes=res['total_classes'],
        vocab_size=res['vocab_size'],
        embedding_size = embedding_size,
        id2Vecs = Id2Vec,
        batch_size=batch_size
    )

## split data to train and test
n = len(data)
q = 0.15 # ratio of test and train
test_data_len = int(q*n)
a = random.sample(range(1,n),int(q*n))
test_data = np.zeros([test_data_len,max_sequence_length])
test_labels = np.zeros([test_data_len,total_classes])
for i,e in enumerate(a):
    test_data[i,:] = data[e,:]
    test_labels[i,:] = label[e,:]
train_data_len = n - test_data_len
train_data = np.zeros([train_data_len,max_sequence_length])
train_labels = np.zeros([train_data_len,total_classes])
for i,e in range(n):
    if e not in a:
        train_data = data[e,:]
        train_labels = label[e,:]

train(lstm,train_data,train_labels)