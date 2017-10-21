import tensorflow as tf

"""
reference : https://arxiv.org/pdf/1710.05941.pdf
"""
def swish(x):
    """
    :param x:
    :return:
    """
    return x*tf.sigmoid(x)