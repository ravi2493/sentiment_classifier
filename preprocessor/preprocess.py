import numpy as np
def splitToWords(text):
    if len(text) > 0:
        return text.split()
    else:
        return None

def load_data(paths=None,Id2Vec=None):
    """

    :param paths: list of paths
    :return: dictionary which contains train data,vocab size, maximum sequence length
    """

    if paths == None:
        ## default we load this paths
        paths = []
        paths.append('data/sentiment_data/rt-polarity.pos')
        paths.append('data/sentiment_data/rt-polarity.neg')
    data = []
    max_sequence_length = 0
    word2Id = {}
    word2Id['end'] = 0
    word2Id['unknown'] = 1
    Id2Word = {}
    Id2Word[0] = 'end'
    Id2Word[1] = 'unknown'
    result = {}
    labels = []
    data_count = 0
    classes = 2
    for path in paths:
        f = open(path,'r',encoding='utf8')
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            words = splitToWords(line)
            if len(words) > max_sequence_length:
                max_sequence_length = len(words)
        data_count += len(lines)
    data = np.zeros([data_count,max_sequence_length])
    labels = np.zeros([data_count,classes])
    i = 0
    for path in paths:
        if path.endswith('pos') == True:
            label = 1
        else:
            label = 0
        f = open(path,'r',encoding='utf8')
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            words = splitToWords(line)
            update(word2Id,Id2Word,words)
            data[i,:] = text2Ids(words,word2Id,max_sequence_length)
            labels[i,label] = 1
            i += 1
    print("data loaded successful!!!\n")
    result['data'] = np.array(data)
    result['label'] = np.array(labels)
    result['word2Id'] = word2Id
    result['Id2Word'] = Id2Word
    result['max_sequence_length'] = max_sequence_length
    result['vocab_size'] = len(word2Id.keys())-2
    result['total_classes'] = classes
    return result

def update(word2Id,Id2Word,words):
    keys = list(word2Id.keys())
    for word in words:
        if word not in keys:
            word2Id[word] = len(keys)
            Id2Word[len(keys)] = word
            keys.append(word)

def text2Ids(words,word2Id,max_sequence_len):
    a = np.zeros(max_sequence_len,dtype=int)
    keys = word2Id.keys()
    for i,word in enumerate(words):
        if word in keys:
            a[i] = word2Id[word]
        else:
            a[i] = word2Id['unknown']
    return a
