import pandas as pd
import jieba
import pickle
import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from meetingschedule.lstm.config import TLSTMConfig

config = TLSTMConfig()

w2v_model_path = './model/w2v_model.pkl'


def train_word2vec(w2v_data):
    print("train word2vec...")
    model = Word2Vec(w2v_data, size=config.embedding_dim, min_count=5)
    model.save(w2v_model_path)


def read_data():
    meeting_data = pd.read_csv("../../data/train_data/meeting_schedule.csv")
    task_data = pd.read_csv("../../data/train_data/task_schedule.csv")
    schedule_data = pd.read_csv("../../data/train_data/schedule.csv")
    notschedule_data = pd.read_csv("../../data/train_data/not_schedule.csv")
    test_data = pd.read_csv("../../data/train_data/schedule_testset.csv")


    def cw(data):
        result_l = []
        for row in data:
            sentence = row.replace('\n', '')
            result_l.append(list(jieba.cut(sentence)))
        return result_l

    meeting_data_words = cw(meeting_data['content'])
    task_data_words = cw(task_data['content'])
    schedule_data_words = cw(schedule_data['content'])
    notschedule_data_words = cw(notschedule_data['content'][:6000])
    test_data_words = cw(test_data['content'])


    print("meeting_data", len(meeting_data_words))
    print("task_data_words", len(task_data_words))
    print("schedule_data_words", len(schedule_data_words))
    print("notschedule_data_words", len(notschedule_data_words))
    print("test:", len(test_data_words))

    texts = meeting_data_words + task_data_words + schedule_data_words + notschedule_data_words
    labels = [3] * len(meeting_data_words) \
             + [2] * len(task_data_words) \
             + [1] * len(schedule_data_words) \
             + [0] * len(notschedule_data_words)

    text_len = len(texts)
    indices = np.random.permutation(np.arange(text_len))

    x_shuffle = list(np.array(texts)[indices])
    y_shuffle = list(np.array(labels)[indices])

    test_x = test_data_words
    test_y = []
    labeles = ['IsMeetingSchedule', 'TaskSchedule', 'IsSchedule', 'NotSchedule']
    for label in test_data['label']:
        index = labeles.index(label)
        test_y.append(index)


    # train_texts, test_texts, train_labels, test_labels = train_test_split(x_shuffle, y_shuffle, test_size=0, random_state=0)

    return x_shuffle, y_shuffle, test_x, test_y

def pad_seq(sentences):
    data = [list(jieba.cut(sentence)) for sentence in sentences]
    with open('./model/task_tokenizer.pkl', 'rb') as fr:
        tokenizer = pickle.load(fr)
        sequences = tokenizer.texts_to_sequences(data)
        data = pad_sequences(sequences, maxlen=config.seq_length)
        return data

def text_ready():
    train_texts, train_labels, test_texts, test_labels = read_data()
    texts = train_texts + test_texts
    labels = train_labels + test_labels
    tokenizer = Tokenizer(num_words=config.vocab_size)
    tokenizer.fit_on_texts(texts)
    if not os.path.exists("./model"):
        os.makedirs("./model")
    with open('./model/task_tokenizer.pkl', 'wb') as fw:
        pickle.dump(tokenizer, fw)

    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=config.seq_length)
    print("labels_text:", labels[:10])
    labels = to_categorical(np.asarray(labels))
    print("labels:", labels[:10])

    # 切割数据
    x_train = data[:len(train_texts)]
    y_train = labels[:len(train_labels)]
    x_val = data[len(train_texts):]
    y_val = labels[len(train_labels):]

    return x_train, y_train, x_val, y_val

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]  # shuffle 重新洗牌
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__ == '__main__':

    train_texts, train_labels, test_texts, test_labels = read_data()
    text_ready(train_texts, train_labels, test_texts, test_labels)

