# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC


stopwordsfiles = "../svm_data/stop_words.txt"
segment_dict = "../svm_data/segment_dict.txt"

# 导入停顿词
with open(stopwordsfiles, 'r') as fr:
    lines = fr.readlines()
    stopwords = [word for word in lines]

# 导入分词
with open(segment_dict, 'r') as fr:
    lines = fr.readlines()
    for line in lines:
        jieba.add_word(line.strip())


def deleteStopWords(words):
    if isinstance(words, list):
        assert TypeError

    new_words = []
    for word in words:
        if not word in stopwords:
            new_words.append(word)
    return new_words


# 加载文件，导入数据,分词
def loadfile():
    meeting_data = pd.read_csv("../data/meeting_train.csv")
    not_meeting_data = pd.read_csv("../data/not_meeting_train.csv")
    test_data = pd.read_csv("../data/test.csv")

    cw = lambda x: list(jieba.cut(str(x).split("_time_")[0]))
    meeting_words = list(meeting_data['content'].apply(cw))
    not_meeting_words = list(not_meeting_data['content'].apply(cw))
    x_test = list(test_data['content'].apply(cw))

    meeting_words = [deleteStopWords(words) for words in meeting_words]
    not_meeting_words = [deleteStopWords(words) for words in not_meeting_words]
    x_test = [deleteStopWords(words) for words in x_test]

    y_train = ['IsMeetingSchedule'] * len(meeting_words) + ['NotMeetingSchedule'] * len(not_meeting_words)
    x_train = meeting_words + not_meeting_words
    y_test = list(test_data['label'])
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    np.save('../svm_data/y_train.npy', y_train)
    np.save('../svm_data/y_test.npy', y_test)
    return x_train, x_test


# 对每个句子的所有词向量取均值
def buildWordVector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算词向量
def get_train_vecs(x_train, x_test):
    n_dim = 64
    # Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=2)
    imdb_w2v.build_vocab(x_train)

    # Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.epochs)

    train_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in x_train])
    # train_vecs = scale(train_vecs)

    np.save('../svm_data/train_vecs.npy', train_vecs)
    print(train_vecs.shape)
    # Train word2vec on test tweets
    imdb_w2v.train(x_test, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.epochs)
    imdb_w2v.save('../svm_data/w2v_model/w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in x_test])
    # test_vecs = scale(test_vecs)
    np.save('../svm_data/test_vecs.npy', test_vecs)
    print(test_vecs.shape)


def get_data():
    train_vecs = np.load('../svm_data/train_vecs.npy')
    y_train = np.load('../svm_data/y_train.npy')
    test_vecs = np.load('../svm_data/test_vecs.npy')
    y_test = np.load('../svm_data/y_test.npy')
    return train_vecs, y_train, test_vecs, y_test


##训练svm模型
def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf = SVC(kernel='rbf', verbose=True, C=1)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, '../svm_data/svm_model/model.pkl')
    print(clf.score(test_vecs, y_test))

def svm_cross_validation(train_x, train_y, test_vecs, y_test):
    from sklearn.grid_search import GridSearchCV

    model = SVC(probability=True)
    param_grid = [
        {'kernel': ['rbf'],
         'C': [1, 10, 100, 1000],
         'gamma': [1e-3, 1e-4]},

        {'kernel': ['linear'],
         'C': [1, 10, 100, 1000]}
    ]
    grid_search = GridSearchCV(model, param_grid, n_jobs=2, verbose=1, cv=5)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    joblib.dump(model, '../svm_data/svm_model/model_grid.pkl')
    print(model.score(test_vecs, y_test))


##得到待预测单个句子的词向量
def get_predict_vecs(words):
    n_dim = 64
    imdb_w2v = Word2Vec.load('../svm_data/w2v_model/w2v_model.pkl')
    # imdb_w2v.train(words)
    train_vecs = buildWordVector(words, n_dim, imdb_w2v)
    # print train_vecs.shape
    return train_vecs


####对单个句子进行情感判断
def svm_predict(string):
    words = deleteStopWords(jieba.lcut(string))
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('../svm_data/svm_model/model_grid.pkl')
    result = clf.predict(words_vecs)
    return result


def predict_file():
    test_data = pd.read_csv("../data/test.csv")

    errors = []
    error_labels = []
    error_count = 0

    for index, row in test_data.iterrows():
        re = svm_predict(row['content'].split("_time_")[0])
        if re == ['IsMeetingSchedule']:
            if not "会" in row['content']:
                re = ['NotMeetingSchedule']

        if not re == [row['label']]:
            print(error_count, row['label'], row['content'])
            error_count += 1

            errors.append(row['content'])
            error_labels.append(row['label'])

    neg_pd = pd.DataFrame({'content': errors, 'label': error_labels})
    neg_pd.to_csv("../data/error_re.csv", index=False, quoting=1)

def not_meeting():
    pass


if __name__ == '__main__':
    ##导入文件，处理保存为向量
    # x_train,x_test = loadfile() #得到句子分词后的结果，并把类别标签保存为y_train。npy,y_test.npy
    # get_train_vecs(x_train,x_test) #计算词向量并保存为train_vecs.npy,test_vecs.npy
    # train_vecs,y_train,test_vecs,y_test = get_data()#导入训练数据和测试数据
    # # svm_train(train_vecs,y_train,test_vecs,y_test)#训练svm并保存模型
    # svm_cross_validation(train_vecs, y_train, test_vecs, y_test)

    ##对输入句子情感进行判断
    string = '下周三，元华超市 定海店、开鲁店开业，先给你一个陈柳敏吧'
    # string='东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'

    print(svm_predict(string))
    predict_file()

