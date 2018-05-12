import re
import os, sys
import pandas as pd
import fasttext

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from ltp_server.pyltp_server import load_model_and_lexicon, segment, release_model

_ROOTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_ROOTPATH.replace('/monitor', ''))

_DATA_PATH = _ROOTPATH + '/../data/monitor.csv'
_STOP_WORD_PATH = _ROOTPATH + '/dictionary/stopword.txt'


class Fast_Text:
    def __init__(self):
        pass

    def data_preprocessing(self, file_path):
        logging.info('data preprocessing  ......')
        stop_word = self.load_stop_word()
        data = pd.read_csv(file_path)

        load_model_and_lexicon('segmentor')
        for index, row in data.iterrows():
            words = [word for word in segment(re.sub(r'(\\n|\\t|\[.{1,3}\])', ' ', row['expression'])) ]
            data.loc[index]['expression'] = '\t'.join(words)

        data = data['expression'] + "\t__label__" + data['label']

        release_model('segmentor')
        data_train = data.head(int(len(data) * 1))
        data_train.to_csv('news_fasttext_train.csv', header=False, index=False)
        data_test = data.tail(int(len(data) * 0.2))
        data_test.to_csv('news_fasttext_test.csv', header=False, index=False)

    def load_stop_word(self):
        stop_words = []
        with open(_STOP_WORD_PATH) as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())
        return stop_words

    def train(self, train_data_path):
        self.data_preprocessing(train_data_path)
        # 训练模型
        logging.info('training  ......')
        classifier = fasttext.supervised('news_fasttext_train.csv', "news_fasttext.model", label_prefix="__label__")
        result = classifier.test('news_fasttext_test.csv')
        print(result.precision)
        print(result.recall)


    def predict(self, sents):
        stop_word = self.load_stop_word()
        load_model_and_lexicon('segmentor')
        test = ['\t'.join([word for word in segment(sent)]) for sent in sents]
        release_model('segmentor')

        # load训练好的模型
        classifier = fasttext.load_model('news_fasttext.model.bin', label_prefix='__label__')
        label = classifier.predict(test)

        return label


if __name__ == '__main__':
    ft = Fast_Text()
    # ft.train(_DATA_PATH)
    test = ['嗯，周二面试人员赶不上，安排项目沟通吧。', '帮我订酒店吧', '你叫什么名字', '帮我打印1份，交给咸宁公司质管员。她的学习计划。然后再看普总怎么安排工作。']
    label = ft.predict(test)
    print(label)
