from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import randomized_svd
import numpy as np

class LSA():
    def __init__(self, doc_ls, topic_num):
        self.doc_ls = doc_ls
        self.topic_num = topic_num

    def DTM(self):
        count_vect = CountVectorizer()
        self.dtm = count_vect.fit_transform(self.doc_ls)

        self.words = count_vect.get_feature_names()

    def SVD(self):
        self.U, self.s, self.VT = randomized_svd(self.dtm, n_components=self.topic_num, n_iter=10, random_state=None)
                              
    def topic_modeling(self, word_num):
        self.word_num = word_num
        self.DTM()
        self.SVD()

        self.idx = self.VT.argsort() # 토픽별 단어의 중요도를 오름차순으로 정렬해 인덱스로 저장

        self.topic_words = []
        for topic in self.idx:
            tmp = []
            for word in topic[::-1][:self.word_num]: # 내림차순으로 바꾸고 word_num만큼의 단어 추출
                tmp.append(self.words[word])
            self.topic_words.append(tmp)

        return self.topic_words
