from konlpy.tag import Kkma
import pandas as pd
import numpy as np
import copy

class luhnSummarize():
    def __init__(self, document, min = 0.001, max = 0.5):
        self.doc = document
        self.min = min
        self.max = max
        self.kkma_tokens = []
        self.word = []
        self.doc_split = []
        self.importance = []
        self.txt = []

    # NNG만 고려한 Kkma 토큰화
    def get_kkma_token(self):    
        kkma= Kkma()
        self.kkmaTag = kkma.pos(self.doc)

        for i in self.kkmaTag:
            if i[1] == 'NNG':
                self.kkma_tokens.append(i[0])

    # 문서내 단어 빈도
    def get_freq(self):
        keys = list(set(self.kkma_tokens))
        zero = [0 for i in range(len(keys))]
        self.freq = dict(zip(keys, zero))

        for i in keys:
            self.freq[i] = self.kkma_tokens.count(i)

    # 중요단어 결정
    def get_word(self):
        word_count = len(self.kkma_tokens)

        for i in self.freq:
            if self.freq[i] / word_count > self.min and self.freq[i] / word_count < self.max:
                self.word.append(i)

    # 문장단위로 나누기
    def split_doc(self):
        doc_split_ = self.doc.split('.')
        doc_split_.pop()

        for i in doc_split_:
            tmp = i.split('\n')
            for j in tmp:
                if j != '' and j != ' ':
                    self.doc_split.append(j)

    # 문장 중요도 계산
    def doc_importance(self):

        for i in self.doc_split:
            sig_word = []
            tokens = Kkma().morphs(i)

            for j in tokens:
                if j in self.word:
                    sig_word.append(tokens.index(j))

            length = max(sig_word) - min(sig_word) + 1
            self.importance.append(len(sig_word)*len(sig_word) / length)

    # 문장 중요도 순위별 출력
    def summarize(self):
        self.get_kkma_token()
        self.get_freq()
        self.get_word()
        self.split_doc()
        self.doc_importance()

        importance_cpy = self.importance.copy()
        importance_cpy.sort(reverse=True)

        for i in importance_cpy:
            self.txt.append(self.doc_split[self.importance.index(i)])

        return self.txt




class textRank():
    def __init__(self, sentences):
        self.s_list = sentences
        self.edge_arr = [] 
        self.score = []
        self.diff = []
        self.arr = []
        self.max_iter = 50
        self.threshold = 0.001

    # 자카드 유사도 계산하는 함수
    def jaccard(self, a, b):
        token_a = a.split()
        token_b = b.split()

        union = set(token_a).union(set(token_b))
        intersection = set(token_a).intersection(set(token_b))

        return len(intersection)/len(union)

    # 자카드 유사도 행렬
    def jaccard_arr(self):
        for i in range(len(self.s_list)):
            self.edge_arr.append([0 for i in range(len(self.s_list))])
    
        for i in range(len(self.edge_arr) - 1):
            for j in range(i + 1, len(self.edge_arr)):
                self.edge_arr[i][j], self.edge_arr[j][i] = self.jaccard(self.s_list[i], self.s_list[j]), self.jaccard(self.s_list[i], self.s_list[j])

    # 첫번째 스코어
    def first_score(self):
        for i in range(len(self.edge_arr)):
            sum = 0
            for j in range(len(self.edge_arr)):
                sum += self.edge_arr[i][j]
            self.score.append(sum)

    # 엣지 가중치 
    def get_edge(self):
        for i in range(len(self.edge_arr)):
            for j in range(len(self.edge_arr)):
                self.edge_arr[i][j] /= self.score[i]

    # 행렬 구하기
    def get_arr(self):
        self.arr = copy.deepcopy(self.edge_arr)

        for i in range(len(self.arr)):
            for j in range(len(self.arr)):
                self.arr[i][j] *= self.score[i]

    # 스코어 구하기
    def get_score(self):
        self.diff = []

        for i in range(len(self.arr)):
            sum = 0
            for j in range(len(self.arr)):
                sum += self.arr[j][i]
            self.diff.append(abs(self.score[i] - ((1 - 0.85) + 0.85 * sum)))
            self.score[i] = (1 - 0.85) + 0.85 * sum

    # 결과
    def summarize(self):
        self.jaccard_arr()
        self.first_score()
        self.get_edge()

        for i in range(self.max_iter):
            self.get_arr()
            self.get_score()
            if min(self.diff) < self.threshold:
                break

        return self.s_list[self.score.index(max(self.score))]