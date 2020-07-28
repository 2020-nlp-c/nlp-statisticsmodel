import numpy as np
import copy

class textSummarize():
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
    def textRank_result(self):
        self.jaccard_arr()
        self.first_score()
        self.get_edge()

        for i in range(self.max_iter):
            self.get_arr()
            self.get_score()
            print('arr')
            print(self.arr)
            print('score')
            print(self.score)
            if min(self.diff) < self.threshold:
                break

        print()
        print(self.s_list[self.score.index(max(self.score))])