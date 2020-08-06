import pandas as pd
import numpy as np
import copy
from nltk.tokenize import sent_tokenize

# 입력값(텍스트, 윈도우 사이즈)
class TextSummarize:
    
    def __init__(self, text):
        self.text = text
        self.sent_lst = sent_tokenize(text)
        self.token_lst = [set(x) for x in self.sent_lst]
        self.score_lst = []

    def edge(self):
        # 문장 유사도 계산
        sim_mat = np.zeros((len(self.token_lst), len(self.token_lst)))
        for idx_1, tokens in enumerate(self.token_lst):
            for idx_2, other in enumerate(self.token_lst):
                if idx_1 != idx_2:
                    sim = len(tokens.intersection(other))/len(tokens.union(other))
                    sim_mat[idx_1, idx_2] = sim
                    
        # 엣지, 최초 스코어 계산
        edge = np.zeros((len(self.token_lst), len(self.token_lst)))
        score_lst = []
        for idx_1, tokens in enumerate(self.token_lst):
            total = np.sum(sim_mat[idx_1])
            score_lst.append(total)
            for idx_2, other in enumerate(self.token_lst):
                if idx_1 != idx_2:
                    edge[idx_1, idx_2] = sim_mat[idx_1, idx_2]/total
        self.score_lst = score_lst
        return edge, sim_mat
                    

    def score(self, threshold=0.001, damp=0.85):
        # 행렬 계산 위해서 damp 값으로 (4, 0) 행렬 만들기
        damp_np = np.array([damp for x in range(len(self.token_lst))])
        edge, sim_mat = self.edge()
        
        # threshold까지 반복해서 스코어 갱신
        while True:
            prev_score = copy.copy(self.score_lst)
            for idx_1, score in enumerate(prev_score):
                for idx_2, other in enumerate(prev_score):
                    if edge[idx_1, idx_2] != 0:
                        sim = score*edge[idx_1][idx_2]
                        sim_mat[idx_1, idx_2] = sim
            new_score = damp_np.dot(sim_mat) + (1-damp)
            self.score_lst = copy.copy(new_score)
            
            # threshold조건 도달하면 데이터프레임 리턴
            if np.sum(np.fabs(prev_score-new_score)) <= threshold:
                score_dic = {'스코어': self.score_lst}
                idx = [' '.join(self.sent_lst[x]) for x in range(len(self.score_lst))]
                df = pd.DataFrame(score_dic, index= idx)
                print(df)
                return df