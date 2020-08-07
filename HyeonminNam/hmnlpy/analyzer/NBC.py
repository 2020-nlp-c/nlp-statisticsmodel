import pandas as pd
import os
import numpy as np
import pickle

class NaiveBayesClassifier() :
    def __init__(self):
        pass
    
    def _tokenize(self, docs):
        return [d.split() for d in docs]
    # 사전확률, 우도학습
    def train(self, docs, labels, k=0.5, model='nbc.model'):
        #label별 인덱스 지정
        label_2i = {k:i for i, k in enumerate(np.unique(labels))}
        self.i_2label = {i:k for k,i  in label_2i.items()}
        tokenized_docs = self._tokenize(docs)
        label_prob = np.zeros(len(label_2i))


        #label별 빈도 계산
        nbc_dic = {}
        for i, doc in enumerate(tokenized_docs):
            for w in doc:
                if w in nbc_dic.keys():
                     nbc_dic[w]['count'][label_2i[labels[i]]] += 1
                else:
                    nbc_dic[w] = {'count' : np.zeros(len(label_2i)), 'prob' : np.zeros(len(label_2i)), 'log_prob' : np.zeros(len(label_2i))}
                    nbc_dic[w]['count'][label_2i[labels[i]]] = 1
                label_prob[label_2i[labels[i]]] += 1

        #label 확률, 로그확률
        for w in nbc_dic.keys():
            for label in label_2i.keys():
                nbc_dic[w]['prob'][label_2i[label]] = (k +nbc_dic[w]['count'][label_2i[label]]) / (2*k + label_prob[label_2i[label]])
                nbc_dic[w]['log_prob'][label_2i[label]] = np.log((k +nbc_dic[w]['count'][label_2i[label]]) / (2*k + label_prob[label_2i[label]]))
        print('test')

        self.nbc_dic = nbc_dic
        self.label_2i = label_2i
        self.label_prob = np.log(label_prob/label_prob.sum())

        with open(model, 'wb') as f:
            tmp = {'label_2i' : self.label_2i,
            'label_prob' : label_prob,
            'nbc_dic' : self.nbc_dic,
            'i_2label': self.i_2label}
            pickle.dump(tmp, f)
        

    # 새로운 문서 분류
    def predict(self, docs, model = None):
        if model:
            with open(model, 'rb') as f:
                tmp = pickle.load(f)
                self.nbc_dic = tmp['nbc_dic']
                self.label_prob = tmp['label_prob']
                self.label_2i = tmp['label_2i']
                self.i_2label = tmp['i_2label']

        tokenized_docs = self._tokenize(docs)
        results = []

        for d in tokenized_docs:
            prob_for_label = np.zeros(len(self.label_2i))
            for w in d:
                for label, i in self.label_2i.items():
                    prob_for_label[i] += self.nbc_dic[w]['log_prob'][i]

            tmp = np.exp(prob_for_label + self.label_prob)
            prob = tmp / tmp.sum()

            results.append(self.i_2label[prob.argsort()[::-1][0]])

        return results


    def score(self, docs, labels, model = None):
        predictions = self.predict(docs, model)
        return np.mean(np.array(predictions) == np.array(labels))

# if __name__ == '__main__' :
#     os.chdir(r'D:\git\local')
#     df_train = pd.read_csv('train.csv')
#     df_test = pd.read_csv('test.csv')
    
#     X_train, Y_train = df_train['mail'].tolist(), df_train['label'].tolist()
#     X_test, Y_test = df_test['mail'].tolist(), df_test['label'].tolist()
#     nbc = NaiveBayesClassifier()
#     nbc.train(X_train, Y_train)
#     nbc.predict(X_test)
#     nbc.score(X_test, Y_test, model = 'nbc.model')