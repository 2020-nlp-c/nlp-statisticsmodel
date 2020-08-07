# Document Classifier
class NaiveBayesClassfier():
    def __init__(self):
        pass

    # 사전확률 , 우도 학습
    def train(self, docs, labels, k=0.5, model= "nbc.model"):
        # label 별 인덱스 지정
        import numpy as np
        label_ls = np.unique(labels)
        label_dic = {k:i for i, k in enumerate(label_ls)}
        label_count =np.zeros(len(label_ls))

        tokenized_docs = [d.split() for d in docs]
        # label 별 빈도 계산
        nbc_dic = {} #노멀과 스팸 둘다 한번에 구성하기 위해서 튜플로 구성
        
        # 라플라스 스무딩을 위한 k
        k=0.5
        for i ,doc in enumerate(tokenized_docs) :
            for w in doc :
                #dic에 있는지 유무에 따라 새로 만들거나 카운팅 해주거나
                if w in nbc_dic.keys():
                    nbc_dic[w][label_dic[labels[i]]] = nbc_dic[w][label_dic[labels[i]]] +1
                    
                else :
                    nbc_dic[w] = np.zeros(len(label_ls) *3)
                    nbc_dic[w][label_dic[labels[i]]] = 1
                label_count[label_dic[labels[i]]] += 1
            
            # 확률계산
            for w in nbc_dic.keys() :
                # 라벨별로 빈도계산을 위한 루프 
                
                # 딕셔너리에서 확률 값을 어떻게 해야 효율적으로 관리할까 
                for label in label_dic.keys() :
                    nbc_dic[w][label_dic[label] +len(label_ls) *1] = (k + nbc_dic[w][label_dic[label]]) /( 2*k + label_count[label_dic[label]])
                    nbc_dic[w][label_dic[label] +len(label_ls) *2] = np.log((k + nbc_dic[w][label_dic[label]]) /( 2*k + label_count[label_dic[label]]))
            self.nbc_dic = nbc_dic

            import pickle 
            with open(model, 'wb') as f:
                pickle.dump(self.nbc_dic, f)

        print('test')


        # label 별 확률, 로그확룰

        print("test")
        pass
    
    #  새로운 문서 분류
    #  여기서 docs는 트레인 도튜먼트랑 다른 (새로운) 테스트 도큐먼트이다.
    def predict(self , docs):
        pass

    # 정확도 측정
    def score(self):
        pass


if __name__ =="__main__":
    
    import os
    import pandas as pd

    # 경로 설정
    os.chdir("/Users/sy/nlp-statisticsmodel/sy/nlp_package/analyzer/")
    # 예제 데이터 설정
    df_train = pd.read_csv("train.csv")
    # 테스트(예측) 데이터 설정
    df_test = pd.read_csv("test.csv")
    
    # 머신러닝에서 익숙한 변수로 설정
    X_train, Y_train = df_train['mail'].tolist(), df_train['label'].tolist()
    X_test, Y_test = df_test['mail'].tolist(), df_test['label'].tolist()

    nbc = NaiveBayesClassfier()
    nbc.train(X_train, Y_train)






