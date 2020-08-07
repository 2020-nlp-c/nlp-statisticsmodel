import numpy as np
from sklearn.decomposition import randomized_svd
from sklearn.feature_extraction.text import CountVectorizer
class LSA():
    def __init__(self) :     # 클래스는 뼈대를 먼저 잡아놓고 한다. def, pass로 잡고시작
        pass

    def _make_tdm(self) :
        from sklearn.feature_extraction.text import CountVectorizer
        
        def tokenize(x):
            return x.split()

        cv = CountVectorizer(tokenizer = tokenize)
        self.DTM  = cv.fit_transform(self.docs).toarray() # 카운터라이저에서 학습한걸 array한걸 TDM에 넣는다 
        self.feature_names = cv.get_feature_names()
        self.word2id = cv.vocabulary_
        # TDM.toarray() # watch는 내가 보고싶은거 넣는거임 

    def _truncatedSVD(self) : 
        from sklearn.decomposition import randomized_svd
        self.U, s, self.VT = randomized_svd(self.DTM,               # U는 문서, VT는 단어행렬이다~
                                n_components = self.k,
                                n_iter = 10)

    def print_topics(self) : 
        for topic in self.VT :
            print([self.feature_names[i] for i in topic.argsort()[::-1][:self.n_words]])
            # topic.argsort()[::-1][:self.n_words]

    def get_word_vec(self, keyword) :
        v = self.VT.T[self.word2id[keyword]]
        print("단어 {} : {} ".format(keyword, v))
        return v

    def get_doc_vec(self, idx_doc) : 
        v = self.U[idx_doc]
        print("문서 {} : {} ".format(idx_doc, v))
        return v

    def calc_similarity(self, x, y) :
        import numpy as np
            # x와 y, 두 벡터의 코사인 유사도를 계산하는 함수
        nominator = np.dot(x, y) # 분자
        denominator = np.linalg.norm(x)*np.linalg.norm(y) # 분모
        print("유사도 : {}".format(nominator / denominator))
        return nominator / denominator
    
    def train(self, docs, k, n_words = 3) : # 실행 순서가 있으니까 만들어줌
        self.docs = docs
        self.k = k
        self.n_words = n_words
        
        self._make_tdm()
        self._truncatedSVD()
        self.print_topics()        # tdm은 카운터벡터라이저 -> sklearn
        
    def search(self, keyword, n_docs = 5) :
        wv = self.get_word_vec(keyword)    
        print([self.docs[i] for i in np.dot(wv, self.U.T).argsort()[::-1][:n_docs]])


if __name__ == "__main__" :  # 해당 문장쓰면 여기부터 돌아감 -> 코드 확인시 많이씀 
    doc_ls = ['바나나 사과 포도 포도 짜장면',
         '사과 포도',
         '포도 바나나',
         '짜장면 짬뽕 탕수육',
         '볶음밥 탕수육',
         '짜장면 짬뽕',
         '라면 스시',
         '스시 짜장면',
         '가츠동 스시 소바',
         '된장찌개 김치찌개 김치',
         '김치 된장 짜장면',
         '비빔밥 김치'
         ]
    lsa = LSA()
    lsa.train(doc_ls, k = 4, n_words = 3)

    lsa.get_doc_vec(1)
    lsa.get_word_vec("라면")
    lsa.calc_similarity(lsa.get_doc_vec(1), lsa.get_word_vec("라면"))
    lsa.calc_similarity(lsa.get_word_vec("포도"), lsa.get_word_vec("사과"))
    lsa.search('비빔밥')