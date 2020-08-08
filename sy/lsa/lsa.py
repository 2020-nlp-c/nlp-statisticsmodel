

class LSA():
    def __init___(self) :
        pass
    
    def _make_tdm(self) :
        from sklearn.feature_extraction.text import CountVectorizer
        
        def tokenize(x) :
            return x.split()

        cv = CountVectorizer(tokenizer = tokenize)
        self.DTM = cv.fit_transform(self.docs).toarray()
        self.feature_names = cv.get_feature_names()
        self.word2id =cv.vocabulary


    def _truncatedSVD(self) :
        from sklearn.decomposition import randomized_svd
        self.U, s, self.VT = randomized_svd(self.DTM ,
                                  n_components = self.k ,
                                  n_iter=10 )
        
    def print_topics(self) :
        for topic in self.VT:
            print([self.feature_names[i] for i in topic.argsort()[::-1][:self.n_words]])
            

        pass

    def get_wrod_vec(self, keyword) :
        v= self.VT.T[self.word2id[keyword]]
        print("문서 {} : {}".format(idx_doc, self.U[idx_doc]))
        return v

    def get_doc_vec(self , idx_doc) :
        v = self.U[idx_doc]
        print("문서 {} : {}".format(idx_doc, self.U[idx_doc]))
        return v
        

    def calc_similarity(self) :
        import numpy as np
            # x와 y, 두 벡터의 코사인 유사도를 계산하는 함수
        nominator = np.dot(x, y)    # 분자
        denominator = np.linalg.norm(x)*np.linalg.norm(y)  # 분모
        return nominator/denominator
        

    def train(self, docs, k, n_words =3 ) :
        self.docs = docs
        self.k = k
        self.n_words = n_words

        self._make_tdm()
        self._truncatedSVD()
        self.print_topics()
        

if __name__ == "__main__" :
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
    
    lsa =LSA()
    lsa.train(doc_ls, k=4, n_words=3)

    lsa.get_doc_vec(1)
    lsa.get_wrod_vec("라면")
    lsa.calc_similarity(lsa.get_doc_vec(1), lsa.get_wrod_vec("라면") )


