
class NaiveBayesClassifier:
    def __init__(self, docs, labels, words, k = 1, package= 'manual'):
    # docs는 토큰화한 문서들을 리스트화 한 이중 리스트
        self.docs = docs
        self.labels = labels
        self.words = words
        self.k = k #laplace smoothing을 위한 상수 k
        self.package = package 
        
        self.total_token = []
        self.unique_token = np.array((1,1))
        self.category_name=[]
        self.classified_token=[]
        self.prior_p=[]
        self.probability_dict=[]


    def _cal_prior(self):
        #토큰 개수 세기 
        for doc in self.docs:
            for token in doc:
                self.total_token.append(token)
        self.unique_token = np.unique(self.total_token)        
        #카테고리별 확률 구하기
        self.category_name = np.unique(labels)

        count_labels=[]
        for name in self.category_name:
            tmp = []
            for idx, doc in enumerate(self.docs):
                if self.labels[idx] == name:
                    tmp.extend(doc)
            self.classified_token.append(tmp)
            count_labels.append(len(tmp))
        self.prior_p = list(np.array(count_labels)/len(self.total_token))
        return dict(zip(list(self.category_name),self.prior_p))

    def _cal_posterior(self):
        self._cal_prior()
       #각각의 카테고리에서 토큰이 나온 횟수세기        
        ls_n_classified_token=[]            
        for classfiedtoken in self.classified_token:
            n_classified_token=[]
            for token in self.unique_token:
                tmp = classfiedtoken.count(token)
                n_classified_token.append(tmp)
            ls_n_classified_token.append(n_classified_token)

        for i, v in enumerate(ls_n_classified_token):
            for idx, w in enumerate(v):
                ls_n_classified_token[i][idx] = np.log((w+1*self.k)/(sum(v)+2*self.k))
            tmp = dict(zip(list(self.unique_token),ls_n_classified_token[i]))
            self.probability_dict.append(tmp)
        return self.probability_dict
        
    def _use_handmadeone(self):
        self._cal_prior()
        self._cal_posterior()
        p_list= []
        for idx, p in enumerate(self.prior_p): 
            tmp = np.log(p)
            for word in self.words:
                tmp += self.probability_dict[idx][word]
            tmp = np.exp(tmp)
            p_list.append(tmp) 

        return self.category_name[p_list.index(max(p_list))]

    def _use_plain_sklearn(self):
        text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultinomialNB()), ])
        og_docs = [" ".join(i) for i in self.docs]
        self.docs = og_docs
        text_clf = text_clf.fit(self.docs, self.labels)

        self.words = [" ".join(self.words)]
        return text_clf.predict(self.words)

    def _use_gs_sklearn(self, parameters_dict):
        text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultinomialNB()), ])
        og_docs = [" ".join(i) for i in self.docs]
        self.docs = og_docs
        text_clf = text_clf.fit(self.docs, self.labels)
        
        # parameters_dict = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False)}
        gs_clf = GridSearchCV(text_clf, parameters_dict, n_jobs=-1, verbose=2)
        gs_clf = gs_clf.fit(self.docs, self.labels)

        self.words = [" ".join(self.words)]
        return gs_clf.best_estimator_.get_params(), gs_clf.predict(self.words)

    def classify(self):
        if self.package == 'manual':
            self._cal_prior()
            self._cal_posterior()
            return self._use_handmadeone()

        elif self.package == 'sklearn':
            return self._use_plain_sklearn()

        elif self.package == 'sklearn_gs':
            from ast import literal_eval
            parameters_dict = input("조정할 파라미터 딕셔너리를 입력해주세요: ")
            parameters_dict = literal_eval(parameters_dict) 
            return self._use_gs_sklearn(parameters_dict)

        else:
            print("지원하지 않는 방식입니다.")