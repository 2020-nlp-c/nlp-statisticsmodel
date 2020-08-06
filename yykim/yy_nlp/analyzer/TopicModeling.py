class LatentSemanticAnalysis:
    def __init__(self, docs, package = 'manual'):
        self.docs = docs
        self.total_unique_token = []
        self.package = package
        self.U = np.zeros((1,1))
        self.s = np.zeros((1,1))
        self.VT = np.zeros((1,1))
        
    def _make_matrix(self):
        total_token = (" ".join(self.docs)).split(" ")
        self.total_unique_token = list(np.unique(total_token))
        tokenized_doc_ls = [doc.split(" ") for doc in doc_ls]
        
        token_count_in_doc = []
        for doc in tokenized_doc_ls:
            tmp = []
            for token in self.total_unique_token:
                tmp.append(doc.count(token))
            token_count_in_doc.append(tmp)        
        return np.array(token_count_in_doc)

    def _get_svm(self): 
        A = self._make_matrix()
        svd = TruncatedSVD(n_components= A.shape[0], n_iter=10)
        self.U = svd.fit_transform(A)
        self.s = svd.explained_variance_ratio_
        self.VT = svd.components_

    def predict(self, n_topics, n_words): #행렬A에서 n_topics개의 토픽 추출, 토픽당 n_words개의 단어를 추출
        if self.package == 'manual': 
            self._make_matrix()
            self._get_svm()
            for topic in range(n_topics) :
                sort = np.argsort(self.VT[topic,:])[::-1]
                top = sort[0:n_words]

                words_group = []
                for i in top :
                    words_group.append((self.total_unique_token[i], '%.3f' %self.VT.T[:,topic][i]))
                print("Topic{}: {}".format(topic+1, words_group))
        elif self.package == 'plain_sklearn':
            def my_tokenizer(text):
                return [w for w in text.split() if len(w) > 1]

            lsa_pipeline = Pipeline([('vect', CountVectorizer(tokenizer = my_tokenizer)),
                                    ('tfidf', TfidfTransformer(smooth_idf=True)),
                                    ('lsa', TruncatedSVD(n_components=n_topics, algorithm='randomized', n_iter=100)), ])
            lsa_pipeline.fit(self.docs)
            lsa = lsa_pipeline.named_steps['lsa']
            count_vect = lsa_pipeline.named_steps['vect']
            vocab = count_vect.get_feature_names() 
            for idx, topic in enumerate(lsa.components_):
                print("Topic %d:" % (idx), [(count_vect.get_feature_names()[i], topic[i].round(5)) for i in topic.argsort()[:-n_words - 1:-1]])

    #유사도 시각화
    def cosine_similarity(self, x, y):
        return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

    def sim_btw_words(self):
        self._get_svm()
        similarity = np.zeros((self.VT.shape[1],self.VT.shape[1]))
        for i, word in enumerate(self.VT.T):
            for j, another_word in enumerate(self.VT.T):
                similarity[i][j] = self.cosine_similarity(word, another_word)
        sns.heatmap(similarity,linewidth=.1, cmap='Oranges',xticklabels=["word{}".format(i+1) for i in range(similarity.shape[0])], yticklabels=["word{}".format(i+1) for i in range(similarity.shape[0])])
        
    def sim_btw_docs(self):
        self._get_svm()
        similarity = np.zeros((self.U.shape[0],self.U.shape[0]))
        for i, doc in enumerate(self.U):
            for j, another_doc in enumerate(self.U):
                similarity[i][j] = cosine_similarity(doc, another_doc)
        sns.heatmap(similarity,linewidth=.1, cmap='Blues',xticklabels=["doc{}".format(i+1) for i in range(similarity.shape[0])], yticklabels=["doc{}".format(i+1) for i in range(similarity.shape[0])])

    def sim_btw_words_docs(self):
        self._get_svm()
        similarity = np.zeros((self.U.shape[0],self.VT.shape[1]))
        for i, word in enumerate(self.U):
            for j, another_word in enumerate(self.VT.T):
                similarity[i][j] = cosine_similarity(word, another_word)
        sns.heatmap(similarity,linewidth=.1, cmap='Reds',xticklabels=["doc{}".format(i+1) for i in range(similarity.shape[1])], yticklabels=["word{}".format(i+1) for i in range(similarity.shape[0])])