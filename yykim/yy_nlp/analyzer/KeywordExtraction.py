class TfIdf():
    def __init__(self, docs):
        self.docs = docs
    
    def _make_token_list(self):
        total_tokens = []
        for doc in self.docs:
            total_tokens.extend(doc)
        total_unique_tokens = list(set(total_tokens))    
        total_unique_tokens = dict(zip(total_unique_tokens, [i for i in range(len(total_unique_tokens))]))
        return total_unique_tokens
  
    def _cal_tf(self, target_doc):
        tf = {}
        for idx, token in enumerate(self._make_token_list()):
            tf.setdefault(token, target_doc.count(token)/len(target_doc))
        return tf

    def _cal_idf(self):
        idf = np.zeros((len(self._make_token_list())))
        for doc in self.docs:
            idf += np.array([0 if tf==0 else 1 for tf in self._cal_tf(doc).values()])
        return dict(zip(self._make_token_list().keys(), -np.log(idf/len(self.docs))))
    
    def cal_tfidf(self, target_doc):
        return np.array(list(self._cal_tf(target_doc).values()))*np.array(list(self._cal_idf().values()))