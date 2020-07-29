class TfIdf:
    def __init__(self, a, docs): #a는 문장1개, docs는 bunch(리스트화 한거)
        self.a = a
        self.docs = docs
        self.total = []
        self.a_count = []
        self.a_token = []
        self.idf = 1
    
    def total_token(self):
        for doc in self.docs:
            doc_token = doc.split(" ")
            self.total.extend(doc_token)
        #전체 토큰의 개수 세기
        total_unique = np.unique(self.total)        
        return total_unique
    
    def count_token(self):
        #토크나이징
        self.a_token = self.a.split(" ")

        #해당 문서에 토큰 개수 세기 
        self.a_count = []
        for token in self.total:
            self.a_count.append(self.a_token.count(token))
        self.a_count = np.array([self.a_count])

        return self.a_count

    def get_tf(self):
        return self.a_count/len(self.a_token)

    def get_idf(self):
        tmp=[]
        for doc in self.docs: #문서 번치에서 문서 하나씩 빼냄
            doc_token = doc.split(" ") #문서 토크나이징
            doc_count = []  
            for token in self.total:
                doc_count.append(doc_token.count(token))           
            tmp.append(np.array([0 if i == 0 else 1 for i in doc_count]))

        nt = np.zeros((len(self.total),))
        for i in tmp:
            nt += i
        self.idf = np.log(len(self.docs)/nt)

        return self.idf

    def get_tf_idf(self):
        return self.a_count*self.idf