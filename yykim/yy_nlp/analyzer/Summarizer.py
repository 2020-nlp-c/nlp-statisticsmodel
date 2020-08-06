
class TextRank:
    def __init__(self, doc, max_iter=10, keysentence_number=1, DampingFactor=0.85, threshold=0.001 ): #s는 문서내 각각 토큰화된 문장을 리스트에 넣은 것
        self.max_iter = max_iter
        self.keysentence_number = keysentence_number
        self.link_graph = np.zeros((1,1))
        self.edge_graph = np.zeros((1,1))
        self.scores = np.ones((1,1))
        self.edge_count = []
        self.doc = doc
        self.DampingFactor = DampingFactor
        self.similarity ={}
        self.graph = np.zeros((1,1))
        self.threshold = threshold


    def _count_similarity(self):
        for i, v in enumerate(self.doc):
            for j in range(len(self.doc)):
                union = set(v).union(set(self.doc[j]))
                intersection = set(v).intersection(set(self.doc[j]))
                self.similarity.setdefault((i,j),len(intersection)/len(union))
        return self.similarity                
    
    def _make_graph(self):
        self.graph = np.zeros((len(self.doc),len(self.doc)))
        for i in self.similarity:
            self.graph[i] = self.similarity[i]
        for i in range(len(self.graph)):
            self.graph[(i,i)] = 0
        return self.graph

    def _make_edge_weight(self):
        self.edge_weight = np.zeros((len(self.doc),len(self.doc)))
        for i, v in enumerate(self.graph):
            self.edge_weight[i] = self.graph[i]/np.sum(v)
        return self.edge_weight

    def _make_scores(self):
        self.scores = np.zeros((len(self.doc),))
        for i,v in enumerate(self.graph):
            for j, w in enumerate(self.graph):
                self.scores[i] += w[i]
        self.scores = (1-self.DampingFactor) + self.DampingFactor*self.scores
        return self.scores
    
    def _learning(self):
        self.scores = self._make_scores()
        for iter in range(self.max_iter):
            before_scores = self.scores.copy()

            for i, v in enumerate(self.graph):
                for j,w in enumerate(self.graph):
                    self.graph[(i,j)] = self.edge_weight[(i,j)]*self.scores[i]
            
            self.scores = self._make_scores()

            if np.sum(np.square(self.scores-before_scores)) < self.threshold :
                break
        
    def text_rank(self): #스코어순서대로 정렬 및 해당 단어 보여주기
        self._count_similarity()
        self._make_edge_weight()
        self._make_scores()
        self._learning()

        sort = np.argsort(self.scores)[::-1]
        top = sort[0:self.keysentence_number]

        keysentence_ls = []
        for i in top :
            keysentence_ls.append(self.doc[i])
        return keysentence_ls


class LuhnSummarize:
    def __init__(self, text, doc, keysentence_number=1, min=0.01, max=0.2):
    # text는 문서 전체, doc는 문서 내 문장을 각각 리스트화 한 이중 리스트
        self.keysentence_number = keysentence_number
        self.doc = doc
        self.min = min
        self.max = max
        self.sentence_score = []

    def _make_score(self):
        kkma= Kkma()

        #전체 문서 토크나이징
        total_token = kkma.morphs(text)
        unique_token = np.unique(total_token)
        
        #{유니크 토큰: 토큰 등장 횟수}로 딕셔너리 정리        
        n_unique_token=[]
        for token in unique_token:
            tmp = total_token.count(token)
            n_unique_token.append(tmp)
        dict_unique_token = dict(zip(list(unique_token),n_unique_token))

        #문장별로 토크나이즈
        tokenized_sens = []
        for sens in doc:
            tokenized_sen = kkma.morphs(sens)
            tokenized_sens.append(tokenized_sen)

        # 각 문장의 토큰들을 해당 토큰의 빈도수/전체토큰수(score)로 치환
        words_score = tokenized_sens.copy()

        for sen_idx, sen in enumerate(tokenized_sens):
            for morph_idx,morph in enumerate(sen):
                words_score[sen_idx][morph_idx] = dict_unique_token[morph]/len(total_token)

        # 중요 단어로 판단된 단어들의 문장 내 인덱스 확인
        keywords_idx=[]
        for sen_idx, sen in enumerate(words_score):
            keywords_idx_tmp = []
            for morph_idx,morph in enumerate(sen):
                if self.min < morph < self.max:
                    keywords_idx_tmp.append(morph_idx)
            keywords_idx.append(keywords_idx_tmp)

        for sens in keywords_idx:
            try:
                self.sentence_score.append(np.square(sens[-1]-sens[0]+1)/len(sens))
            except:
                self.sentence_score.append(0) #문장 내에 중요 단어가 1개 이하인 경우
        return self.sentence_score                       
    
    def luhn_summarize(self):
        self._make_score()
        sort = np.argsort(self.sentence_score)[::-1]
        top = sort[0:self.keysentence_number]

        keysentence_ls = []
        for i in top :s
            keysentence_ls.append(self.doc[i])
        return keysentence_ls