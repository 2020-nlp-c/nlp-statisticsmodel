from konlpy.tag import Kkma
import pandas as pd

class luhnSummarize():
    def __init__(self, document, min=0.001, max=0.5):
        self.doc = document
        self.min = min
        self.max = max
        self.kkma_tokens = []
        self.word = []
        self.doc_split = []
        self.importance = []
        self.txt = []

    # NNG만 고려한 Kkma 토큰화
    def get_kkma_token(self):    
        kkma= Kkma()
        self.kkmaTag = kkma.pos(self.doc)

        for i in self.kkmaTag:
            if i[1] == 'NNG':
                self.kkma_tokens.append(i[0])

    # 문서내 단어 빈도
    def get_freq(self):
        keys = list(set(self.kkma_tokens))
        zero = [0 for i in range(len(keys))]
        self.freq = dict(zip(keys, zero))

        for i in keys:
            self.freq[i] = self.kkma_tokens.count(i)

    # 중요단어 결정
    def get_word(self):
        word_count = len(self.kkma_tokens)

        for i in self.freq:
            if self.freq[i] / word_count > self.min and self.freq[i] / word_count < self.max:
                self.word.append(i)

    # 문장단위로 나누기
    def split_doc(self):
        doc_split_ = self.doc.split('.')
        doc_split_.pop()

        for i in doc_split_:
            tmp = i.split('\n')
            for j in tmp:
                if j != '' and j != ' ':
                    self.doc_split.append(j)

    # 문장 중요도 계산
    def doc_importance(self):

        for i in self.doc_split:
            sig_word = []
            tokens = Kkma().morphs(i)

            for j in tokens:
                if j in self.word:
                    sig_word.append(tokens.index(j))

            length = max(sig_word) - min(sig_word) + 1
            self.importance.append(len(sig_word)*len(sig_word) / length)

    # 문장 중요도 순위별 출력
    def summarize(self):
        self.get_kkma_token()
        self.get_freq()
        self.get_word()
        self.split_doc()
        self.doc_importance()

        importance_cpy = self.importance.copy()
        importance_cpy.sort(reverse=True)

        for i in importance_cpy:
            self.txt.append(self.doc_split[self.importance.index(i)])

        self.df = pd.DataFrame(self.txt, columns=['문장 중요도 순서'], index = [i + 1 for i in range(len(self.txt))])
        print(self.df)