class Summerizer:    # 클래스
    def __init__(self, doc):
        self.doc = doc
 
    def summerize(self, n):
        print('{0}개 문장으로 요약합니다.'.format(n))