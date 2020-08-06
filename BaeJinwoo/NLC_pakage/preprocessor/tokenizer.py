class Tokenizer :

    def eng_tokenizer(self, document):
    #pip install nltk ## 오류문구 해주기
        
        print("nltk 라이브러리를 사용하기 때문에 없을 경우 다운 받을 수 있습니다. 기다려주세요.") ##있어도 다운받아짐. 고치기

        import nltk
        nltk.download('punkt')
        from nltk.tokenize import word_tokenize
        # word_tokenize() : 마침표와 구두점(온점(.), 컴마(,), 물음표(?), 세미콜론(;), 느낌표(!) 등과 같은 기호)으로 구분하여 토큰화

        text = document
        word_tokens = word_tokenize(text)
        return word_tokens


    ## 직접 konlpy를 가져다 쓰면되지만, 전부 외울 수 없기 때문에 편의성을 위해 개인 라이브러리에 저장하는 용도

    # def kor_tokenizer(self, document):

    #     print("konlypy 라이브러리를 사용하기 때문에 없을 경우 다운 받을 수 있습니다. 기다려주세요.") ##있어도 다운받아짐. 고치기

    #     import konlpy
    #     from konlpy.tag import Komoran
    #     komoran= Komoran()
    #     kor_text = document
    #     komoran_tokens = komoran.morphs(kor_text)
    #     return komoran_tokens

    ###불용어(Stopword) 처리

    ###N-gram