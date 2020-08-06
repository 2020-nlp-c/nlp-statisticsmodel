class Tokenizer :
    def tokenizer(document):
    #pip install nltk ## 오류문구 해주기
        
        import nltk
        nltk.download('punkt')
        from nltk.tokenize import word_tokenize
        # word_tokenize() : 마침표와 구두점(온점(.), 컴마(,), 물음표(?), 세미콜론(;), 느낌표(!) 등과 같은 기호)으로 구분하여 토큰화

        text = document
        word_tokens = word_tokenize(text)
        return word_tokens

 