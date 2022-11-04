import nltk
from nltk.corpus import stopwords

def tokenize_count(sentence, p):

    tkn = nltk.word_tokenize(sentence)

    if p:
        stop_words = set(stopwords.words('english'))
        # add some data specific useless words
        stop_words.add('cnn')
        stop_words.add('cnnpolitics')
        stop_words.add('us')
        tkn = [t for t in tkn if not t in stop_words]
    
    return len(tkn)

def tokenize_proportion(sentence):
    c = tokenize_count(sentence, True)
    return c / tokenize_count(sentence, False)