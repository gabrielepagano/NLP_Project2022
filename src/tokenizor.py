import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def tokenize_count(sentence, p):
    tkn = nltk.word_tokenize(sentence)

    if p:
        stop_words_en = set(stopwords.words('english'))
        stop_words_pt = set(stopwords.words('portuguese'))
        stop_words_sp = set(stopwords.words('spanish'))
        stop_words = stop_words_en.union(stop_words_sp, stop_words_pt)
        tkn = [t for t in tkn if not t in stop_words]

    return len(tkn)

def tokenize_clean_text(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # convert to lower case
    text = text.lower()
    return text

def tokenize_countUncommonCharacters(text):
    commons = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    text_cleaned = tokenize_clean_text(text)
    count_uncommons = 0
    for elem in text_cleaned:
        if elem not in commons:
            count_uncommons += 1
    return count_uncommons



def tokenize_proportion(sentence):
    c = tokenize_count(sentence, True)
    return c / tokenize_count(sentence, False)


def tokenize_count_pronouns(sentence):
    pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']
    c = 0
    for t in word_tokenize(sentence):
        if t.lower() in pronouns:
            c += 1
    return c
