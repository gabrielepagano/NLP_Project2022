import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def tokenize_count(sentence, p):
    """
        Counts the number of tokens for a provided sentence or paragraph

        Args:
            sentence: a natural language text. It can be one sentence or a whole paragraph
            p: a boolean, True to consider and count stopwords

        Returns:
            len(tkn): the number of tokens in the given sentence or paragraph
    """

    tkn = nltk.word_tokenize(sentence)

    if p:
        stop_words_en = set(stopwords.words('english'))
        stop_words_pt = set(stopwords.words('portuguese'))
        stop_words_sp = set(stopwords.words('spanish'))
        stop_words = stop_words_en.union(stop_words_sp, stop_words_pt)
        tkn = [t for t in tkn if t not in stop_words]

    return len(tkn)


def tokenize_clean_text(text):
    """
        Cleans the provided text of any punctuation marks

        Args:
            text: the text to clean

        Returns:
            text: the clean text
    """

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # convert to lower case
    text = text.lower()
    return text


def tokenize_count_characters(text):
    """
        Counts the common & uncommon characters in a provided text

        Args:
            text: the text to count uncommon characters in

        Returns:
            count_common: the number of common characters in the provided text
            count_uncommon: the number of uncommon characters in the provided text
    """

    # the considered common characters
    commons = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z',
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    text_cleaned = tokenize_clean_text(text)

    count_common = 0
    count_uncommon = 0

    for elem in text_cleaned:
        if elem not in commons:
            count_uncommon += 1
        else:
            count_common += 1
    return count_common, count_uncommon


def tokenize_proportion(sentence):
    """
        Calculates proportion of stopwords and uncommon characters in a provided text

        Args:
            sentence: the text to calculate the proportion on

        Returns:
            proportion: the calculated proportion of stopwords and uncommon characters in the text
    """

    # counting the number of total tokens and stopwords
    t = tokenize_count(sentence, True)
    total_t = tokenize_count(sentence, False)

    # counting the number of total characters and uncommon characters
    common, uncommon = tokenize_count_characters(sentence)
    total_c = common + uncommon

    proportion = (t / total_t) + (uncommon / total_c)

    return proportion


def tokenize_count_pronouns(sentence):
    """
        Counts the number of personal pronounce in a given text

        Args:
            sentence: the text to count personal pronouns in

        Returns:
            c: the amount of personal pronounce in the provided text
    """

    # list of personal pronouns
    pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']

    c = 0
    for t in word_tokenize(sentence):
        if t.lower() in pronouns:
            c += 1

    return c
