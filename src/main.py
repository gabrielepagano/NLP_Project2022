import string

import numpy as np
import scipy
import pandas as pd
import nltk
import math
import random
import sklearn
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import FreqDist
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import matplotlib.pyplot as plt
import os


def standardize_text(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # convert to lower case
    text = text.lower()
    return text


def get_tokens(text, stop_words):
    # get individual words
    tokens = word_tokenize(text)
    # remove stopwords
    tokens = [t for t in tokens if not t in stop_words]
    return tokens


def itemCollaborativeFiltering(CNN_articles_df, BR_articles_df):
    Labels = ['news', 'sport', 'politics', 'business', 'health']
    Labels_BR = ['colunas', 'esporte', 'poder', 'mercado', 'equilibrioesaude']
    stop_words = set(stopwords.words('english'))
    # add some data specific useless words
    stop_words.add('cnn')
    stop_words.add('cnnpolitics')
    stop_words.add('us')
    # select only the rows and columns that we need to perform text categorization
    CNN_articles_df = CNN_articles_df[['Category', 'Keywords']]
    CNN_articles_df = CNN_articles_df[CNN_articles_df['Category'].isin(Labels)]
    BR_articles_df = BR_articles_df[['category', 'text']]
    BR_articles_df = BR_articles_df[BR_articles_df['category'].isin(Labels_BR)]
    # random 80%-20% into training and test set
    CNN_train_df, CNN_test_df = train_test_split(CNN_articles_df,
                                                 test_size=0.20,
                                                 random_state=42)
    BR_train_df, BR_test_df = train_test_split(BR_articles_df,
                                               test_size=0.20,
                                               random_state=42)
    # frequency of the words in the document
    tokens = defaultdict(list)
    for index, row in CNN_articles_df.iterrows():
        label = row['Category']
        text = standardize_text(row['Keywords'])
        doc_tokens = get_tokens(text, stop_words)
        tokens[label].extend(doc_tokens)
    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))



def userItemRating(interactions_df, users_enough_interactions, items_enough_rated):
    user_item_matrix = np.zeros((len(items_enough_rated), len(users_enough_interactions)))

    # efficient version of the algorithm
    for i in interactions_df.index:
        if ((np.isin(interactions_df['personId'][i], users_enough_interactions)) and
                (np.isin(interactions_df['contentId'][i], items_enough_rated))):
            row_index = np.where(items_enough_rated == interactions_df['contentId'][i])[0][0]
            col_index = np.where(users_enough_interactions == interactions_df['personId'][i])[0][0]
            if interactions_df['eventType'][i] == "LIKE":
                user_item_matrix[row_index][col_index] = 5

            elif interactions_df['eventType'][i] == "VIEW" and user_item_matrix[row_index][col_index] == 0:
                user_item_matrix[row_index][col_index] = 1

            elif interactions_df['eventType'][i] == "COMMENT" and user_item_matrix[row_index][col_index] < 2:
                user_item_matrix[row_index][col_index] = 2

            elif ((interactions_df['eventType'][i] == "FOLLOW" or interactions_df['eventType'][i] == "BOOKMARK") and
                  user_item_matrix[row_index][col_index] < 3):

                user_item_matrix[row_index][col_index] = 3

            elif ((interactions_df['eventType'][i] == "FOLLOW" or
                   interactions_df['eventType'][i] == "BOOKMARK") and
                  user_item_matrix[row_index][col_index] == 3):

                user_item_matrix[row_index][col_index] = 4

    # creation of a dataframe containing only effective user-item interactions
    temp_index = 0
    nonzero_len = len(user_item_matrix[np.nonzero(user_item_matrix)])
    temp_array = np.zeros((nonzero_len, 3))
    for i in range(len(users_enough_interactions)):
        for j in range(len(items_enough_rated)):
            if user_item_matrix[j][i] != 0:
                temp_array[temp_index][0] = users_enough_interactions[i]
                temp_array[temp_index][1] = items_enough_rated[j]
                temp_array[temp_index][2] = user_item_matrix[j][i]
                temp_index = temp_index + 1
    user_item_df = pd.DataFrame(temp_array, columns=['personId', 'contentId', 'Rate'])

    # random 80%-20% into training and test set
    interactions_train_df, interactions_test_df = train_test_split(user_item_df,
                                                                   stratify=user_item_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)


def dataPreProcessing(articles_df, interactions_df):
    # array containing IDs of users that rated at least three different items
    user_interactions = interactions_df[['personId', 'contentId']].drop_duplicates().groupby(['personId'])[
        'contentId'].count()
    user_interactions_df = pd.DataFrame(
        {'personId': user_interactions.index, 'n_interactions': user_interactions.values})
    enough_user_interactions_df = user_interactions_df[user_interactions_df['n_interactions'] >= 3]
    users_enough_interactions = enough_user_interactions_df['personId'].to_numpy()

    # array containing IDs of items that have been rated at least by two different users
    items_rated = interactions_df[['personId', 'contentId']].drop_duplicates().groupby(['contentId'])[
        'personId'].count()
    items_rated_df = pd.DataFrame({'contentId': items_rated.index, 'n_ratings': items_rated.values})
    enough_items_rated_df = items_rated_df[items_rated_df['n_ratings'] >= 2]
    items_enough_rated = enough_items_rated_df['contentId'].to_numpy()

    return users_enough_interactions, items_enough_rated


def main():
    # path statement necessary to let the project work in different environments with respect to PyCharm
    here = os.path.dirname(os.path.abspath(__file__))
    nltk.download('punkt')
    nltk.download('stopwords')
    # CSV files readings
    articles_df = pd.read_csv(os.path.join(here, '../files/shared_articles.csv'))
    interactions_df = pd.read_csv(os.path.join(here, '../files/users_interactions.csv'))
    CNN_articles_df = pd.read_csv(os.path.join(here, '../files/CNN_articles.csv'))
    BR_articles_df = pd.read_csv(os.path.join(here, '../files/brazilian_articles.csv'))
    users_enough_interactions, items_enough_rated = dataPreProcessing(articles_df, interactions_df)
    userItemRating(interactions_df, users_enough_interactions, items_enough_rated)
    itemCollaborativeFiltering(CNN_articles_df, BR_articles_df)


if __name__ == '__main__':
    main()

# I studied also here: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54


# users = np.zeros(len(users_with_enough_interactions_df))
# personId = users_with_enough_interactions_df.index[0]
# interactions = interactions_df[interactions_df['personId'] == personId]


# print(users_with_enough_interactions_df.index[0])
# print(users_with_enough_interactions_df.iloc[0])
