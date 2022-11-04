import pickle
import string
import openpyxl
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
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import matplotlib.pyplot as plt
import os


def itemCollaborativeFiltering(articles_enough_df, user_item_df):
    # random 80%-20% into training and test set
    interactions_train_df, interactions_test_df = train_test_split(user_item_df,
                                                                   stratify=user_item_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)
    interactions_test_predictions_df = interactions_test_df.copy()
    interactions_test_predictions_df.drop('Rate', inplace=True, axis=1)
    stop_words_en = set(stopwords.words('english'))
    stop_words_pt = set(stopwords.words('portuguese'))
    stop_words_sp = set(stopwords.words('spanish'))
    stop_words = stop_words_en.union(stop_words_sp, stop_words_pt)
    # splitting between training and test data
    data = []
    content = []
    articles_arr = articles_enough_df['contentId'].to_numpy()
    for index, row in articles_enough_df.iterrows():
        data.append(row['text'])
        content.append(row['contentId'])
    # object that turns text into vectors
    vectorizer = TfidfVectorizer(stop_words=stop_words,
                                 ngram_range=(1, 3),
                                 analyzer='word')
    # create doc-term matrix
    dtm = vectorizer.fit_transform(data)
    dtm = csr_matrix(dtm)

    # series containing user IDs and their average ratings
    averages_users = interactions_train_df.groupby('personId')['Rate'].mean()
    users = list(user_item_df['personId'].unique())
    items = list(articles_enough_df['contentId'].unique())


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

    return user_item_df


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
    articles_enough_df = pd.DataFrame()
    text = []
    lang = []
    content = []
    for index, row in articles_df.iterrows():
        if row['contentId'] in items_enough_rated:
            content.append(row['contentId'])
            text.append(row['text'])
            lang.append(row['lang'])
    articles_enough_df['contentId'] = content
    articles_enough_df['text'] = text
    articles_enough_df['lang'] = lang

    return articles_enough_df, users_enough_interactions, items_enough_rated


def main():
    # path statement necessary to let the project work in different environments with respect to PyCharm
    here = os.path.dirname(os.path.abspath(__file__))
    nltk.download('punkt')
    nltk.download('stopwords')
    # CSV files readings
    articles_df = pd.read_csv(os.path.join(here, '../files/shared_articles.csv'))
    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    interactions_df = pd.read_csv(os.path.join(here, '../files/users_interactions.csv'))
    articles_enough_df, users_enough_interactions, items_enough_rated = dataPreProcessing(articles_df,
                                                                                          interactions_df)
    user_item_df = userItemRating(interactions_df, users_enough_interactions,
                                  items_enough_rated)
    itemCollaborativeFiltering(articles_enough_df, user_item_df)


if __name__ == '__main__':
    main()

# I studied also here: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54


# users = np.zeros(len(users_with_enough_interactions_df))
# personId = users_with_enough_interactions_df.index[0]
# interactions = interactions_df[interactions_df['personId'] == personId]


# print(users_with_enough_interactions_df.index[0])
# print(users_with_enough_interactions_df.iloc[0])
