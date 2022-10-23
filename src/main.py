import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

#path statement necessary to let the project work in different environments with respect to PyCharm
here = os.path.dirname(os.path.abspath(__file__))

#CSV files readings
articles_df = pd.read_csv(os.path.join(here, '../files/shared_articles.csv'))
interactions_df = pd.read_csv(os.path.join(here, '../files/users_interactions.csv'))

#we only consider articles that have not been deleted
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']

#DataFrame containing IDs of users that rated at least three different items
user_interactions = interactions_df[['personId', 'contentId']].drop_duplicates().groupby(['personId'])['contentId'].count()
user_interactions_df = pd.DataFrame({'personId':user_interactions.index, 'n_interactions':user_interactions.values})
enough_user_interactions_df = user_interactions_df[user_interactions_df['n_interactions'] >= 3]
users_enough_interactions = enough_user_interactions_df['personId'].to_numpy()


#DataFrame containing IDs of items that have been rated at least by two different users
items_rated = interactions_df[['personId', 'contentId']].drop_duplicates().groupby(['contentId'])['personId'].count()
items_rated_df = pd.DataFrame({'contentId':items_rated.index, 'n_ratings':items_rated.values})
enough_items_rated_df = items_rated_df[items_rated_df['n_ratings'] >= 2]
items_enough_rated = enough_items_rated_df['contentId'].to_numpy()


#users = np.zeros(len(users_with_enough_interactions_df))
#personId = users_with_enough_interactions_df.index[0]
#interactions = interactions_df[interactions_df['personId'] == personId]




#print(users_with_enough_interactions_df.index[0])
#print(users_with_enough_interactions_df.iloc[0])

