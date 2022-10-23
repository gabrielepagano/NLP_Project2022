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

articles_df = pd.read_csv('../files/shared_articles.csv')
interactions_df = pd.read_csv('../files/users_interactions.csv')

users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 3].reset_index()[['personId']]
print('# users with at least 3 interactions: %d' % len(users_with_enough_interactions_df))

items_ratings_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('contentId').size()
print('# items: %d' % len(items_ratings_count_df))
items_enough_rated_df = items_ratings_count_df[items_ratings_count_df >= 2].reset_index()[['contentId']]
print('# items rated at least 2 times: %d' % len(items_enough_rated_df))

users = np.zeros(len(users_with_enough_interactions_df))
