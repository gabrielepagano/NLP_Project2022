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

class DataFrame:
    def __init__(this, path1, path2):
        #path statement necessary to let the project work in different environments with respect to PyCharm
        here = os.path.dirname(os.path.abspath(__file__))

        #CSV files readings
        this.articles_df = pd.read_csv(os.path.join(here, path1))
        this.interactions_df = pd.read_csv(os.path.join(here, path2))

        #array containing IDs of users that rated at least three different items
        user_interactions = this.interactions_df[['personId', 'contentId']].drop_duplicates().groupby(['personId'])['contentId'].count()
        user_interactions_df = pd.DataFrame({'personId':user_interactions.index, 'n_interactions':user_interactions.values})
        enough_user_interactions_df = user_interactions_df[user_interactions_df['n_interactions'] >= 3]
        this.users_enough_interactions = enough_user_interactions_df['personId'].to_numpy()


        #array containing IDs of items that have been rated at least by two different users
        items_rated = this.interactions_df[['personId', 'contentId']].drop_duplicates().groupby(['contentId'])['personId'].count()
        items_rated_df = pd.DataFrame({'contentId':items_rated.index, 'n_ratings':items_rated.values})
        enough_items_rated_df = items_rated_df[items_rated_df['n_ratings'] >= 2]
        this.items_enough_rated = enough_items_rated_df['contentId'].to_numpy()

        this.create_rating_matrix()
        this.improve_rating_matrix()

    def create_rating_matrix(this):
        row_index = 0
        col_index = 0
        this.user_item_matrix = np.zeros((len(this.items_enough_rated),len(this.users_enough_interactions)))

        #algorithm (inefficient) used to build the user-item matrix
        #for x in items_enough_rated:
        #    for y in users_enough_interactions:
        #        temp_rate = 0
        #        interactions = interactions_df[(interactions_df['contentId'] == x) & (interactions_df['personId'] == y)]
        #        for label,value in interactions['eventType'].items():
        #            if value == "LIKE":
        #                temp_rate = 5
        #                break
        #            elif (value == "VIEW" and temp_rate == 0):
        #                temp_rate = 1
        #            elif (value == "COMMENT" and temp_rate < 2):
        #                temp_rate = 2
        #            elif ((value == "FOLLOW" or value == "BOOKMARK") and temp_rate < 3):
        #                temp_rate = 3
        #            elif ((value == "FOLLOW" or value == "BOOKMARK") and temp_rate == 3):
        #                temp_rate = 4
        #        user_item_matrix[row_index,col_index] = temp_rate
        #    col_index = col_index + 1
        #row_index = row_index + 1

        #efficient version of the previous algorithm
        for i in this.interactions_df.index:
            if((np.isin(this.interactions_df['personId'][i], this.users_enough_interactions)) and
                    (np.isin(this.interactions_df['contentId'][i], this.items_enough_rated))):
                    row_index = np.where(this.items_enough_rated == this.interactions_df['contentId'][i])[0][0]
                    col_index = np.where(this.users_enough_interactions == this.interactions_df['personId'][i])[0][0]
                    if (this.interactions_df['eventType'][i] == "LIKE"):
                            this.user_item_matrix[row_index][col_index] = 5

                    elif (this.interactions_df['eventType'][i] == "VIEW" and this.user_item_matrix[row_index][col_index] == 0):
                            this.user_item_matrix[row_index][col_index] = 1

                    elif (this.interactions_df['eventType'][i] == "COMMENT" and this.user_item_matrix[row_index][col_index] < 2):
                            this.user_item_matrix[row_index][col_index] = 2

                    elif ((this.interactions_df['eventType'][i] == "FOLLOW" or this.interactions_df['eventType'][i] == "BOOKMARK") and
                            this.user_item_matrix[row_index][col_index] < 3):

                            this.user_item_matrix[row_index][col_index] = 3

                    elif ((this.interactions_df['eventType'][i] == "FOLLOW" or
                           this.interactions_df['eventType'][i] == "BOOKMARK") and
                            this.user_item_matrix[row_index][col_index] == 3):

                            this.user_item_matrix[row_index][col_index] = 4

    def improve_rating_matrix(this):
        #creation of a dataframe containing only effective user-item interactions
        temp_index = 0
        nonzero_len = len(this.user_item_matrix[np.nonzero(this.user_item_matrix)])
        temp_array = np.zeros((nonzero_len, 3))
        for i in range(len(this.users_enough_interactions)):
            for j in range(len(this.items_enough_rated)):
                if (this.user_item_matrix[j][i] != 0):
                    temp_array[temp_index][0] = this.users_enough_interactions[i]
                    temp_array[temp_index][1] = this.items_enough_rated[j]
                    temp_array[temp_index][2] = this.user_item_matrix[j][i]
                    temp_index = temp_index + 1
        this.user_item_df = pd.DataFrame(temp_array, columns=['personId', 'contentId', 'Rate'])
    
    def random_split(this):
        #random 80%-20% into training and test set
        this.interactions_train_df, this.interactions_test_df = train_test_split(this.user_item_df,
                                                                   stratify=this.user_item_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)