import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from main import userItemRating
import modelEvaluator

class PopularityRecommender:

    MODEL_NAME = 'Popularity'

    def __init__(self, interactions_test_indexed_df, popularity_df, items_df=None):
        self.interactions_test_indexed_df = interactions_test_indexed_df
        self.popularity_df = popularity_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
            .sort_values('Rate', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[
                ['Rate', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


def inspect_interactions(articles_df, interactions_train_indexed_df, interactions_test_indexed_df, person_id,
                         test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df
    return interactions_df.loc[person_id].merge(articles_df, how='left',
                                                left_on='contentId',
                                                right_on='contentId') \
        .sort_values('Rate', ascending=False)[['Rate',
                                               'contentId',
                                               'title', 'url', 'lang']]


def inspect_interactions_user(user_id, interactions_df, users_enough_interactions, items_enough_rated):
    # the amount of most similar users chosen for recommendation
    top = 5

    user_item_df, user_item_matrix = userItemRating(interactions_df, users_enough_interactions, items_enough_rated)

    # The Sample User's content scores
    user_index = np.where(users_enough_interactions == user_id)[0][0]

    similar_users = []
    for i in range(len(users_enough_interactions)):
        if i != user_index:
            similar_users.append(
                [i, cosine_similarity([user_item_matrix[:, i]], [user_item_matrix[:, user_index]])[0][0]])

    similar_users = np.array(similar_users)
    ind = np.argsort(similar_users[:, -1])
    similar_users = similar_users[ind]

    # sorted in asc order, therefore most similar users are put in the end of the array
    similar_users = similar_users[-top:]

    # simply initialising the items array
    items = np.zeros(len(items_enough_rated))

    # for each item there will be a summed score based on the ratings of each similar user
    for similar_user in similar_users:
        s_u_index = int(similar_user[0])
        for i in range(len(items_enough_rated)):
            s = user_item_matrix[:, s_u_index]
            items[i] += s[i]

    item_scores_df = pd.DataFrame()
    item_scores_df['contentId'] = items_enough_rated
    item_scores_df['Rate'] = items

    item_scores_df = item_scores_df.sort_values(by='Rate',
                                                ascending=False)

    return item_scores_df


def itemCollaborativeFiltering(articles_enough_df, articles_df, interactions_full_df):
    # The User Sample's ID
    user_id = -1479311724257856983

    print('\nPreparing data for Item-based Collaborative Filtering model...\n')

    interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                   stratify=interactions_full_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)
    # Indexing by personId to speed up the searches during evaluation
    interactions_full_indexed_df = interactions_full_df.set_index('personId')
    interactions_train_indexed_df = interactions_train_df.set_index('personId')
    interactions_test_indexed_df = interactions_test_df.set_index('personId')

    items = inspect_interactions(articles_df, interactions_train_indexed_df, interactions_test_indexed_df,
                                 user_id, test_set=True)

    item_popularity_df = interactions_full_indexed_df.groupby('contentId')['Rate'].sum().sort_values(
        ascending=False).reset_index()
    model_evaluator = modelEvaluator.ModelEvaluator(interactions_full_indexed_df, interactions_train_indexed_df,
                                     interactions_test_indexed_df, articles_df, item_popularity_df)
    print("\n Here there are the ten most relevant items basing on the sum of ratings:\n")
    print(item_popularity_df.head(10))
    popularity_model = PopularityRecommender(interactions_test_indexed_df, item_popularity_df, articles_df)
    print('\nEvaluating Popularity recommendation model...\n')
    pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model, items)
    print('\nGlobal metrics:\n%s' % pop_global_metrics)
    print(pop_detailed_results_df.head(10))

    print(
        "\nFirst 20 relevant items for the user ID -1479311724257856983 \n(just a sample one, if you want to change user just call the function with an other ID): \n")

    print(items.head(20))


def userCollaborativeFiltering(articles_df, interactions_full_df, interactions_df):
    # The User Sample's ID
    user_id = -1479311724257856983

    print('\nPreparing data for User-based Collaborative Filtering model...\n')

    interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                   stratify=interactions_full_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)
    # Indexing by personId to speed up the searches during evaluation
    interactions_full_indexed_df = interactions_full_df.set_index('personId')
    interactions_train_indexed_df = interactions_train_df.set_index('personId')
    interactions_test_indexed_df = interactions_test_df.set_index('personId')

    users_enough_interactions_train = interactions_train_df['personId'].unique()
    users_enough_interactions_test = interactions_test_df['personId'].unique()
    items_enough_rated_train = interactions_train_df['contentId'].unique()
    items_enough_rated_test = interactions_test_df['contentId'].unique()

    item_scores_df = inspect_interactions_user(user_id, interactions_df, users_enough_interactions_test, items_enough_rated_test)

    print(
        "\nFirst 10 most relevant items for the user ID -1479311724257856983 \n(just a sample one, if you want to change user just call the function with an other ID): \n")

    print(item_scores_df.head(10))

    model_evaluator = modelEvaluator.ModelEvaluator(interactions_full_indexed_df, interactions_train_indexed_df,
                                     interactions_test_indexed_df, articles_df, item_scores_df)

    popularity_model = PopularityRecommender(interactions_test_indexed_df, item_scores_df, articles_df)
    print('\nEvaluating Recommendation model...\n')
    pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model, item_scores_df)

    print('\nGlobal metrics:\n%s' % pop_global_metrics)
    print(pop_detailed_results_df.head(10))