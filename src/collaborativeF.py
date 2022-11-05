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

# Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100


class ModelEvaluator:

    def __init__(self, interactions_full_indexed_df, interactions_train_indexed_df, interactions_test_indexed_df,
                 articles_df, item_popularity_df):
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df
        self.articles_df = articles_df
        self.item_popularity_df = item_popularity_df

    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, self.interactions_full_indexed_df)
        all_items = set(self.articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def evaluate_model_for_user(self, model, person_id):
        # Getting the items in test set
        interacted_values_testset = self.interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id,
                                               items_to_ignore=get_items_interacted(person_id,
                                                                                    self.interactions_train_indexed_df),
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id,
                                                                               sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                               seed=item_id % (2 ** 32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['contentId'].values
            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count': hits_at_5_count,
                          'hits@10_count': hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10
                          }
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(self.interactions_test_indexed_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
            .sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        list_relevant_items = self.item_popularity_df['contentId'].to_numpy()

        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10,
                          'NDCG@5': ndcg_at_k(list_relevant_items, 5, 0),
                          'NDCG@10': ndcg_at_k(list_relevant_items, 10, 0)}
        return global_metrics, detailed_results_df


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


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


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


def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


def itemCollaborativeFiltering(articles_enough_df, articles_df, interactions_full_df):
    interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                   stratify=interactions_full_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)
    # Indexing by personId to speed up the searches during evaluation
    interactions_full_indexed_df = interactions_full_df.set_index('personId')
    interactions_train_indexed_df = interactions_train_df.set_index('personId')
    interactions_test_indexed_df = interactions_test_df.set_index('personId')
    item_popularity_df = interactions_full_indexed_df.groupby('contentId')['Rate'].sum().sort_values(
        ascending=False).reset_index()
    model_evaluator = ModelEvaluator(interactions_full_indexed_df, interactions_train_indexed_df,
                                     interactions_test_indexed_df, articles_df, item_popularity_df)
    print("\n Here there are the most ten relevant items basing on the sum of ratings:\n")
    print(item_popularity_df.head(10))
    popularity_model = PopularityRecommender(interactions_test_indexed_df, item_popularity_df, articles_df)
    print('\nEvaluating Popularity recommendation model...\n')
    pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
    print('\nGlobal metrics:\n%s' % pop_global_metrics)
    print(pop_detailed_results_df.head(10))

    print(
        "\nFirst 20 relevant items for the user ID -1479311724257856983 \n(just a sample one, if you want to change user just call the function with an other ID): \n")
    print(inspect_interactions(articles_df, interactions_train_indexed_df, interactions_test_indexed_df,
                               -1479311724257856983, test_set=False).head(20))


def contentBasedFiltering(articles_enough_df, user_item_df):
    # random 80%-20% into training and test set
    interactions_train_df, interactions_test_df = train_test_split(user_item_df,
                                                                   stratify=user_item_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)
    interactions_train_df = interactions_train_df.set_index('personId')
    interactions_test_df = interactions_test_df.set_index('personId')
    stop_words_en = set(stopwords.words('english'))
    stop_words_pt = set(stopwords.words('portuguese'))
    stop_words_sp = set(stopwords.words('spanish'))
    stop_words = stop_words_en.union(stop_words_sp, stop_words_pt)
    # splitting between training and test data
    data = []
    # let's order articles by their ID
    articles_enough_df = articles_enough_df.sort_values(by=['contentId'])
    # array containing articles ordered by their appearence in articles_df
    articles_arr = articles_enough_df['contentId'].to_numpy()
    for index, row in articles_enough_df.iterrows():
        data.append(row['text'])
    # object that turns text into vectors
    vectorizer = TfidfVectorizer(stop_words=stop_words,
                                 ngram_range=(1, 3),
                                 analyzer='word')
    # create doc-term matrix
    dtm = vectorizer.fit_transform(data)
    dtm = csr_matrix(dtm)
    interactions_train_df = interactions_train_df.sort_values(by=['personId'])
    interactions_test_df = interactions_test_df.sort_values(by=['personId'])
    Rates = []
    for i in range(len(interactions_test_df.index)):
        user = interactions_test_df.index[i]
        if user in interactions_train_df.index:
            print(user)
            print(interactions_test_df['contentId'].loc[user])
            if len(interactions_test_df['contentId'].loc[user].items()) == 1:
                print("mammt")
                print(interactions_test_df['contentId'].loc[user].items())
            for index, item in interactions_test_df['contentId'].loc[user].items():
                test_ind = np.where(articles_arr == item)[0][0]
                sum = 0
                weighted_sum = 0
                for indx, it in interactions_train_df['contentId'].loc[user].items():
                    if it != item:
                        train_ind = np.where(articles_arr == it)[0][0]
                        weight = cosine_similarity(dtm[train_ind], dtm[test_ind])[0][0]
                        rate = interactions_train_df[interactions_train_df['contentId'] == it]['Rate'].loc[user]
                        weighted_sum = weighted_sum + rate * weight
                        sum = sum + weight
                Rates.append(round(weighted_sum / sum))
                print(Rates)
        else:
            Rates.append(3)
            print(Rates)
