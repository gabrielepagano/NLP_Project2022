import numpy as np
import pandas as pd
import random

# Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100


class ModelEvaluator:

    def __init__(self, interactions_full_indexed_df, interactions_train_indexed_df, interactions_test_indexed_df,
                 articles_df, rec_items_df):
        """
            This class is a filtering model evaluator, and it performs RECALL and NDCG Evaluations.

            Args:
                interactions_full_indexed_df: an indexed (on personId) dataframe of all user interactions
                interactions_train_indexed_df: the train sub-set of interactions_full_indexed_df
                interactions_test_indexed_df: the test sub-set of interactions_full_indexed_df
                articles_df: a complete dataframe of the articles
                rec_items_df: a dataframe of potential item recommendations
        """

        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df
        self.articles_df = articles_df
        self.item_popularity_df = rec_items_df

    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        """
            Identifies the items the provided user has not interacted with yet

            Args:
               person_id: the user's id
               sample_size: the size of the sample
               seed: default at 42

            Returns:
                set: the set of not interacted items for the provided user
        """

        interacted_items = get_items_interacted(person_id, self.interactions_full_indexed_df)
        all_items = set(self.articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        """
            Args:
                item_id: the article id
                recommended_items: the list of recommended items
                topn: a mex value number

            Returns:
                hit: the hit
                index: the index of hit
        """

        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def evaluate_model_for_user(self, model, person_id):
        """
            Evaluates a model for a single provided user

            Args:
                model: the evaluated model
                person_id: the user's id

            Returns:
                person_metrics: the evaluation metrics for the provided user
        """

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
                                                                               sample_size=
                                                                               EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                               seed=item_id % (2 ** 32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100
            # non-interacted items
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

    def evaluate_model(self, model, items):
        """
            Evaluates a model for a single provided user

            Args:
                model: the evaluated model
                items: a global articles rating if needed by the evaluated model

            Returns:
                global_metrics: the global metrics
                detailed_results_df: the detailed results dataframe
        """

        people_metrics = []
        for idx, person_id in enumerate(list(self.interactions_test_indexed_df.index.unique().values)):
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

        global_ndcg_at_5 = ndcg_at_k(items['Rate'].to_numpy(), 5, 0)
        global_ndcg_at_10 = ndcg_at_k(items['Rate'].to_numpy(), 10, 0)

        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10,
                          'ndcg@5': global_ndcg_at_5,
                          'ndcg@10': global_ndcg_at_10}
        return global_metrics, detailed_results_df.drop(['hits@5_count'], axis=1).drop(['hits@10_count'], axis=1)


def dcg_at_k(r, k, method=0):
    """
    Score is discounted cumulative gain (dcg)
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
    """
    Score is normalized discounted cumulative gain (ndcg)
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


def get_items_interacted(person_id, interactions_df):
    """
    Args:
        person_id: the user's id
        interactions_df: a complete dataframe of user interactions

    Returns:
        set: a set of interacted items for the user
    """

    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']

    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])
