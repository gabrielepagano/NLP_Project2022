from scipy import spatial

from sentiment import *
from tokenizor import *
from collaborativeF import *


class ContentBasedRecommender:
    MODEL_NAME = 'Content-Based'

    def __init__(self, item_ids, articles_scores_df, user_profiles, user_ids, interactions_full_indexed_df,
                 items_df=None):
        """
            This class is a Content Recommendation Model. It recommends based on a user's previous choices. The
            recommendation is aided by a sentiment analysis on each item / article but also text computations

            Args:
                item_ids: item indexing list
                articles_scores_df: the sentiment scores of each article
                user_profiles: all the user profiles created based on their interacted items
                user_ids: user indexing list
                interactions_full_indexed_df:
                items_df: a complete dataframe of the articles
        """

        self.user_ids = user_ids
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.item_ids = item_ids
        self.articles_scores_df = articles_scores_df
        self.articles_scores_df = self.articles_scores_df.sort_values(by=['Rate'], ascending=False)
        self.user_profiles = user_profiles
        self.items_df = items_df

    def get_model_name(self):
        """
            Returns:
                model_name: the name of the model
        """

        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, person_id):
        """
            Retrieves all similar items to the user's profile score, based on the sentiment and text analysis

            Args:
                person_id: the user's id

            Returns:
                recommended_items: all the similar items to user's profile
        """

        ind = np.where(self.user_ids == person_id)[0]
        myprofile = self.user_profiles[ind]
        cosine_similarities = []
        for item in self.item_ids:
            if item not in self.interactions_full_indexed_df.loc[person_id]:
                it = self.articles_scores_df['Rate'][self.articles_scores_df['contentId'] == item].tolist()[0]
                for mp in myprofile[0]:
                    cos = 1 - spatial.distance.cosine(mp, it)
                #cos = cosine_similarity([self.articles_scores_df['Rate'][self.articles_scores_df['contentId'] == item][0]], myprofile[0])[0][0]
                cosine_similarities.append(cos)
        recommended_items = pd.DataFrame()
        recommended_items['contentId'] = self.item_ids
        recommended_items['Rate'] = cosine_similarities
        recommended_items = recommended_items.sort_values(by='Rate', ascending=False)
        return recommended_items

    def recommend_items(self, user_id, items_to_ignore=None, topn=10, verbose=False):
        """
            Args:
                user_id: the id of the user that the model is recommending items to
                items_to_ignore: the items that should be ignored by the recommender
                topn: defaults to 10. Defines amount of items to be recommended
                verbose: defaults to False
            Returns:
                recommendations_df: the topn recommended items
        """

        if items_to_ignore is None:
            items_to_ignore = []
        similar_items = self._get_similar_items_to_user_profile(user_id)
        # Ignores items the user has already interacted

        recommendations_df = similar_items.head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[
                ['Rate', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


def get_item_profile(item_id, item_ids, articles_score_df):
    """
        Args:
            item_id: the item id
            item_ids: the item index
            articles_score_df: a complete dataframe of the article sentiment scores
        Returns:
            item_profile: the item sentiment profile
    """

    item_profile = articles_score_df['Rate'][articles_score_df['contentId'] == item_id]
    return item_profile


def get_item_profiles(ids, item_ids, articles_score_df):
    """
        Args:
            ids: item indexing
            item_ids: the item index
            articles_score_df: a complete dataframe of the article sentiment scores
        Returns:
            item_profiles_list: the list of item sentiment profiles
    """

    item_profiles_list = [get_item_profile(x, item_ids, articles_score_df) for x in ids]
    return item_profiles_list


def build_users_profile(person_id, interactions_indexed_df, item_ids, articles_score_df):
    """
        Args:
            person_id: the user's id
            interactions_indexed_df: a complete indexed dataframe of user interactions with articles
            item_ids: the item index
            articles_score_df: a complete dataframe of the article sentiment scores
        Returns:
            user_item_strengths_weighted_avg: the user's sentiment profile
    """

    interactions_person_df = interactions_indexed_df.loc[person_id]

    try:
        user_item_profiles = get_item_profiles(interactions_person_df['contentId'].to_numpy(), item_ids,
                                               articles_score_df)
    except AttributeError:
        user_item_profiles = get_item_profiles(np.array([interactions_person_df['contentId']]), item_ids,
                                               articles_score_df)

    user_item_strengths = np.array(interactions_person_df['Rate']).reshape(-1, 1)
    # Weighted average of item profiles by the interactions strength
    mul = [[a * b for x in user_item_profiles for a in x] for b in user_item_strengths]

    user_item_strengths_weighted_avg = [0, 0, 0, 0, 0]
    for a in mul:
        for b in a:
            i = 0
            for x in b:
                user_item_strengths_weighted_avg[i] += b
                i += 1

    for a in user_item_strengths_weighted_avg:
        a /= np.sum(user_item_strengths)


    return user_item_strengths_weighted_avg


def build_users_profiles(interactions_train_df, articles_enough_df, item_ids, articles_score_df):
    """
        Args:
            interactions_train_df: the train subset of interactions_full_df
            articles_enough_df: a list of articles that were interacted by a minimum of two (2) different users
            item_ids: the item index
            articles_score_df: a complete dataframe of the article sentiment scores
        Returns:
            user_profiles: the built user sentiment profiles
            interactions_indexed_df: relevant indexed dataframe
    """

    interactions_indexed_df = interactions_train_df[interactions_train_df['contentId']
                                                    .isin(articles_enough_df['contentId'])].set_index('personId')
    user_profiles = []
    for person_id in interactions_indexed_df.index.unique():
        user_profiles.append(build_users_profile(person_id, interactions_indexed_df, item_ids, articles_score_df))
    return user_profiles, interactions_indexed_df.index.unique()


def content_based_filtering_alt(articles_enough_df, interactions_full_df):
    """
        Performs sentiment-based content filtering. This model depends on item profiles created by performing vader
        sentiment analysis and general text computations on the article title and content.

        Args:
            articles_enough_df: a list of articles that have at least two (2) different users interacting with them
            interactions_full_df: a complete dataframe of all user ratings for the articles

        Returns:
            global_metrics: the metrics of the evaluation
    """

    # The User Sample ID
    user_id = -1479311724257856983

    print('\nPreparing data for Vector Content-Based Filtering model...\n')

    interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                   stratify=interactions_full_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)
    # Indexing by personId to speed up the searches during evaluation
    interactions_full_indexed_df = interactions_full_df.set_index('personId')
    interactions_train_indexed_df = interactions_train_df.set_index('personId')
    interactions_test_indexed_df = interactions_test_df.set_index('personId')

    title_text = []
    scores = []
    item_ids = articles_enough_df['contentId'].tolist()
    for index, row in articles_enough_df.iterrows():
        content = row['title'] + "" + row['text']
        title_text.append(content)
        neg, neu, pos = sentiment_scores(content, False)
        ln = tokenize_count(content, True)
        r = tokenize_proportion(content)
        prons = tokenize_count_pronouns(content)
        scores.append([pos, neg, ln, r, prons])
    articles_scores_df = pd.DataFrame()
    articles_scores_df['contentId'] = item_ids
    articles_scores_df['title_text'] = title_text
    articles_scores_df['Rate'] = scores

    user_profiles, user_indexes = build_users_profiles(interactions_train_df, articles_enough_df, item_ids,
                                                       articles_scores_df)

    user_profiles = np.array(user_profiles)
    ind = np.where(user_indexes == user_id)[0]

    myprofile = user_profiles[ind]

    print('\nThe user profile of the user with ID -1479311724257856983:\n')

    print(myprofile)

    print('\nPreparing data...\n')

    content_based_recommender_model = ContentBasedRecommender(item_ids, articles_scores_df, user_profiles,
                                                              user_indexes, interactions_full_indexed_df,
                                                              articles_enough_df)

    recommendations_df = content_based_recommender_model.recommend_items(user_id)

    print("\nThese are the top 10 items recommended to the user with ID -1479311724257856983 \n(just a sample one, "
          "if you want to change user just call the function with an other ID): \n")

    print(recommendations_df)

    print('Evaluating Content-Based Filtering model...')

    model_evaluator = modelEvaluator.ModelEvaluator(interactions_full_indexed_df, interactions_train_indexed_df,
                                                    interactions_test_indexed_df, articles_enough_df,
                                                    recommendations_df)

    global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model,
                                                                               recommendations_df)
    print('\nGlobal metrics:\n%s' % global_metrics)
    cb_detailed_results_df.head(10)

    return global_metrics
