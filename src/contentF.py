import pandas as pd

from sentiment import *
from collaborativeF import *
from sklearn.metrics import mean_squared_error


class ContentBasedRecommender:
    MODEL_NAME = 'Content-Based'

    def __init__(self, item_ids, articles_scores_df, user_profiles, items_df=None):
        self.item_ids = item_ids
        self.articles_scores_df = articles_scores_df
        self.articles_scores_df = self.articles_scores_df.sort_values(by=['Rate'], ascending=False)
        self.user_profiles = user_profiles
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def get_similar_items_to_user_profile(self, person_id, user_indexes, interactions_full_indexed_df, topn=1000):
        ind = np.where(user_indexes == person_id)[0]
        myprofile = self.user_profiles[ind]
        differences = []
        for item in self.item_ids:
            if item not in interactions_full_indexed_df.loc[person_id]:
                mse = mean_squared_error(self.articles_scores_df['Rate'][self.articles_scores_df['contentId'] == item], myprofile)
                differences.append(mse)
        recommended_items = pd.DataFrame()
        recommended_items['contentId'] = self.item_ids
        recommended_items['Errors'] = differences
        recommended_items = recommended_items.sort_values(by='Errors')
        return recommended_items




def get_item_profile(item_id, item_ids, articles_score_df):
    item_profile = articles_score_df['Rate'][articles_score_df['contentId'] == item_id]
    return item_profile


def get_item_profiles(ids, item_ids, articles_score_df):
    item_profiles_list = [get_item_profile(x, item_ids, articles_score_df) for x in ids]
    return item_profiles_list


def build_users_profile(person_id, interactions_indexed_df, item_ids, articles_score_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['contentId'], item_ids, articles_score_df)

    user_item_strengths = np.array(interactions_person_df['Rate']).reshape(-1, 1)
    # Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(np.multiply(user_item_profiles, user_item_strengths), axis=0) / np.sum(
        user_item_strengths)

    return user_item_strengths_weighted_avg


def build_users_profiles(interactions_train_df, articles_enough_df, item_ids, articles_score_df):
    interactions_indexed_df = interactions_train_df[interactions_train_df['contentId'] \
        .isin(articles_enough_df['contentId'])].set_index('personId')
    user_profiles = []
    for person_id in interactions_indexed_df.index.unique():
        user_profiles.append(build_users_profile(person_id, interactions_indexed_df, item_ids, articles_score_df))
    print(user_profiles)
    return user_profiles, interactions_indexed_df.index.unique()


def contentBasedFiltering(articles_enough_df, interactions_full_df):
    user_id = -1479311724257856983

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
        scores.append(sentiment_score(content, False))
    articles_scores_df = pd.DataFrame()
    articles_scores_df['contentId'] = item_ids
    articles_scores_df['title_text'] = title_text
    articles_scores_df['Rate'] = scores

    user_profiles, user_indexes = build_users_profiles(interactions_train_df, articles_enough_df, item_ids, articles_scores_df)

    user_profiles = np.array(user_profiles)
    ind = np.where(user_indexes == user_id)[0]

    myprofile = user_profiles[ind]
    print(myprofile)

    content_based_recommender_model = ContentBasedRecommender(item_ids, articles_scores_df, user_profiles,
                                                              articles_enough_df)
    recommendations_df = content_based_recommender_model.get_similar_items_to_user_profile(user_id, user_indexes, interactions_full_indexed_df, topn=1000)

    print(recommendations_df)

    model_evaluator = ModelEvaluator(interactions_full_indexed_df, interactions_train_indexed_df,
                                     interactions_test_indexed_df, articles_enough_df, recommendations_df)


    print('Evaluating Content-Based Filtering model...')
    cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
    print('\nGlobal metrics:\n%s' % cb_global_metrics)
    cb_detailed_results_df.head(10)
