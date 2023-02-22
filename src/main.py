from corrcoef import *
from comment_injector import *
from contentF import *
from contentFAlt import *
import matplotlib.pyplot as plt


def user_item_rating(interactions_df, users_enough_interactions, items_enough_rated):
    """
    Generate Item Ratings for each user based on their interactions with the articles: LIKE, COMMENT, BOOKMARK, etc
    The ratings are generated as such:
    a LIKE corresponds to a rate of 5
    a FOLLOW and a BOOKMARK together correspond to a rate of 4
    a FOLLOW or a BOOKMARK (but not both) corresponds to a rate of 3
    a COMMENT corresponds to a rate of 2
    just a VIEW corresponds to a rate of 1

    Args:
        interactions_df: a complete dataframe of user interactions with each article
        users_enough_interactions: a list of users that have interacted with at minimum three (3) articles
        items_enough_rated: a list of the articles that have at minimum two (2) different users interacting with it

    Returns:
        user_item_df: a dataframe containing all users and their respective rate for each article
        user_item_matrix: user_item_df in matrix representation
    """

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

            elif interactions_df['eventType'][i] == "COMMENT CREATED" and user_item_matrix[row_index][col_index] < 2:
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

    return user_item_df, user_item_matrix


def data_pre_processing(articles_df, interactions_df):
    """
        Receives and pe-processes the original dataset. Preprocessing includes excluding users with less than three (3)
        interactions (with different items), articles with less than two (2) different users interacting with them, etc

        Args:
            articles_df: the original dataframe of articles
            interactions_df: the original dataframe of user interactions with each article

        Returns:
            articles_df: the processed dataframe of articles
            articles_enough_df: and article df excluding articles with less than two (2) interactions
            users_enough_interactions: a list of users with at minimum three (3) different article interactions
            items_enough_rated: a list of articles with at least two (2) different users interactions
    """

    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']

    # array containing IDs of users that rated at least three different items
    user_interactions = interactions_df[['personId', 'contentId']].drop_duplicates().groupby(['personId'])[
        'contentId'].count()

    user_interactions_df = pd.DataFrame(
        {'personId': user_interactions.index, 'n_interactions': user_interactions.values})

    enough_user_interactions_df = user_interactions_df[user_interactions_df['n_interactions'] >= 3]
    users_enough_interactions = enough_user_interactions_df['personId'].to_numpy()
    users_enough_interactions = users_enough_interactions

    # array containing IDs of items that have been rated at least by two different users
    items_rated = interactions_df[['personId', 'contentId']].drop_duplicates().groupby(['contentId'])[
        'personId'].count()

    items_rated_df = pd.DataFrame({'contentId': items_rated.index, 'n_ratings': items_rated.values})

    enough_items_rated_df = items_rated_df[items_rated_df['n_ratings'] >= 2]
    items_enough_rated = enough_items_rated_df['contentId'].to_numpy()
    items_enough_rated = items_enough_rated

    articles_enough_df = pd.DataFrame()
    text = []
    lang = []
    content = []
    title = []

    for index, row in articles_df.iterrows():
        if row['contentId'] in items_enough_rated:
            content.append(row['contentId'])
            text.append(row['text'])
            lang.append(row['lang'])
            title.append(row['title'])
    articles_enough_df['contentId'] = content
    articles_enough_df['text'] = text
    articles_enough_df['lang'] = lang
    articles_enough_df['title'] = title

    return articles_df, articles_enough_df, users_enough_interactions, items_enough_rated


def main():
    """
        The main function of the project. Here take place all the initialisations and function calling for each task
    """

    # FOR DEBUG PURPOSES (cuts the data into smaller sets for faster debug computations)
    DEBUG = 0

    # Data preprocessing and preparation

    # path statement necessary to let the project work in different environments with respect to PyCharm
    here = os.path.dirname(os.path.abspath(__file__))

    # Necessary downloading for nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    # CSV files readings
    articles_df = pd.read_csv(os.path.join(here, '../files/shared_articles.csv'))
    interactions_df = pd.read_csv(os.path.join(here, '../files/users_interactions.csv'))

    # Sampling based on DEBUG
    if 0 < DEBUG < 1:
        interactions_df = interactions_df.sample(frac=DEBUG)
        articles_df = articles_df.loc[np.isin(articles_df["contentId"], interactions_df["contentId"])]

    # data pre-processing
    articles_df, articles_enough_df, users_enough_interactions, items_enough_rated = data_pre_processing(articles_df,
                                                                                                         interactions_df)
    # creation of item ratings
    user_item_df, user_item_matrix = user_item_rating(interactions_df, users_enough_interactions,
                                                      items_enough_rated)

    # ITEM-BASED COLLABORATIVE FILTERING (Task 2)
    item_coll_metrics = item_collaborative_filtering(articles_enough_df, articles_df, user_item_df)

    # USER-BASED COLLABORATIVE FILTERING (Task 3)
    user_coll_metrics = user_collaborative_filtering(articles_df, user_item_df, interactions_df)

    # Tasks 4 to 6
    if os.path.exists(os.path.join(here, '../files/comments.csv')):
        post_injection_df = pd.read_csv(os.path.join(here, '../files/comments.csv'))
    else:
        post_injection_df = comments_injection(interactions_df, user_item_df, users_enough_interactions,
                                               items_enough_rated)

    # Sampling based on DEBUG
    if 0 < DEBUG < 1:
        post_injection_df = post_injection_df[np.isin(post_injection_df["contentId"], articles_df["contentId"])]

    # SENTIMENT SCORE GENERATION (Task 4)

    sentiment_scores = []
    for i in post_injection_df.index:
        sentiment_scores.append([post_injection_df['personId'][i], post_injection_df['contentId'][i],
                                 post_injection_df['generatedScore'][i],
                                 sentiment_score(post_injection_df['comment'][i], False)])

    # dataframe creation
    sentiment_scores = np.array(sentiment_scores)

    sentiment_scores_df = pd.DataFrame(
        {'personId': sentiment_scores[:, 0], 'contentId': sentiment_scores[:, 1], 'baseScore': sentiment_scores[:, 2],
         'score': sentiment_scores[:, 3]})

    print("\n\nSENTIMENT SCORES:\n")
    print(sentiment_scores_df)

    plt.rcParams["figure.autolayout"] = True

    temp_df = sentiment_scores_df * [1, 1, 20, 1]
    ax = temp_df.plot(x='contentId', y='baseScore', kind='scatter', color='orange')
    sentiment_scores_df.plot(ax=ax, x='contentId', y='score', kind='scatter', color='blue')
    plt.show()

    pd_pear_corr(sentiment_scores_df[['baseScore', 'score']], True)

    # COMMENT LENGTH SCORE GENERATION (Task 5)

    comment_length_scores = []
    for i in post_injection_df.index:
        comment_length_scores.append([post_injection_df['personId'][i], post_injection_df['contentId'][i],
                                      post_injection_df['generatedScore'][i],
                                      tokenize_count(post_injection_df['comment'][i], True)])

    # dataframe creation
    comment_length_scores = np.array(comment_length_scores)

    comment_length_scores_df = pd.DataFrame(
        {'personId': comment_length_scores[:, 0], 'contentId': comment_length_scores[:, 1],
         'baseScore': comment_length_scores[:, 2], 'score': comment_length_scores[:, 3]})

    print("\n\nCOMMENT LENGTH SCORES:\n")
    print(comment_length_scores_df)

    plt.rcParams["figure.autolayout"] = True
    temp_df = comment_length_scores_df - [0, 0, 1, 0]
    temp_df = temp_df * [1, 1, 40, 1]
    ax = temp_df.plot(x='contentId', y='baseScore', kind='scatter', color='orange')
    comment_length_scores_df.plot(ax=ax, x='contentId', y='score', kind='scatter', color='blue')
    plt.show()

    pd_pear_corr(comment_length_scores_df[['baseScore', 'score']], True)

    # STOPWORD PROPORTION SCORE GENERATION (Task 6)

    stopword_scores = []
    for i in post_injection_df.index:
        stopword_scores.append([post_injection_df['personId'][i], post_injection_df['contentId'][i],
                                post_injection_df['generatedScore'][i],
                                tokenize_proportion(post_injection_df['comment'][i])])

    # dataframe creation
    stopword_scores = np.array(stopword_scores)

    stopword_scores_df = pd.DataFrame(
        {'personId': stopword_scores[:, 0], 'contentId': stopword_scores[:, 1], 'baseScore': stopword_scores[:, 2],
         'score': stopword_scores[:, 3]})

    print("\n\nSTOPWORD PROPORTION SCORES:\n")
    print(stopword_scores_df)

    plt.rcParams["figure.autolayout"] = True

    temp_df = stopword_scores_df * [1, 1, 0.2, 1]
    ax = temp_df.plot(x='contentId', y='baseScore', kind='scatter', color='orange')
    stopword_scores_df.plot(ax=ax, x='contentId', y='score', kind='scatter', color='blue')
    plt.show()

    pd_pear_corr(stopword_scores_df[['baseScore', 'score']], True)

    # SENTIMENT-BASED CONTENT FILTERING (Task 7)
    sentiment_cb_metrics = content_based_filtering(articles_enough_df, user_item_df)

    # SENTIMENT-VECTOR BASED CONTENT FILTERING (Task 8)
    sent_text_cb_metrics = content_based_filtering_alt(articles_enough_df, user_item_df)

    # CONCLUSION

    global_metrics_df = pd.DataFrame([item_coll_metrics, user_coll_metrics, sentiment_cb_metrics, sent_text_cb_metrics]) \
        .set_index('modelName')
    print(global_metrics_df)

    ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15, 8))
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points')
    plt.show()

if __name__ == '__main__':
    main()

# I studied also here: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
