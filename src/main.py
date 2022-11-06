from corrcoef import *
from sentiment import *
from comment_injector import *
from tokenizor import *
from collaborativeF import *
from contentF import *


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


def dataPreProcessing(articles_df, interactions_df):
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
    # FOR DEBUG PURPOSES (cuts the data into smaller sets for faster debug computations)
    DEBUG = 0.25

    # path statement necessary to let the project work in different environments with respect to PyCharm
    here = os.path.dirname(os.path.abspath(__file__))
    nltk.download('punkt')
    nltk.download('stopwords')
    # CSV files readings
    articles_df = pd.read_csv(os.path.join(here, '../files/shared_articles.csv'))
    interactions_df = pd.read_csv(os.path.join(here, '../files/users_interactions.csv'))

    if 0 < DEBUG < 1:
        articles_df = articles_df.sample(frac=DEBUG)
        interactions_df = interactions_df.sample(frac=DEBUG)

    articles_df, articles_enough_df, users_enough_interactions, items_enough_rated = dataPreProcessing(articles_df,
                                                                                          interactions_df)
    user_item_df, user_item_matrix = userItemRating(interactions_df, users_enough_interactions,
                                  items_enough_rated)
    itemCollaborativeFiltering(articles_enough_df, articles_df, user_item_df)

    userCollaborativeFiltering(articles_df, user_item_df, interactions_df)

    #Tasks 4 to 6

    post_injection_df = []
    if (os.path.exists(os.path.join(here, '../files/comments.csv'))):
        post_injection_df = pd.read_csv(os.path.join(here, '../files/comments.csv'))
    else:
        post_injection_df = comments_injection(interactions_df, user_item_df, users_enough_interactions,
                                               items_enough_rated)

    print(post_injection_df)

    # SENTIMENT SCORE GENERATION

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

    pd_pearcorr(sentiment_scores_df[['baseScore', 'score']])

    # COMMENT LENGTH SCORE GENERATION

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

    pd_pearcorr(comment_length_scores_df[['baseScore', 'score']])

    # STOPWORD PROPORTION SCORE GENERATION

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

    pd_pearcorr(stopword_scores_df[['baseScore', 'score']])

    contentBasedFiltering(articles_enough_df, user_item_df)


if __name__ == '__main__':
    main()

# I studied also here: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54


# users = np.zeros(len(users_with_enough_interactions_df))
# personId = users_with_enough_interactions_df.index[0]
# interactions = interactions_df[interactions_df['personId'] == personId]


# print(users_with_enough_interactions_df.index[0])
# print(users_with_enough_interactions_df.iloc[0])
