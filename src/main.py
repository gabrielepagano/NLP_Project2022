from DataFrame import DataFrame
from corrcoef import *
from sentiment import *

if __name__ == "__main__" :
    df = DataFrame('../files/shared_articles.csv', '../files/users_interactions.csv')

    comments = ["I like it a lot. It was fun to read and very imformative!",
            "I didn't love it, but it was still ok. I found the final part interesting.",
            "It was rather dissapointing to say the least. It was repeating the same points over and over and had no information we didn't already know...",
            "I loved this article! I learned so many new things from it. I hope the author releases more great work!",
            "The article itself was fine, but I think the author should put more effort to not repeat imformation too much."]

    print("\n1st statement :")
    sentiment_scores(comments[0])
 
    print("\n2nd Statement :")
    sentiment_scores(comments[1])
 
    print("\n3rd Statement :")
    sentiment_scores(comments[2])

    print("\n4rd Statement :")
    sentiment_scores(comments[3])

    print("\n5rd Statement :")
    sentiment_scores(comments[4])

    print("\n\n\n")

    pd_pearcorr_rating_sentiment(df.user_item_df)

    print("\n\n\n")

    np1 = [4, 1, 5, 2, 3, 3, 4]
    np2 = [0.03, -0.1, 0.09, -0.06, 0.02, 0.03, 0.8]

    np_pearcorr_rating_sentiment(np1, np2)

