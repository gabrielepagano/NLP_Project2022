from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# function to print sentiments
# of the sentence.
def sentiment_score(sentence, p):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    if p: print("Overall sentiment dictionary is : ", sentiment_dict)
    if p: print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
    if p: print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
    if p: print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

    if p: print("Sentence Overall Rated As", end=" ")

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        if p: print("Positive")

    elif sentiment_dict['compound'] <= - 0.05:
        if p: print("Negative")

    else:
        if p: print("Neutral")

    return sentiment_dict['compound'] * 50 + 50
