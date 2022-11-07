from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def sentiment_score(sentence, p):
    """
        Calculates an Overall Rate using Vader Sentiment Analysis

        Args:
            sentence: the sentence / paragraph to calculate sentiment analysis on
            p: boolean, if True it prints results

        Returns:
            score: an overall sentiment score in percentage form
    """

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    if p:
        print("Sentence Overall Rated As", end=" ")

        # decide sentiment as positive, negative and neutral
        if sentiment_dict['compound'] >= 0.05:
            print("Positive")

        elif sentiment_dict['compound'] <= - 0.05:
            print("Negative")

        else:
            print("Neutral")

    # represent sentiment score as a percentage
    score = sentiment_dict['compound'] * 50 + 50

    return score


def sentiment_scores(sentence, p):
    """
        Calculates three different sentiment scores for the provided sentence / paragraph using Vader Sentiment Analyzer

        Args:
            sentence: the sentence / paragraph to calculate sentiment analysis on
            p: boolean, if True it prints results

        Returns:
            neg: the negative sentiment score as percentage
            neu: the neutral sentiment score as percentage
            pos: the positive sentiment score as percentage
    """

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    if p:
        print("Overall sentiment dictionary is : ", sentiment_dict)
        print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
        print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
        print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

    return sentiment_dict['neg'] * 100, sentiment_dict['neu'] * 100, sentiment_dict['pos'] * 100
