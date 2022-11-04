import numpy as np
import pandas as pd
import random as rand
import csv
import os



def comments_injection(interactions_df, user_item_df, users_enough_interactions, items_enough_rated):
    # path statement necessary to let the project work in different environments with respect to PyCharm
    here = os.path.dirname(os.path.abspath(__file__))
    #CSV files readings
    hotel_reviews_df = pd.read_csv(os.path.join(here, '../files/Hotel_Reviews.csv'))


    post_injection_df = []
    track = 0
    interactions_comments_df = interactions_df.loc[(interactions_df['eventType'] == "COMMENT CREATED"), ['personId', 'contentId']]
    interactions_comments_df = interactions_comments_df.reset_index(drop=True)
    print(interactions_comments_df)
    for j in hotel_reviews_df.index:
        if(track >= len(interactions_comments_df['personId'])): break
        for i in interactions_comments_df.index:
            #index bound extra check
            if i + track >= len(interactions_comments_df['personId']): break

            if ((np.isin(interactions_comments_df['personId'].iloc[i+track], users_enough_interactions)) and
                    (np.isin(interactions_comments_df['contentId'][i+track], items_enough_rated))):
                #retrieving the correct generated score from user_item_df
                rate = int(user_item_df.loc[(user_item_df['personId'] == interactions_comments_df['personId'][i+track]) & (user_item_df['contentId'] == interactions_comments_df['contentId'][i+track])]['Rate'])
                #10% of comments will be randomly assigned
                if(rand.randrange(0, 100) < 10):
                    #both positive and negative comments are added on the randomised assignment
                    post_injection_df.append([interactions_comments_df['personId'][i+track], interactions_comments_df['contentId'][i+track], hotel_reviews_df['Reviewer_Score'].get(j), rate, hotel_reviews_df['Positive_Review'][j] + "\n" + hotel_reviews_df['Negative_Review'][j]])
                    break
                #mapping hotel review scores to user item generated scores as follows:
                #[7-10] -> 5, [4-7) -> 3 or 4, [0,4) -> 1 or 2
                elif ((float(hotel_reviews_df['Reviewer_Score'].get(j)) >= 7) and 
                    (float(hotel_reviews_df['Reviewer_Score'].get(j)) <= 10) and
                    (rate == 5)):
                    #positive comment gets added
                    post_injection_df.append([interactions_comments_df['personId'][i+track], interactions_comments_df['contentId'][i+track], hotel_reviews_df['Reviewer_Score'].get(j), rate, hotel_reviews_df['Positive_Review'][j]])
                    break
                elif ((float(hotel_reviews_df['Reviewer_Score'].get(j)) >= 4) and
                        (float(hotel_reviews_df['Reviewer_Score'].get(j)) < 7) and
                        (rate == 3 or rate == 4)):
                    #both positive and negative comments are added
                    post_injection_df.append([interactions_comments_df['personId'][i+track], interactions_comments_df['contentId'][i+track], hotel_reviews_df['Reviewer_Score'].get(j), rate, hotel_reviews_df['Positive_Review'][j] + "\n" + hotel_reviews_df['Negative_Review'][j]])
                    break
                elif ((float(hotel_reviews_df['Reviewer_Score'].get(j)) >= 0) and
                        (float(hotel_reviews_df['Reviewer_Score'].get(j)) < 4) and
                        (rate == 1 or rate == 2)):
                    #only negative comment is added
                    post_injection_df.append([interactions_comments_df['personId'][i+track], interactions_comments_df['contentId'][i+track], hotel_reviews_df['Reviewer_Score'].get(j), rate, hotel_reviews_df['Negative_Review'][j]])
                    break
        track += 1
    post_injection_df = np.array(post_injection_df)

    #dataframe creation
    post_injection_df = pd.DataFrame(
        {'personId': post_injection_df[:, 0], 'contentId': post_injection_df[:, 1], 'userReview':post_injection_df[:, 2], 'generatedScore':post_injection_df[:, 3], 'comment': post_injection_df[:, 4]})
    
    #print df and csv file creation
    print(post_injection_df)
    create_csv(post_injection_df, here)

    #returns the created df
    return post_injection_df

#a simple csv file creator function
def create_csv(post_injection_df, here):

    header = ['personId', 'contentId', 'userReview', 'generatedScore', 'comment']

    with open(os.path.join(here, '../files/comments.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(post_injection_df.to_numpy())
