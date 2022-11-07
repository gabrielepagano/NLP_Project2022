# Natural Language Processing and Text Mining Project on Recommender Systems

<p align="center">
  <img width="500" src="files/unioulu_logo.png" alt="UniOulu Logo" />
  <br>
</p>

## Contributors

- [__Gabriele Pagano__](https://github.com/gabrielepagano) (2207403)
- [__Antonios Saviolidis Alexandris__](https://github.com/AnthonyAl) (2207364)


## Description of the files

|File|Description|
|---------------|-----------|
| `'files' folder` | contains the two CSV files provided as data source (shared_articles and user_interactions), the Uni Oulu logo and a file (comments.csv) in which there are a series of comments used to perform sentiment analysis |
| `collaborativeF.py` | performs item-related and user-related collaborative filtering |
| `comment_injector.py` | takes the 'comments.csv' file and injects into our dataset the comments inside it|
| `contentF.py` | performs content-based filtering |
| `contentFAlt.py` | performs content-based filtering with the alternative definition of vector|
| `corrcoef.py` | calculates, given a dataframe, the Pearson correlation coefficient between its columns |
| `main.py` | launch file that runs the whole project and contains the data pre-processing and rating algorithm functions |
| `modelEvaluator.py` | contains the function that evaluates the given recommender model |
| `sentiment.py` | performs Vader sentiment analysis |
| `tokenizor.py` | contains some functions that deal with tokenization and text analysis |



## Python libraries

Here there is a list of the Python libraries that should be installed in order to correct execute the code:

- __numpy:__ used to deal with matrixes and arrays
- __pandas:__ used to deal with dataframes
- __sklearn:__ provided us various utilities to deal with evaluation of recommender system
- __nltk:__ provided us stopwords in different languages and tokenization utilities
- __scipy:__ provided us the method to perform cosine similarity
- __vaderSentiment:__ used to perform sentiment analysis


To install each of these libraries open a terminal and run the following command:

```
pip install <library_name>
```


__NB:__ the above command requires 'pip' already installed on your terminal. If you do not have it please read the guide in the following link: https://pip.pypa.io/en/stable/installation/



## How to run the project

In order to run the whole project you have just to run the 'main.py' file


