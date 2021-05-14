#!/usr/bin/env python
# coding: utf-8

# # Book recommendation system project

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

import warnings
import os
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO

from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


users = pd.read_csv('BX-Users.csv', error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1') #encoding = "latin-1"
users.shape


# In[ ]:


books = pd.read_csv('BX-Books.csv', error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1') #encoding = "latin-1
books.shape


# In[ ]:


ratings = pd.read_csv('BX-Book-Ratings.csv', error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1')
ratings.shape


# In[ ]:


books.head()


# In[ ]:


users.head()


# In[ ]:


ratings.head()


# In[ ]:


#mihaela to delete
print(ratings['Book-Rating'].value_counts())


# ### Merging data sets

# Merge Ratings and Users, dropping users who gave no ratings.

# In[ ]:


print(f'How many user IDs in users: {users["User-ID"].nunique()}')
print(f'How many user IDs in ratings: {ratings["User-ID"].nunique()}')


# In[ ]:


data = pd.merge(ratings, users, on='User-ID', how='inner')


# In[ ]:


print(f'New data size: {data.shape}')
print(f'Total number of users: {users["User-ID"].nunique()}')
print(f'Number of users left (those with at least one review): {data["User-ID"].nunique()}')


# Merge data (Ratings + Users) and Books, dropping Books with no review.

# In[ ]:


print(f'How many user ISBNs in data (that is, in original Ratings): {data["ISBN"].nunique()}')
print(f'How many user ISBNs in Books: {books["ISBN"].nunique()}')


# In[ ]:


data = pd.merge(data, books, on='ISBN', how='inner')

data.columns


# In[ ]:


#book ratings left after the 2 merges
data['Book-Rating'].value_counts()


# In[ ]:


# Droping image columns
data.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'],axis=1,inplace=True)


# ### EDA

# In[ ]:


data.head(10)


# ### How many books and users do we have?

# In[ ]:


print('Number of books: ', data['ISBN'].nunique())


# In[ ]:


print('Number of users: ',data['User-ID'].nunique())


# #### Missing values

# In[ ]:


print('Missing data [%]')
round(data.isnull().sum() / len(data) * 100, 4)


# In[ ]:


print(f'Missing book author: {data["Book-Author"].isnull().sum()}')
print(f'Missing publisher: {data["Publisher"].isnull().sum()}')


# In[ ]:


#percentage of missing data from the users dataset relative to the number of users
plt.figure()
(users.isnull().sum()/len(users)*100).plot.bar()
plt.ylim((0,50))
plt.xlabel("User features")
plt.xticks(rotation=0)
plt.ylabel("% missing")
plt.title("Percentage of missing features in the users dataset", fontsize=16)


# In[ ]:


sns.distplot(data['Age'].dropna(), kde=False)


# In[ ]:


print('Number of outliers: ', sum(data['Age'] > 100))


# In[ ]:


print(data['Book-Rating'].value_counts())


# In[ ]:


plt.title('Book ratings distribution', fontsize=16)
sns.countplot(x='Book-Rating', data=data)


# In[ ]:


#data['Book-Rating'] = data['Book-Rating'].replace(0, None) #this doesn't work as expected
data['Book-Rating'] = data['Book-Rating'].replace({0:np.nan})


# In[ ]:


#sanity check -> this number should match the 0 count two cells above
print(f'Number of ratings with na value: {data["Book-Rating"].isna().sum()}')


# In[ ]:


plt.title('Actual book rating distribution', fontsize=16)
sns.countplot(x='Book-Rating', data=data)


# In[ ]:


print('Average book rating: ', round(data['Book-Rating'].mean(), 2))


# In[ ]:


data.isnull().sum()


# ## Feature Engineering

# In[ ]:


#data['Book-Rating'] = data['Book-Rating'].replace(0, None)
#0 values were already replaced with np.na a few cells above


# In[ ]:


temp = data.copy()


# In[ ]:


data = temp.copy()


# ## Age

# How many null values ?

# In[ ]:


data['Age'].isna().sum()


# What is the distribution of age ?

# In[ ]:


sns.distplot(data['Age'].dropna(), kde=False)


# How many entries over 90 yo ?

# In[ ]:


(data['Age']>90).sum()


# Age over 90 years old is most likely a mistake.

# In[ ]:


#Handle outliers -> replace with na
data['Age'] = np.where(data['Age']>90, None, data['Age'])


# In[ ]:


# Impute nulls - Categorical features
data[['Book-Author', 'Publisher']] = data[['Book-Author', 'Publisher']].fillna('Unknown')


# In[ ]:


# Check cat features
data[['Book-Author', 'Publisher']].isnull().sum()


# In[ ]:


median = data["Age"].median()
std = data["Age"].std()
is_null = data["Age"].isnull().sum()
rand_age = np.random.randint(median - std, median + std, size = is_null)
age_slice = data["Age"].copy()
age_slice[pd.isnull(age_slice)] = rand_age
data["Age"] = age_slice
data["Age"] = data["Age"].astype(int)


# In[ ]:


# Extract features country of the user
data['Country'] = data['Location'].apply(lambda row: str(row).split(',')[-1])


# In[ ]:


data['Country'].head()


# Minimize sample size otherwise too big for execution in deepnote, n=8000

# In[ ]:


data_min = data.sample(n=8000).reset_index(drop=True)


# Generate a random sample of 2 users for testing purposes

# In[ ]:


random_sample = data_min.sample(n = 2, random_state=42) 
random_sample


# ## 1. Content-based Recommender System
# 
# **Problem Formulation:**    
# Build a recommender system that recommends based on book titles. So if our user gives us a book title, our goal is to recommend books that have similar titles.

# In[ ]:


df_books = data_min[['ISBN','Book-Title','Book-Author','Year-Of-Publication']].drop_duplicates()
df_books.shape


# In[ ]:


# Calculate the word count for book title
df_books['word_count'] = df_books['Book-Title'].apply(lambda x: len(str(x).split()))

# Plotting the word count
df_books['word_count'].plot(
    kind='hist',
    bins = 30,
    figsize = (12,8),title='Word Count Distribution for book title')


# In[ ]:


df_books.head()


# **TF(Term Frequency)-IDF(Inverse Document Frequency) **  
# - TFIDF(word) = TF(Document, Word) * IDF (Word)  
# - The TF of a word is the frequency of a word (i.e. number of times it appears) in a document.    
# - The IDF of a word is the measure of how significant that term is in the whole corpus; log(Nr of documents / Nr of documents containing word)
# - Unigrams vs. Bigrams: Bigram is 2 consecutive words in a sentence. E.g. “The boy is playing football”. The bigrams here are: "The boy", "boy is"...
# 
# More: https://www.youtube.com/watch?v=ouEVPRMHR1U 

# **Cosine Similarity:**  
# Cosine similarity is a measure of similarity between two non zero vectors. One of the beautiful thing about vector representation is we can now see how closely related two sentence are based on what angles their respective vectors make.
# 
# - Cosine value ranges from -1 to 1.
# - If two vectors make an angle 0, then cosine value would be 1, which in turn would mean that the sentences are closely related to each other.
# - If the two vectors are orthogonal, i.e. cos 90 then it would mean that the sentences are almost unrelated. 
# 
# Source: https://www.linkedin.com/pulse/content-based-recommender-engine-under-hood-venkat-raman/

# In[ ]:


# Function for recommending books based on Book title. It takes book title as an input.
def recommend(title):
    
    # Convert the index into series
    indices = pd.Series(data_min.index, index = data_min['Book-Title'])
    
    #Natural Language Processing: converting the book title (str) into vectors bigrams 
        #- ngram_range = unigram/bigram/trigram ...
        #- min_df = When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
    tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df = 1)
    tfidf_matrix = tf.fit_transform(data_min['Book-Title'])

    #print(tfidf_matrix)
    #print(tfidf_matrix.shape)
    
    # Calculating the similarity measures based on Cosine Similarity
    sg = cosine_similarity(tfidf_matrix,tfidf_matrix)

    # Get the index corresponding to original_title       
    idx = indices[title]
    # Get the pairwsie similarity scores 
    sig = list(enumerate(sg[idx]))

    # Sort the books
    sig = sorted(sig, key=lambda x: x[1], reverse=True)
    # Scores of the 3 most similar books 
    sig = sig[1:4]

    # Book indicies
    book_indices = [i[0] for i in sig]

     
    # Top 5 book recommendation
    rec = data_min[['Book-Title', 'Book-Author', 'Year-Of-Publication']].iloc[book_indices]
    print("After reading '" + title + "' make sue to check out those books:")
    print("--------------------") 
    counter = 1
    for index,row in rec.iterrows():
        print(f"Recommendation {counter}:")
        print(f"{rec['Book-Title'][index]} by {rec['Book-Author'][index]} with {round(sig[counter - 1][1], 3)} similarity score") 
        print("--------------------") 
        counter = counter + 1


# In[ ]:


# test recommendation based on random book title 
#recommend(df_books['Book-Title'].iloc[4])


# In[ ]:


'''for index,row in random_sample.iterrows():
    #selected_user = data_min[data_min['Book-Title'] == random_sample['Book-Title'][index]].sort_values(by='Book-Rating', ascending=False).head(1)["User-ID"]
    #print(f"For user {selected_user} we recommend the following.")
    print("----------------------------------------") 
    recommend(random_sample['Book-Title'][index])
    '''


# ## 2. Collaborative Filtering Memory-Based Recommender System

# ### Problem Formulation:
# Build a recommender system that recommends books based on the users similarity in books rating.

# In[ ]:


df_users = data[['User-ID', 'Age', 'Country','ISBN','Book-Rating','Book-Title']]
df_users.shape


# In[ ]:


df_users_min = df_users.sample(n=8000).reset_index(drop=True)


# In[ ]:


df_users_min.hist(figsize=(10,8))


# In[ ]:


df_users_min.head()


# ### Creating the matrix

# In[ ]:


matrix=df_users_min.pivot(index="User-ID",columns="ISBN",values="Book-Rating")


# In[ ]:


matrix.fillna(0,inplace=True)


# #### Cosine similiarity between the users
# 

# In[ ]:


cos_sim = cosine_similarity(matrix)
np.fill_diagonal(cos_sim,0)        # zero here means that both ids are same,it should be 1 here but i am using 0 so as to ease further coding process
rec_cos=pd.DataFrame(cos_sim,index=matrix.index)
rec_cos.columns=matrix.index
rec_cos.head()


# ##### Validating our result using user id

# In[ ]:


df_users_min[df_users_min["User-ID"]==16795.0][["Book-Title","Book-Rating"]].head()


# ### Building a function to show top 3 users that are similiar to input random user
# 

# In[ ]:


def sim(userid,n):          # userid is the id for which recommendations has to be made, n represents total no. of similiar users wanted 
    print(np.array(rec_cos[userid].sort_values(ascending=False).head(n).index))


# In[ ]:


print(np.array(rec_cos[150968.0].sort_values(ascending=False).head(3).index))


# In[ ]:


data[data["User-ID"]==200226.0][["Age","Location"]].head()


# In[ ]:


data[data["User-ID"]==20119.0][["Age","Location"]].head()


# In[ ]:


data[data["User-ID"]==125039.0][["Age","Location"]].head()


# In[ ]:


print(np.array(rec_cos[26544.0].sort_values(ascending=False).head(3).index))


# In[ ]:


data[data["User-ID"]==123054.0][["Age","Location"]].head()


# In[ ]:


data[data["User-ID"]==54885.0][["Age","Location"]].head()


# In[ ]:


data[data["User-ID"]==168144.0][["Age","Location"]].head()


# In[ ]:


def book_recommender():              # userid is the id for which recommendations has to be made, n represents total no. of similiar users wanted 
    print()
    print()
    userid = int(input("Enter the user id to whom you want to recommend : "))
    print()
    print()
    n= int(input("Enter how many books you want to recommend : "))
    print()
    print()
    arr=np.array(rec_cos[userid].sort_values(ascending=False).head(5).index)
    recom_arr=[]

    for i in arr:
        recom_arr.append(data[data["User-ID"]==i][["Book-Title","Book-Rating"]].sort_values(by="Book-Rating",ascending=False))
    
    return(pd.Series(recom_arr[0].append([recom_arr[1],recom_arr[2],recom_arr[3],recom_arr[4]]).groupby("Book-Title")["Book-Rating"].mean().sort_values(ascending=False).index).head(n))


# ### Testing 

# In[ ]:


#book_recommender()


# ## 3. Collaborative Filtering Model-Based Recommender System

# In[ ]:


#data_mih = data_min.copy()
data_mih = data.copy()


# In[ ]:


data_mih.shape


# **Missing values**

# In[ ]:


data_mih.isna().sum()


# It doesn't make sense to keep in our reviews dataset entries where the review is missing.  
# We either add entries for all combinations of (user, book) missing reviews or we remove all entries there review for (user, book) is 0.

# In[ ]:


print('Before: ', data_mih.shape)
data_mih = data_mih.dropna()
print('After dropping null reviews: ', data_mih.shape)


# In[ ]:


print(f"Number of users in our train set: {data_mih['User-ID'].nunique()}")
print(f"Number of books in our train set: {data_mih['ISBN'].nunique()}")
print(f"Train set shape: {data_mih.shape}")


# In[ ]:


#data_mih.groupby('User-ID')['Book-Rating'].count().reset_index().sort_values('Book-Rating', ascending=False).hist()
print('Frequency of number of reviews / user. Most users gave 0-3 reviews')
print(data_mih.groupby('User-ID')['Book-Rating'].count().value_counts()[:10])

fig = plt.figure(figsize=(10,10))
'''
plt.subplot(3, 1, 1)
data_mih.groupby('User-ID')['Book-Rating'].count().value_counts().plot(kind='bar')
plt.title('Top frequencies of ratings per user', fontsize=16)
'''
plt.subplot(3, 1, 2)
data_mih.groupby('User-ID')['Book-Rating'].count().plot(kind='hist', bins=50)
plt.title('Ratings per user', fontsize=16)

plt.subplot(3, 1, 3)
data_mih.groupby('ISBN')['Book-Rating'].count().plot(kind='hist', bins=50)
plt.title('Ratings per book', fontsize=16)
plt.tight_layout()
plt.show()


# Most prolific users

# In[ ]:


#df.groupby('userID')['bookRating'].count().reset_index().sort_values('bookRating', ascending=False)[:10]
print('Most prolific users: Descending order of highest number of reviews given by a user')
data_mih.groupby('User-ID')['Book-Rating'].count().reset_index().sort_values('Book-Rating', ascending=False)[:10]


# Most of the users gave less than 5 ratings.  
# There are few users who gave many ratings.  
# The most productive user gave 7000 ratings.

# **Most popular books**

# In[ ]:


#df.groupby('userID')['bookRating'].count().reset_index().sort_values('bookRating', ascending=False)[:10]
print('Most popular books: Descending order of highest number of reviews received by a book')
data_mih.groupby('ISBN')['Book-Rating'].count().reset_index().sort_values('Book-Rating', ascending=False)[:10]


# **Filter out unpopular books and rarely rating users**
# 
# We want to reduce dimensionality of our dataset (too large & sparse).  
# 
# We will filter out rarely rated books and rarely rating users.

# In[ ]:


min_book_ratings = 50
filter_books = data_mih['ISBN'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

min_user_ratings = 50
filter_users = data_mih['User-ID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

data_mih_new = data_mih[(data_mih['ISBN'].isin(filter_books)) & (data_mih['User-ID'].isin(filter_users))]
print('The original data frame shape:\t{}'.format(data_mih.shape))
print('The new data frame shape:\t{}'.format(data_mih_new.shape))


# In[ ]:


print(f"Number of users in our restricted train set: {data_mih_new['User-ID'].nunique()}")
print(f"Number of books in our restricted train set: {data_mih_new['ISBN'].nunique()}")
print(f"Restricted train set shape: {data_mih_new.shape}")


# ### Surprise package - how to use the library
# 
# "A Python scikit for recommender systems"
# 
# Provides multiple prediction algorithms: 
# - baseline algorithms
# - neighborhood methods
# - matrix factorization-based ( SVD, PMF, SVD++, NMF)
# - and more. 
# 
# Also, various similarity measures (cosine, MSD, pearson…) are built-in.
# 

# In[ ]:


from collections import defaultdict
from surprise import SVD
from surprise import Dataset


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    all_pred = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        all_pred[uid].append((iid, est))
    #print(all_pred)
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in all_pred.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def play_surprise():
    # First train an SVD algorithm on the movielens dataset.
    data_surprise = Dataset.load_builtin('ml-100k')
    trainset = data_surprise.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    top_n = get_top_n(predictions, n=10)

    # Print the first 10 recommended items for each user (in decreasing order of their estimated rating)
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings])
        #print(uid, [est for (_, est) in user_ratings])

#play_surprise()


# ### Preparing dataset for Surprise

# In[ ]:


data_mih = data_mih_new.copy()


# In[ ]:


data_mih.isna().sum()


# In[ ]:


from surprise import Reader, Dataset, SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering

'''
 classmethod load_from_df(df, reader)

    Load a dataset from a pandas dataframe.

    Use this if you want to use a custom dataset that is stored in a pandas dataframe. See the User Guide for an example.
    Parameters:	

        df (Dataframe) – The dataframe containing the ratings. It must have three columns, corresponding to the user (raw) ids, the item (raw) ids, and the ratings, in this order.
        reader (Reader) – A reader to read the file. Only the rating_scale field needs to be specified.

'''
reader = Reader(rating_scale=(0, 10))
data_surprise = Dataset.load_from_df(data_mih[['User-ID', 'ISBN', 'Book-Rating']], reader)

trainset = data_surprise.build_full_trainset() 


# In[ ]:


print(f'Surprise trainset items: {trainset.n_items}')
print(f'Surprise trainset users: {trainset.n_users}')
print(f'Surprise trainset ratings: {trainset.n_ratings}')


# ### Model training

# Read about SVD model parameters here: https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD

# In[ ]:


svd_model = SVD(n_epochs = 20, n_factors = 50)
svd_model.fit(trainset)


# ### Model prediction

# In[ ]:


random_sample


# How to use the predict method in Surprise package: https://surprise.readthedocs.io/en/stable/getting_started.html

# In[ ]:


for i in range(0,10):
#   pred = svd_model.predict(data_mih['User-ID'][i], data_mih['ISBN'][i], r_ui=data_mih['Book-Rating'][i], verbose=True)
    pred = svd_model.predict(data_mih['User-ID'].iloc[i], 
        data_mih['ISBN'].iloc[i], 
        r_ui=data_mih['Book-Rating'].iloc[i], verbose=True)


# In[ ]:


#pred = svd_model.predict(random_sample['User-ID'][0], random_sample['ISBN'][0], r_ui=random_sample['Book-Rating'][0], verbose=True)
pred = svd_model.predict(10447, '0395633206', r_ui=4, verbose=True)


# In[ ]:


pred = svd_model.predict(185233, '0590494457', r_ui=5, verbose=True)


# ## Models comparison

# In[ ]:


#from sklearn.model_selection import cross_validate
from surprise.model_selection import cross_validate

reader = Reader(rating_scale=(0, 10))
data_surprise = Dataset.load_from_df(data_mih[['User-ID', 'ISBN', 'Book-Rating']], reader)

benchmark = []

# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
    print('Currently running ... ', algorithm)

    # Perform cross validation
    results = cross_validate(algorithm, data_surprise, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    print('Results: ', tmp)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse') 


# ## Comparison with content-based and with memory-based collaborative filtering

# In[ ]:


print(data_min.shape)
data_min.isna().sum()


# In[ ]:


data_mih_min = data_min.dropna()
print(data_mih_min.shape)


# ### Train SVD model

# In[ ]:


from surprise import Reader, Dataset, SVD

reader = Reader(rating_scale=(1, 10))
data_surprise = Dataset.load_from_df(data_mih_min[['User-ID', 'ISBN', 'Book-Rating']], reader)

trainset = data_surprise.build_full_trainset() 

svd_model = SVD(n_epochs = 20, n_factors = 50)
svd_model.fit(trainset)


# In[ ]:


random_sample


# In[ ]:


for i in range(0,random_sample.shape[0]):
#   pred = svd_model.predict(data_mih['User-ID'][i], data_mih['ISBN'][i], r_ui=data_mih['Book-Rating'][i], verbose=True)
    pred = svd_model.predict(random_sample['User-ID'].iloc[i], 
        random_sample['ISBN'].iloc[i], 
        r_ui=random_sample['Book-Rating'].iloc[i], verbose=True)


# In[ ]:


random_sample[['User-ID', 'ISBN', 'Book-Rating']]


# In[ ]:


uid = 184339
data_mih_min[data_mih_min['User-ID']==uid]


# In[ ]:


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# In[ ]:


from surprise import Reader, Dataset, SVD

# Then predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = svd_model.test(testset)

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])


# In[ ]:


import pandas as pd
levels = pd.Series([1, 8, 10], index = ['rest', 'Yuliia et al', 'Emilie'])
levels.plot(kind='bar')


# In[ ]:




