#%%

# Goal: Develop a sentiment classifier that determines the sentiment of documents as positive or negative.

import pandas as pd

# [10000 rows x 10 columns]
df = pd.read_csv('/Users/jeewonkim/Desktop/yelp.csv')
# print(df)

df = df.drop(['business_id', 'date', 'review_id', 'type', 'user_id', 'cool', 'useful', 'funny'], axis = 1)

print(df.head())

# print(df['text'][0])

print(df.stars.value_counts())


df.stars.plot.hist()

# delete reviews having stars=3
df = df[df['stars']!=3]
# adding new column, setiment
df['sentiment'] = df['stars'].apply(lambda x:1 if x>3 else 0)

print(df.head())
print(df.sentiment.value_counts())


from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ' '.join(df.text.values)
# wc = WordCloud(width=2000, height=1000, max_words=500).generate(text)
# plt.imshow(wc)
# plt.axis("off")
# plt.show()


# Document vectorization
# Two options:
# - `CountVectorizer`: return vectors having **term frequencies** (docuemnt [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html))
# - `TfidfVectorizer`: return vectors having **Tf-Idf (term frequency-inverse docuement frequency) values** (document [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

count_vect = CountVectorizer(max_features = 1000,
                             ngram_range = (1,2),
                             lowercase = True,
                             stop_words = 'english')
count_vect.fit(df['text'])
X_counts = count_vect.transform(df['text'])

tfidf_vect = TfidfVectorizer(max_features = 1000, # select top 1000 frequent tokens
                            ngram_range = (1,2), # use unigram and bigram
                            lowercase = True, # convert all characters to lowercase before tokenizing
                            stop_words = 'english' # remove predefined stop words for English
                            )
tfidf_vect.fit(df['text'])
X_tfidfs = tfidf_vect.transform(df['text'])

print(X_counts.shape, X_tfidfs.shape)

df_bow_tf = pd.DataFrame(X_counts.todense(), columns = count_vect.get_feature_names_out())
print(df_bow_tf.head())

df_bow_tfidf = pd.DataFrame(X_tfidfs.todense(), columns=tfidf_vect.get_feature_names_out())
print(df_bow_tfidf.head())


# Develop sentiment classifiers

X = df_bow_tfidf
y = df.sentiment

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# data split
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
X_trn.shape, X_tst.shape, y_trn.shape, y_tst.shape

# model training: Lasso (l1 penalty)
clf = LogisticRegression(penalty='l1', solver='saga', C=10, random_state=0)
clf.fit(X_trn, y_trn)

# model evaluation
y_trn_pred = clf.predict(X_trn)
acc = accuracy_score(y_trn, y_trn_pred)
f1 = f1_score(y_trn, y_trn_pred)
print('Training Accuracy: {:.4f}, Fscore: {:.4f}'.format(acc, f1))
y_tst_pred = clf.predict(X_tst)
acc = accuracy_score(y_tst, y_tst_pred)

# f1 score 쓰는 이유 : class imbalance 문제때문. accuracy is not enough for this problem
f1 = f1_score(y_tst, y_tst_pred)
print('Test Accuracy: {:.4f}, Fscore: {:.4f}'.format(acc, f1))


import numpy as np
coefficient_values = clf.coef_.squeeze()
feature_names = X.columns

args = np.argsort(coefficient_values)

print('top 20 negative words:')
for token, coef in zip(feature_names[args[:20]], coefficient_values[args[:20]]):
    print('{:<20}: {:.4f}'.format(token, coef))
    
print('top 20 positive words:')
for token, coef in zip(feature_names[args[-20:][::-1]], coefficient_values[args[-20:]][::-1]): # [::-1] for reversing the order
    print('{:>20}: {:.4f}'.format(token, coef))
    

# Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# data split
X_trn, X_tst, y_trn, y_tst = train_test_split(df['text'], df['sentiment'], test_size=0.3, stratify=y, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipe = Pipeline([("preprocessing", None), ("classifier", None)])

param_grid = [
    {'preprocessing': [CountVectorizer(max_features=1000, stop_words='english'), TfidfVectorizer(max_features=1000, stop_words='english')],
     'preprocessing__ngram_range': [(1, 1), (1, 2)],     
     'classifier': [LogisticRegression(penalty='l1', solver='saga')], 
     'classifier__C': [0.1, 1, 10]},
    {'preprocessing': [CountVectorizer(max_features=1000, stop_words='english'), TfidfVectorizer(max_features=1000, stop_words='english')],
     'preprocessing__ngram_range': [(1, 1), (1, 2)],
     'classifier': [RandomForestClassifier()]}]

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1')
grid.fit(X_trn, y_trn)

print("Best hyperparams: {}".format(grid.best_params_))
print("Best cross-validation score: {}".format(grid.best_score_))
print("Test-set score: {}".format(grid.score(X_tst, y_tst)))

import pandas as pd
results = pd.DataFrame(grid.cv_results_)

chosen_extractor = grid.best_estimator_.named_steps['preprocessing']
chosen_classifier = grid.best_estimator_.named_steps['classifier']

# coefficient values and their corresponding feature names
coefficient_values = chosen_classifier.coef_.squeeze()
feature_names = chosen_extractor.get_feature_names_out()

# get index to access the values in ascending order
args = np.argsort(coefficient_values)

print('top 10 negative words:')
for token, coef in zip(feature_names[args[:20]], coefficient_values[args[:20]]):
    print('{:<20}: {:.4f}'.format(token, coef))
    
print('top 10 positive words:')
for token, coef in zip(feature_names[args[-20:][::-1]], coefficient_values[args[-20:]][::-1]): # [::-1] for reversing the order
    print('{:>20}: {:.4f}'.format(token, coef))