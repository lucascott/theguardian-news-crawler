import glob
import json
import multiprocessing
import os
import pickle
import random

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from decleanerboi import DeCleanerBoi

# %%
filepath = './data/'
prefix = 'articles_'
docs_path = filepath + 'docs.pickle'
document_list = glob.glob(filepath + prefix + '*.json')
docs = []
categories = {'film', 'technology', 'travel', 'food', 'business', 'fashion', 'education', 'artanddesign',
              'football', 'games'}

# %%
dcb = DeCleanerBoi()

for name in document_list:
    category = os.path.basename(name).split('_')[1].split('.')[0]
    with open(name, 'r') as fin:
        documents = json.load(fin)
        for _, doc in tqdm(documents.items(), desc=category, unit='articles'):
            words = dcb.SPACYPipeline(' '.join(doc['corpus']))
            docs.append(TaggedDocument(words=words, tags=category))

with open(docs_path, 'wb') as fout:
    pickle.dump(docs, fout)
    print(f"{docs_path} dumped successfully.")

# %%

with open(docs_path, 'rb') as fin:
    docs = pickle.load(fin)
    print(f"{docs_path} loaded successfully.")
ds_train, ds_test = train_test_split(docs, test_size=0.25)
model = Doc2Vec(vector_size=100, window=300, min_count=2, workers=multiprocessing.cpu_count())
model.build_vocab(ds_train)
print('Training time!')
for epoch in tqdm(range(30)):
    model.train(ds_train, total_examples=len(ds_train), epochs=1)
    model.alpha -= 0.002
    model.min_alpha = model.alpha


# %% Learning from matrix representation of each of the documents
def vec_for_learning(model, sents):
    return zip(*[(model.infer_vector(doc.words, epochs=20), doc.tags) for doc in sents])


print('Train/test split...')
X_train, y_train, = vec_for_learning(model, ds_train)
X_test, y_test = vec_for_learning(model, ds_test)
print('Training model...')
logreg = LogisticRegression(n_jobs=multiprocessing.cpu_count(), C=1., solver='lbfgs', multi_class='multinomial')
logreg.fit(X_train, y_train)
print('Predicting topics...')
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))

# %%

tfidf = TfidfVectorizer(analyzer='word')

X_train_tfidf = tfidf.fit_transform(*zip(*[(' '.join(d.words), d.tags) for d in ds_train]))
X_test_tfidf = tfidf.transform([' '.join(d.words) for d in ds_test])
logreg = LogisticRegression(n_jobs=multiprocessing.cpu_count(), C=1., solver='lbfgs', multi_class='multinomial')
logreg.fit(X_train_tfidf, y_train)

print('Predicting topics...')
y_pred = logreg.predict(X_test_tfidf)
print(classification_report([d.tags for d in ds_test], y_pred))

# %%

count = CountVectorizer(analyzer='word')

X_train_count = count.fit_transform(*zip(*[(' '.join(d.words), d.tags) for d in ds_train]))
X_test_count = count.transform([' '.join(d.words) for d in ds_test])
logreg = LogisticRegression(n_jobs=multiprocessing.cpu_count(), C=1., solver='lbfgs', multi_class='multinomial')
logreg.fit(X_train_count, y_train)

print('Predicting topics...')
y_pred = logreg.predict(X_test_count)
print(classification_report([d.tags for d in ds_test], y_pred))

user_profile_size = X_train_tfidf.shape[1]

user_query = X_test_tfidf[0]

cosine_similarities = linear_kernel(user_query, X_train_tfidf).flatten()
related_docs_indices = cosine_similarities.argsort()[:-6:-1]

print(cosine_similarities[related_docs_indices])
print([(ds_train[i].tags, ds_train[i].words[:10]) for i in related_docs_indices.tolist()])

# %%

print('Initializing users...')
users = {}
for c in categories:
    users[c] = np.zeros(user_profile_size)
print('Users initialized.')


def tfidf_predict(query, user, user_perc):
    query_vector = tfidf.transform([query])
    query_vector = user * (user_perc) + query_vector * (1 - user_perc)
    cosine_similarities = linear_kernel(query_vector, X_train_tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]
    # print([(ds_train[i].tags, ds_train[i].words[:10]) for i in related_docs_indices.tolist()])
    return related_docs_indices


def q_to_res(query, user, user_perc):
    indexes = tfidf_predict(query, user, user_perc)
    for ind in indexes:
        print(ind, ds_train[ind].tags.upper(), ' '.join(ds_train[ind].words[:30]))
    return indexes


gamma = 0.6
user_perc = 0.3
for i in range(X_test_tfidf.shape[0] - 10):
    user_query = ' '.join(ds_test[i].words)
    category = ds_test[i].tags
    print("QUERY: ", category.upper(), user_query[:200], '\n')
    user = users[category]
    indexes = q_to_res(user_query, user, user_perc)
    chosen_result = random.choice(indexes)
    disc_res = gamma * (X_train_tfidf[chosen_result] - user)
    user = user + disc_res
    users[category] = user
    # print(np.mean(X_train_tfidf[chosen_result]))
    # print(np.mean(user))
    print('-' * 30)

# %%
user_doc = ds_test[-2]
user_query = ' '.join(user_doc.words)
print("QUERY: ", user_doc.tags.upper(), user_query[:200], '\n')
for c in categories:
    print('User who likes: ', c.upper())
    q_to_res(user_query, users[c], user_perc)
    print('-' * 30)
