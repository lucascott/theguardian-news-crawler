import glob
import json
import multiprocessing
import os
import pickle
from collections import namedtuple

from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from decleanerboi import DeCleanerBoi

filepath = './data/'
prefix = 'articles_'
docs_path = filepath + 'docs.pickle'
document_list = glob.glob(filepath + prefix + '*.json')
CleanedDocument = namedtuple('CleanedDocument', 'words tags')
docs = []
# %%
dcb = DeCleanerBoi()

for name in document_list:
    category = os.path.basename(name).split('_')[1].split('.')[0]
    with open(name, 'r') as fin:

        documents = json.load(fin)
        for _, doc in tqdm(documents.items(), desc=category, unit='articles'):
            words = dcb.clean_article(' '.join(doc['corpus']))
            docs.append(CleanedDocument(words, category))

with open(docs_path, 'wb') as fout:
    pickle.dump(docs, fout)
    print(f"{docs_path} dumped successfully.")

# %%

with open(docs_path, 'rb') as fin:
    docs = pickle.load(fin)
    print(f"{docs_path} loaded successfully.")
ds_train, ds_test = train_test_split(docs, test_size=0.25)
model_dbow = Doc2Vec(vector_size=100, window=300, min_count=2, workers=multiprocessing.cpu_count())
model_dbow.build_vocab(ds_train)
for epoch in tqdm(range(30)):
    model_dbow.train(ds_train, total_examples=len(ds_train), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha


def vec_for_learning(model, sents):
    return zip(*[(model.infer_vector(doc.words, epochs=20), doc.tags) for doc in sents])


print('Train/test split...')
X_train, y_train, = vec_for_learning(model_dbow, ds_train)
X_test, y_test = vec_for_learning(model_dbow, ds_test)
print('Training model...')
logreg = LogisticRegression(n_jobs=multiprocessing.cpu_count(), C=1., solver='lbfgs', multi_class='multinomial')
logreg.fit(X_train, y_train)
print('Predicting topics...')
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))
