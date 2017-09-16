from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from gensim.models import Word2Vec

from pprint import pprint
import numpy as np
import collections

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

google = gensim.models.word2vec.Word2Vec.load_word2vec_format('/home/word2vec-model/GoogleNews-vectors-negative300.bin', binary=True)


with open('data/requirements.txt', 'r') as myfile1:
    data1=myfile1.read().replace('\n', '')

stemmer = PorterStemmer()

stemmed_text1 = [stemmer.stem(i) for i in word_tokenize(data1)]

s1 = ' '.join(stemmed_text1)

print 'Stemmed text1: %s \n\n\n' % s1

lemma = WordNetLemmatizer()

lemma_text = [lemma.lemmatize(i, pos="n") for i in word_tokenize(data1)]

ls1 = ' '. join(lemma_text)

print 'Lemma text1: %s \n\n\n' % ls1

with open("data/ls1.txt", 'w') as f:
  f.write(ls1)

#print 'Text1 %s' % string.join(stemmed_text1, " ")

documents = [data1]

stemmed_docs = [s1]

lemmatized_docs = [ls1]

#vectorizer = TfidfVectorizer(stop_words='english')
#X = vectorizer.fit_transform(lemmatized_docs)

seen = set()
result = []
for item in lemma_text:
    if item not in seen:
        seen.add(item)
        result.append(item)


fails = []
hits = []
vectors = []
for keyword in result:
  try:
    google.vocab[keyword]
    #freebase.vocab['/en/' + keyword]
    vectors.append(google[keyword])
    #vectors.append(freebase['/en/' + keyword])
    hits.append(keyword)
  except:
    try:
      unigrams = keyword.split('_')
      uni_vectors = []
      for uni in unigrams:
        #freebase.vocab['/en/' + uni]
        google.vocab[uni]
        #uni_vectors.append(freebase['/en/' + uni])
        uni_vectors.append(google[uni])
      vectors.append(sum(uni_vectors) / len(unigrams))
      hits.append(keyword)

    except:
      fails.append(keyword)


# Standaridze
new_vectors = StandardScaler().fit_transform(vectors)



# Try out various eps to see distribution of clusters
# Adjust eps bounds and step size
for eps in np.arange(13,14.5,0.1):
  print 'Eps: '
  print eps
  new_db = DBSCAN(eps=eps, min_samples=10).fit(new_vectors)
  labels = new_db.labels_
  counter=collections.Counter(labels)
  print counter
  print len(counter)



# Once you found a decent eps, use it here
new_db = DBSCAN(eps=13.9, min_samples=10).fit(new_vectors)
labels = new_db.labels_
counter=collections.Counter(labels)

clusters = {}
for key, value in counter.iteritems():
  clusters[key] = []


for idx, val in enumerate(labels):
  clusters[val].append(hits[idx])

pprint(clusters)