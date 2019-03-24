import string
import gensim
import sys
import numpy as np
import collections

from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from gensim.models import Word2Vec

from pprint import pprint

google = gensim.models.KeyedVectors.load_word2vec_format('~/word2vec-model/GoogleNews-vectors-negative300.bin', binary=True)


with open('data/requirements.txt', 'r') as myfile:
  data=myfile.read().replace('\n', ' ')

#stemmer = PorterStemmer()
#stemmed_text1 = [stemmer.stem(i) for i in word_tokenize(data1)]
#s1 = ' '.join(stemmed_text1)
#print 'Stemmed text1: %s \n\n\n' % s1

lemma = WordNetLemmatizer()
#lemma_text = [lemma.lemmatize(i, pos="n") for i in word_tokenize(data1)]

# Remove stopwords
stops = set(stopwords.words("english"))
#lemma_filtered = [word for word in lemma_text if word not in stops]

#ls1 = ' '. join(lemma_text)

#print 'Lemma text1: %s \n\n\n' % ls1

#with open("data/ls1.txt", 'w') as f:
#  f.write(ls1)

#print 'Text1 %s' % string.join(stemmed_text1, " ")

#documents = [data1]
#stemmed_docs = [s1]
#lemmatized_docs = [ls1]

#vectorizer = TfidfVectorizer(stop_words='english')
#X = vectorizer.fit_transform(lemmatized_docs)

seen = set()
result = []
for word in word_tokenize(data):
  word = word.lower()
  word = word.strip()
  word = word.strip('_')
  word = word.strip('*')
  
  if word in stops:
    continue
  if all(char in string.punctuation for char in word):
    continue
  if len(word) < 3:
    continue
  try:
    #stem_word = stemmer.stem(word)
    lemma_word = lemma.lemmatize(word, pos='n')
    if lemma_word not in seen:
      seen.add(lemma_word)
      result.append(lemma_word)
  except KeyError:
    print lemma_word, " not in vocabulary"

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


# Standardize
new_vectors = StandardScaler().fit_transform(vectors)
#new_vectors = TfidfVectorizer(stop_words='english').fit_transform(vectors)


# Try out various eps to see distribution of clusters
# Adjust eps bounds and step size

'''
for eps in np.arange(16.5,17.5,0.1):
  print 'Eps: '
  print eps
  for min in np.arange(2,4,1):
    print 'Min samples: '
    print min
    new_db = DBSCAN(eps=eps, min_samples=min).fit(new_vectors)
    labels = new_db.labels_
    counter=collections.Counter(labels)
    print counter
    print len(counter)
  
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
'''


# Once you found a decent eps, use it here
new_db = DBSCAN(eps=16.5, min_samples=2).fit(new_vectors)
labels = new_db.labels_
counter=collections.Counter(labels)

clusters = {}
for key, value in counter.iteritems():
  clusters[key] = []


for idx, val in enumerate(labels):
  clusters[val].append(hits[idx])

#pprint(clusters)

#f = open('out.txt','w')
#pprint >>f, clusters

with open('out-cluster-dbscan-eps16.5-min2.txt', 'wt') as out: 
  pprint(clusters, stream=out)
