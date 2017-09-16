from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from gensim.models import Word2Vec



with open('data/requirements.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '')

stemmer = PorterStemmer()

stemmed_text = [stemmer.stem(i) for i in word_tokenize(data)]

s = ' '.join(stemmed_text)

#print 'Stemmed text: %s \n\n\n' % s

lemma = WordNetLemmatizer()
lemma_text = [lemma.lemmatize(i, pos="n") for i in word_tokenize(data)]

ls = ' '. join(lemma_text)

#print 'Lemma text1: %s \n\n\n' % ls

with open("data/ls.txt", 'w') as f:
  f.write(ls)

#documents = [data]
#stemmed_docs = [s]
#lemmatized_docs = [ls]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(lemma_text)
true_k = 4
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :15]:
        print ' %s' % terms[ind],
    print
