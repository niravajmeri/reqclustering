import string 

import gensim
from gensim.models import Word2Vec
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

with open('data/requirements.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '')

data = data.lower()

lemma = WordNetLemmatizer()

# Remove stopwords
stops = set(stopwords.words("english"))
#lemma_filtered = [word for word in lemma_text if word not in stops]

model = gensim.models.KeyedVectors.load_word2vec_format('~/word2vec-model/GoogleNews-vectors-negative300.bin', binary=True) 

lemma_filtered = []
vectors = []

for word in word_tokenize(data):
  word = word.lower()
  word = word.strip()
  word = word.strip('_')
  word = word.strip('*')
  
  if word in stops:
    continue
  if all(char in string.punctuation for char in word):
    continue
 
  try:  
    lemma_word = lemma.lemmatize(word, pos='n')
    lemma_filtered.append(lemma_word)
    vectors.append(model[lemma_word])
  except KeyError:
    print word, " not in vocabulary"

print lemma_filtered

print vectors
