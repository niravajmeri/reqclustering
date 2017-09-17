import gensim
from gensim.models import Word2Vec
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize

with open('data/requirements.txt', 'r') as myfile1:
    data=myfile1.read().replace('\n', '')

data = data.lower

lemma = WordNetLemmatizer()

lemma_text = [lemma.lemmatize(i, pos="n") for i in word_tokenize(data)]

# Remove stopwords
stops = set(stopwords.words("english"))
lemma_filtered = [word for word in lemma_text if word not in stops]

#model = gensim.models.Word2Vec.load_word2vec_format('~/word2vec-model/GoogleNews-vectors-negative300.bin', binary=True) 

model = gensim.models.KeyedVectors.load_word2vec_format('~/word2vec-model/GoogleNews-vectors-negative300.bin', binary=True) 

#print lemma_text1[1]
#print model[lemma_text1[1]]

vectorfile = [model[lemma.lemmatize(i, pos="n")] for i in word_tokenize(data)]

print vectorfile
