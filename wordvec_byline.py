import string 

import gensim
from gensim.models import Word2Vec
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

from nltk.stem.porter import *

#with open('data/requirements.txt', 'r') as myfile:
#    data=myfile.read()

lines = open('data/requirements.txt', 'r').read().splitlines()

stemmer = PorterStemmer()
lemma = WordNetLemmatizer()

# Remove stopwords
stops = set(stopwords.words("english"))
#lemma_filtered = [word for word in lemma_text if word not in stops]

model = gensim.models.KeyedVectors.load_word2vec_format('~/word2vec-model/GoogleNews-vectors-negative300.bin', binary=True) 

lemma_filtered = []
stem_filtered = []
vectors = []
seen = set()

outfile = open("wordvec_stem+lemma_byline.csv", "a")
vectorfile = open("wordvec_vectors.csv","a")
for line in lines:
  #dataline = line.strip('\n')
  print line
  lemma_line = []
  stem_line = []
  for word in word_tokenize(line):
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
      stem_word = stemmer.stem(word)
      stem_line.append(stem_word)
      stem_filtered.append(stem_word)

      lemma_word = lemma.lemmatize(word, pos='n')
      lemma_line.append(lemma_word)
      lemma_filtered.append(lemma_word)
      if lemma_word  not in seen:
        seen.add(lemma_word)
        vectors.append(model[lemma_word])
        vectortext = ""
        #vectortext = lemma_word + "," + model[lemma_word] + "\n"
        vectorfile.write(lemma_word)
        vectorfile.write(",")
        print model[lemma_word]
        vector_string = ["%.6f" % x for x in model[lemma_word]]
        print vector_string
        vectorfile.write(",".join(vector_string))
        vectorfile.write("\n")
    except KeyError:
      print word, " not in vocabulary"
  print " ".join(stem_line)
  print " ".join(lemma_line)
  outtext = ""
  #outtext = "\"",line,"\",\""," ".join(stem_line),"\",\""," ".join(lemma_line),"\"\n"
  outfile.write(line)
  outfile.write("||")
  outfile.write(" ".join(stem_line))
  outfile.write("||")
  outfile.write(" ".join(lemma_line))
  outfile.write("\n")

outfile.close()
vectorfile.close()
#print lemma_filtered

#print vectors
