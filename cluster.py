from nltk.stem.porter import *
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score



with open('data/group-36.csv', 'r') as myfile1:
    data1=myfile1.read().replace('\n', '')

with open('data/group-40.csv', 'r') as myfile2:
    data2=myfile2.read().replace('\n', '')

with open('data/group-52.csv', 'r') as myfile3:
    data3=myfile3.read().replace('\n', '')

with open('data/group-54.csv', 'r') as myfile4:
    data4=myfile4.read().replace('\n', '')

with open('data/group-72.csv', 'r') as myfile5:
    data5=myfile5.read().replace('\n', '')

with open('data/group-76.csv', 'r') as myfile6:
    data6=myfile6.read().replace('\n', '')

#documents = ["Human machine interface for lab abc computer applications",
#             "A survey of user opinion of computer system response time",
#             "The EPS user interface management system",
#             "System and human system engineering testing of EPS",
#             "Relation of user perceived response time to error measurement",
#             "The generation of random binary unordered trees",
#             "The intersection graph of paths in trees",
#             "Graph minors IV Widths of trees and well quasi ordering",
#             "Graph minors A survey"]


stemmer = PorterStemmer()

stemmed_text1 = [stemmer.stem(i) for i in word_tokenize(data1)]

stemmed_text2 = [stemmer.stem(i) for i in word_tokenize(data2)]

stemmed_text3 = [stemmer.stem(i) for i in word_tokenize(data3)]

stemmed_text4 = [stemmer.stem(i) for i in word_tokenize(data4)]

stemmed_text5 = [stemmer.stem(i) for i in word_tokenize(data5)]

stemmed_text6 = [stemmer.stem(i) for i in word_tokenize(data6)]

s1 = ' '.join(stemmed_text1)
s2 = ' '.join(stemmed_text2)
s3 = ' '.join(stemmed_text3)
s4 = ' '.join(stemmed_text4)
s5 = ' '.join(stemmed_text5)
s6 = ' '.join(stemmed_text6)

print 'Text1 %s' % s1

#print 'Text1 %s' % string.join(stemmed_text1, " ")
#print 'Text2 %s' % stemmed_text2
#print 'Text3 %s' % stemmed_text3
#print 'Text4 %s' % stemmed_text4
#print 'Text5 %s' % stemmed_text5
#print 'Text6 %s' % stemmed_text6


documents = [data1,data2,data3,data4,data5,data6]

stemmed_docs = [s1,s2,s3,s4,s5,s6]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(stemmed_docs)
true_k = 5
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
