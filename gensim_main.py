from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from difflib import SequenceMatcher as SM
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities


documents = [
    "The responsiveness of our app is ensured by an ElasticSearch cluster.",
    "The app is made of a Vue.js frontend and a Node.js backend, both written in Typescript.",
    "The almost real-time data processing pipelines hold components written in Rust and Golang.",
    "Our stack is mostly in Node both on the backend and frontend, and we work with React for our interfaces and GraphQL as API.",
    "Our Mobile Apps are made with Swift and Kotlin."
]

stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
index = similarities.MatrixSimilarity(lsi[corpus])
index.save('./similarity_01.index')
index = similarities.MatrixSimilarity.load('./similarity_01.index')

doc = [
    "Application user interface is made of a React.js frontend and a Node.js backend written with Typescript.",
    "I have a car.",
    "The responsiveness of search features is implemented with Elasticsearch cluster"
]
items = {}


def test_text(text):
    vec_bow = dictionary.doc2bow(text.lower().split())
    vec_lsi = lsi[vec_bow]
    # index = similarities.MatrixSimilarity(lsi[corpus])
    # index.save('./similarity_01.index')
    # index = similarities.MatrixSimilarity.load('./similarity_01.index')
    # global index
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for doc_position, doc_score in sims:
        if documents[doc_position] in items and items[documents[doc_position]]["score"] > 70:
            pass
        else:
            items[documents[doc_position]] = {"score": doc_score * 100}
            items[documents[doc_position]
                  ]['score'] += fuzz.ratio(text, documents[doc_position])
            items[documents[doc_position]
                  ]['score'] += fuzz.partial_ratio(text, documents[doc_position])
            items[documents[doc_position]
                  ]['score'] += SM(None, text, documents[doc_position]).ratio() * 100
            items[documents[doc_position]
                  ]['score'] = round(items[documents[doc_position]]['score'] / 4)
    for i in documents:
        if items[i]["score"] > 70:
            items[i]["status"] = "passed"
        else:
            items[i]["status"] = "failed"


for text in doc:
    test_text(text)

print(items)
