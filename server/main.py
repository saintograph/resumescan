from gensim import corpora, models, similarities
from collections import defaultdict
from difflib import SequenceMatcher as SM
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from flask import Flask, jsonify, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, TextAreaField
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from gensim_main import test_text

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'foo'

# def process_entry(resume, ad):
#   punctuation = string.punctuation + "'"
#   def remove_punctuation(text):
#       res = re.sub(r'[^\w\s]', '', text)
#       return res
#   def tokenization(text):
#       # res = nltk.sent_tokenize(text)
#       res = nltk.word_tokenize(text)
#       return res
#   def remove_stopwords(text):
#       return [word for word in text if not word in stopwords.words()]
#   def lemmatizer_func(text):
#       lemmatizer = WordNetLemmatizer()
#       res = [lemmatizer.lemmatize(word) for word in text]
#       return res
#   def process_text(text):
#     text = remove_punctuation(text)
#     text = text.replace("\n", " ")
#     text = tokenization(text)
#     text = lemmatizer_func(text)
#     text = remove_stopwords(text)
#     return text
#   text = [" ".join(process_text(resume)), "".join(process_text(ad))]
#   cv = CountVectorizer()
#   count_matrix = cv.fit_transform(text)
#   match = round(cosine_similarity(count_matrix)[0][1] * 100)
#   return match

# from gensim import models
# from gensim import similarities


documents_raw = [
    "The responsiveness of our app is ensured by an ElasticSearch cluster.",
    "The app is made of a Vue.js frontend and a Node.js backend, both written in Typescript.",
    "The almost real-time data processing pipelines hold components written in Rust and Golang.",
    "Our stack is mostly in Node both on the backend and frontend, and we work with React for our interfaces and GraphQL as API.",
    "Our Mobile Apps are made with Swift and Kotlin."
]


def test_text(docs, documents):
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

    items = {}
    for text in docs:
        vec_bow = dictionary.doc2bow(text.lower().split())
        vec_lsi = lsi[vec_bow]
        # index = similarities.MatrixSimilarity(lsi[corpus])
        # index.save('./similarity_01.index')
        # index = similarities.MatrixSimilarity.load('./similarity_01.index')
        # global index
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for doc_position, doc_score in sims:
            if documents[doc_position] in items and items[documents[doc_position]]["score"] > 50:
                pass
            else:
                items[documents[doc_position]] = {"score": doc_score * 100}
                items[documents[doc_position]
                      ]['score'] += fuzz.ratio(text, documents[doc_position])
                items[documents[doc_position]
                      ]['score'] += fuzz.partial_ratio(text, documents[doc_position])
                items[documents[doc_position]
                      ]['score'] += SM(None, text, documents[doc_position]).ratio() * 100
                print(items[documents[doc_position]]['score'])
                items[documents[doc_position]
                      ]['score'] = round(items[documents[doc_position]]['score'] / 4)
        for i in documents:
            if items[i]["score"] > 50:
                items[i]["status"] = "passed"
            else:
                items[i]["status"] = "failed"
        # print(items)
    return items


# for text in doc:
#     test_text(text)

# print(items)

class JobForm(FlaskForm):
    resume = TextAreaField('Resume')
    ad = TextAreaField('Ad Description')


@app.route("/", methods=["GET", "POST"])
def index():
    form = JobForm()
    result = None
    overall = 0
    if form.is_submitted():
        # print(list(filter(None, [x.replace("\r\n","") for x in form.resume.data.split(".")])))
        print(list(filter(None, [x.replace("\r\n", "") for x in form.ad.data.split(
            ".")])))
        try:
            # print(form.resume.data.split("."))
            # result = process_entry(form.resume.data, form.ad.data)
            # print(list(filter(None, form.resume.data.split("."))))
            # result = test_text(list(filter(None, form.resume.data.split("."))), list(filter(None, form.ad.data.split("."))))
            result = test_text(list(filter(None, [x.replace("\r\n", "") for x in form.resume.data.split(
                ".")])), list(filter(None, [x.replace("\r\n", "") for x in form.ad.data.split(".")])))
            # print("submitted", result)
            for i in list(result):
                score = result[i]['score']
                overall += score
            overall = round(overall / len(list(result)))
            return render_template("index.html", form=form, result=result, overall=overall)
        except Exception as e:
            print(e)
            return render_template("index.html", form=form, result=result)
    return render_template("index.html", form=form)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

"""
Application user interface is made of a React.js frontend and a Node.js backend written with Typescript. I have a car. The responsiveness of search features is implemented with Elasticsearch cluster.
"""
