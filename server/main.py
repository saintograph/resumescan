from flask import Flask, jsonify, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, TextAreaField
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['SECRET_KEY'] = 'foo'

def process_entry(resume, ad):
  punctuation = string.punctuation + "'"
  def remove_punctuation(text):
      res = re.sub(r'[^\w\s]', '', text)
      return res
  def tokenization(text):
      res = nltk.sent_tokenize(text)
      return res
  def remove_stopwords(text):
      return [word for word in text if not word in stopwords.words()]
  def lemmatizer(text):
      res = [wordnet_lemmatizer.lemmatize(word) for word in text]
      return res
  def process_text(text):
    text = remove_punctuation(text)
    text = text.replace("\n", " ")
    text = tokenization(text)
    text = remove_stopwords(text)
    return text
  text = [" ".join(process_text(resume)), "".join(process_text(ad))]
  cv = CountVectorizer()
  count_matrix = cv.fit_transform(text)
  match = round(cosine_similarity(count_matrix)[0][1] * 100)
  return match

class JobForm(FlaskForm):
  resume = TextAreaField('Course Description')
  ad = TextAreaField('Ad Description')

@app.route("/", methods = ["GET", "POST"])
def index():
  form = JobForm()
  result = None
  if form.is_submitted():
    try:
      result = process_entry(form.resume.data, form.ad.data)
      print(result)
      return render_template("index.html", form = form, result = result)
    except:
      return render_template("index.html", form = form, result = result)
  return render_template("index.html", form = form)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')