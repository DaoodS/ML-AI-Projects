import time
import pickle
import numpy as np
import pandas as pd

## nltk libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify, make_response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Load saved modelss and data
data = pd.read_csv('data/sample30.csv', index_col=None)

final_models_file = "models/user_final_rating.pkl"
with open(final_models_file, 'rb') as file:
	final_models = pickle.load(file)
	
user_encoding = "models/user_encoding.pkl"
with open(user_encoding, 'rb') as file:
	le_usr = pickle.load(file)

movie_name_encoding = "models/movie_name_encoding.pkl"
with open(movie_name_encoding, 'rb') as file:
	le_name = pickle.load(file)

tfidf_vect = "models/tfidf_vect.pkl"
with open(tfidf_vect, 'rb') as file:
	tf_vect = pickle.load(file)

tf_models_file = "models/tfidf_model.pkl"
with open(tf_models_file, 'rb') as file:
	tf_models = pickle.load(file)
	
xgtf_models_file = "models/xgb_tf_model.pkl"
with open(xgtf_models_file, 'rb') as file:
	xgtf_models = pickle.load(file)


def process_review_pipeline(df):
				
	stemmer = PorterStemmer()
	def preprocess(document):
		'changes document to lower case and removes stopwords'
		document = document.lower()
		words = word_tokenize(document)
		words = [word for word in words if word not in stopwords.words("english")]
		words = [stemmer.stem(word) for word in words]
		document = " ".join(words)
		return document
	
	df['proc_reviews_text'] = df['reviews_text'].apply(preprocess)
	return df['proc_reviews_text']


@app.route('/', methods=['GET'])
def index():
	# Main page
	return render_template('index.html')


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
	if request.method=='POST':
		try:
			# Get the user id
			user_input = int(request.form.get('userid', None))

			# Return Recommended movies
			d = final_models.loc[user_input].sort_values(ascending=False)[0:20].to_frame()
			d = d.reset_index()
			d.drop(d.columns[1], axis=1, inplace=True)

			movie_name = le_name.inverse_transform(d['movieId'])
			d['movie_name'] = movie_name

			filter_df = data[data['name'].isin(d['movie_name'].to_list())]
			filter_df = filter_df[~filter_df['reviews_text'].isnull()]
			filter_df = filter_df[['name', 'reviews_title', 'reviews_text']]

			filter_df['proc_reviews_text'] = process_review_pipeline(filter_df)
			tf = tf_models.transform(filter_df['proc_reviews_text'])
			tf_df = pd.DataFrame(tf.toarray(), columns=tf_vect.get_feature_names())

			preds = xgtf_models.predict_proba(tf_df)
			best_preds = np.asarray([np.argmax(line) for line in preds])
			filter_df['sentiment'] = best_preds

			final_df = filter_df[filter_df['sentiment']==1]
			final_df = final_df.groupby('name')['sentiment'].sum().reset_index()
			final_df = final_df.sort_values(by='sentiment', ascending=False)
			rec = final_df[:5]['name'].to_list()
			print(rec)
			return make_response(jsonify({'Movies': rec}), 200)

		except Exception as e:
			print(e)
			return render_template('index.html')

	elif request.method == 'GET':
		return render_template('index.html')
	return None


if __name__ == '__main__':
	print('*** App Started ***')
	app.run(debug=True)