import time
import pickle
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

data = pd.read_csv('Model/sample30.csv', index_col=None)

final_model_file = "Model/user_final_rating.pkl"
with open(final_model_file, 'rb') as file:
    final_model = pickle.load(file)
    
user_encoding = "Model/user_encoding.pkl"
with open(user_encoding, 'rb') as file:
    le_usr = pickle.load(file)

movie_name_encoding = "Model/movie_name_encoding.pkl"
with open(movie_name_encoding, 'rb') as file:
    le_name = pickle.load(file)

tfidf_vect = "Model/tfidf_vect.pkl"
with open(tfidf_vect, 'rb') as file:
    tf_vect = pickle.load(file)

tf_model_file = "Model/tfidf_model.pkl"
with open(tf_model_file, 'rb') as file:
    tf_model = pickle.load(file)
    
xgtf_model_file = "Model/xgb_tf_model.pkl"
with open(xgtf_model_file, 'rb') as file:
    xgtf_model = pickle.load(file)



class Recommendation:
	
	def __init__(self):
		pass

	def process_review_pipeline(self, df):
	            
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

	def recommend_products(self, user_id):
		print("User name : {}".format(le_usr.inverse_transform([user_id])[0]))
		print("Fetching Recommendations....")

		d = final_model.loc[user_input].sort_values(ascending=False)[0:20].to_frame()
		d = d.reset_index()
		d.drop(d.columns[1], axis=1, inplace=True)

		movie_name = le_name.inverse_transform(d['movieId'])
		d['movie_name'] = movie_name

		filter_df = data[data['name'].isin(d['movie_name'].to_list())]
		filter_df = filter_df[~filter_df['reviews_text'].isnull()]
		filter_df = filter_df[['name', 'reviews_title', 'reviews_text']]

		filter_df['proc_reviews_text'] = self.process_review_pipeline(filter_df)
		tf = tf_model.transform(filter_df['proc_reviews_text'])
		tf_df = pd.DataFrame(tf.toarray(), columns=tf_vect.get_feature_names())

		preds = xgtf_model.predict_proba(tf_df)
		best_preds = np.asarray([np.argmax(line) for line in preds])
		filter_df['sentiment'] = best_preds

		final_df = filter_df[filter_df['sentiment']==1]
		final_df = final_df.groupby('name')['sentiment'].sum().reset_index()
		final_df = final_df.sort_values(by='sentiment', ascending=False)
		return (final_df[:5]['name'].to_list())



if __name__=='__main__':
	user_input = int(input("Enter your user name: "))
	recom_obj = Recommendation()
	print(recom_obj.recommend_products(user_input))