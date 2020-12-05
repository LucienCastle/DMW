import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

import lightgbm as lgb
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer

import os
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import gc
import re
import string

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from textblob import TextBlob
from tqdm import tqdm

from wordcloud import WordCloud, STOPWORDS


def files_input(dataset):
	df_train = pd.read_csv(dataset[0])
	df_test = pd.read_csv(dataset[1])
	df_train = df_train.dropna(axis=0)
	df_test = df_test.dropna(axis=0)
	return df_train,df_test

def visualization(df_train,df_test):
	conditions = df_train.condition.value_counts().sort_values(ascending=False)
	plt.rcParams['figure.figsize'] = [12, 8]
	conditions[:10].plot(kind='bar')
	plt.title('Top 10 Most Common Conditions')
	plt.xlabel('Condition')
	plt.ylabel('Count');
	plt.show()

	plt.figure(figsize=(9,9))
	sns.distplot(df_train['rating'],kde=False)
	sns.distplot(df_test['rating'],color='violet',kde=False)
	plt.xlabel('Rating')
	plt.ylabel('Dist')
	plt.title("Distribution of rating (train and test)")
	plt.show()

	plt.figure(figsize=(9,9))
	sns.distplot(df_train['usefulCount'],kde=False)
	sns.distplot(df_test['usefulCount'],color='violet',kde=False)
	plt.xlabel('Useful Count')
	plt.ylabel('Dist')
	plt.title("Distribution of Useful Count (train and test)")
	plt.show()

	plt.scatter(df_train.rating, df_train.usefulCount, c=df_train.rating.values, cmap='tab10')
	plt.title('Useful Count vs Rating')
	plt.xlabel('Rating')
	plt.ylabel('Useful Count')
	plt.show()

def preprocessing(df_train,df_test):
	df_all = pd.concat([df_train,df_test]).reset_index()
	del df_all['index']
	percent = (df_all.isnull().sum()).sort_values(ascending=False)
	percent.plot(kind="bar", figsize = (14,6), fontsize = 10, color='green')
	plt.xlabel("Columns", fontsize = 20)
	plt.ylabel("", fontsize = 20)
	plt.title("Total Missing Value ", fontsize = 20)
	plt.show()

	df_all[df_all['condition']=='3</span> users found this comment helpful.'].head(3)
	all_list = set(df_all.index)
	span_list = []
	for i,j in enumerate(df_all['condition']):
	    if '</span>' in j:
	        span_list.append(i)
	
	new_idx = all_list.difference(set(span_list))
	df_all = df_all.iloc[list(new_idx)].reset_index()
	del df_all['index']

	df_condition = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
	df_condition = pd.DataFrame(df_condition).reset_index()
	df_condition.tail(20)
	df_condition_1 = df_condition[df_condition['drugName']==1].reset_index()
	df_condition_1['condition'][0:10]	

	all_list = set(df_all.index)
	condition_list = []
	for i,j in enumerate(df_all['condition']):
	    for c in list(df_condition_1['condition']):
	        if j == c:
	            condition_list.append(i)
	            
	new_idx = all_list.difference(set(condition_list))
	df_all = df_all.iloc[list(new_idx)].reset_index()
	del df_all['index']
	return df_all

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',stopwords = stopwords,max_words = max_words,max_font_size = max_font_size,random_state = 42,width=800,height=400,mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size, 'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  


def review_to_words(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    stops = set(stopwords.words('english'))
    not_stop = ["aren't","couldn't","didn't","doesn't","don't","hadn't","hasn't","haven't","isn't","mightn't","mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]
    for i in not_stop:
	    stops.remove(i)

    meaningful_words = [w for w in words if not w in stops]
    # 6. Stemming
    stemmer = SnowballStemmer('english')
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(stemming_words))

def sentiment_analysis(df_all):
	stops = set(stopwords.words('english'))
	
	plot_wordcloud(stops, title="Word Cloud of stops")
	
	not_stop = ["aren't","couldn't","didn't","doesn't","don't","hadn't","hasn't","haven't","isn't","mightn't","mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]
	for i in not_stop:
		stops.remove(i)

	df_all['review_clean'] = df_all['review'].apply(review_to_words)
	df_all['sentiment'] = df_all["rating"].apply(lambda x: 1 if x > 5 else 0)

	df_train, df_test = train_test_split(df_all, test_size=0.33, random_state=42) 
	len_train = df_train.shape[0]
	df_all = pd.concat([df_train,df_test])
	del df_train, df_test;
	gc.collect()

	df_all['date'] = pd.to_datetime(df_all['date'])
	df_all['day'] = df_all['date'].dt.day
	df_all['year'] = df_all['date'].dt.year
	df_all['month'] = df_all['date'].dt.month

	reviews = df_all['review_clean']
	Predict_Sentiment = []
	for review in tqdm(reviews):
	    blob = TextBlob(review)
	    Predict_Sentiment += [blob.sentiment.polarity]
	df_all["Predict_Sentiment"] = Predict_Sentiment
	df_all.head()

	reviews = df_all['review']

	Predict_Sentiment = []
	for review in tqdm(reviews):
	    blob = TextBlob(review)
	    Predict_Sentiment += [blob.sentiment.polarity]
	df_all["Predict_Sentiment2"] = Predict_Sentiment
	df_all.head()

	df_all['count_sent']=df_all["review"].apply(lambda x: len(re.findall("\n",str(x)))+1)
	df_all['count_word']=df_all["review_clean"].apply(lambda x: len(str(x).split()))
	df_all['count_unique_word']=df_all["review_clean"].apply(lambda x: len(set(str(x).split())))
	df_all['count_letters']=df_all["review_clean"].apply(lambda x: len(str(x)))
	df_all["count_punctuations"] = df_all["review"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
	df_all["count_words_upper"] = df_all["review"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
	df_all["count_words_title"] = df_all["review"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
	df_all["count_stopwords"] = df_all["review"].apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))
	df_all["mean_word_len"] = df_all["review_clean"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

	df_all['season'] = df_all["month"].apply(lambda x: 1 if ((x>2) & (x<6)) else(2 if (x>5) & (x<9) else (3 if (x>8) & (x<12) else 4)))
	return df_all,len_train

def modelML(df_all,len_train):
	df_train = df_all[:len_train]
	df_test = df_all[len_train:]

	target = df_train['sentiment']
	feats = ['usefulCount','day','year','month','Predict_Sentiment','Predict_Sentiment2', 'count_sent','count_word', 'count_unique_word', 'count_letters', 'count_punctuations','count_words_upper', 'count_words_title', 'count_stopwords', 'mean_word_len', 'season']
	sub_preds = np.zeros(df_test.shape[0])

	trn_x, val_x, trn_y, val_y = train_test_split(df_train[feats], target, test_size=0.2, random_state=42) 
	feature_importance_df = pd.DataFrame() 
	    
	clf = LGBMClassifier(n_estimators=10000,learning_rate=0.10,num_leaves=30,subsample=.9,max_depth=7,reg_alpha=.1,reg_lambda=.1,min_split_gain=.01,min_child_weight=2,silent=-1,verbose=-1,)
	        
	clf.fit(trn_x, trn_y, eval_set= [(trn_x, trn_y), (val_x, val_y)], verbose=100, early_stopping_rounds=100)

	sub_preds = clf.predict(df_test[feats])
	        
	fold_importance_df = pd.DataFrame()
	fold_importance_df["feature"] = feats
	fold_importance_df["importance"] = clf.feature_importances_
	feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

	y_test = df_test['sentiment']
	solution = y_test.copy()
	metrics.confusion_matrix(y_pred=sub_preds, y_true=solution)

	cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index

	best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

	plt.figure(figsize=(14,10))
	sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
	plt.title('LightGBM Features (avg over folds)')
	plt.tight_layout()
	plt.savefig('lgbm_importances.png')
	return sub_preds,df_test


def userful_count(data):
    grouped = data.groupby(['condition']).size().reset_index(name='user_size')
    data = pd.merge(data,grouped,on='condition',how='left')
    return data

def dict_sentiment_model(df_test,sub_preds):
	word_table = pd.read_csv("inquirerbasic.csv")

	temp_Positiv = []
	Positiv_word_list = []
	for i in range(0,len(word_table.Positiv)):
	    if word_table.iloc[i,2] == "Positiv":
	        temp = word_table.iloc[i,0].lower()
	        temp1 = re.sub('\d+', '', temp)
	        temp2 = re.sub('#', '', temp1) 
	        temp_Positiv.append(temp2)

	Positiv_word_list = list(set(temp_Positiv))
	len(temp_Positiv)
	len(Positiv_word_list)  #del temp_Positiv

	#Negativ word list          
	temp_Negativ = []
	Negativ_word_list = []
	for i in range(0,len(word_table.Negativ)):
	    if word_table.iloc[i,3] == "Negativ":
	        temp = word_table.iloc[i,0].lower()
	        temp1 = re.sub('\d+', '', temp)
	        temp2 = re.sub('#', '', temp1) 
	        temp_Negativ.append(temp2)

	Negativ_word_list = list(set(temp_Negativ))
	len(temp_Negativ)
	len(Negativ_word_list)

	vectorizer = CountVectorizer(vocabulary = Positiv_word_list)
	content = df_test['review_clean']
	X = vectorizer.fit_transform(content)
	f = X.toarray()
	f = pd.DataFrame(f)
	f.columns=Positiv_word_list
	df_test["num_Positiv_word"] = f.sum(axis=1)

	vectorizer2 = CountVectorizer(vocabulary = Negativ_word_list)
	content = df_test['review_clean']
	X2 = vectorizer2.fit_transform(content)
	f2 = X2.toarray()
	f2 = pd.DataFrame(f2)
	f2.columns=Negativ_word_list
	df_test["num_Negativ_word"] = f2.sum(axis=1)

	df_test["Positiv_ratio"] = df_test["num_Positiv_word"]/(df_test["num_Positiv_word"]+df_test["num_Negativ_word"])
	df_test["sentiment_by_dic"] = df_test["Positiv_ratio"].apply(lambda x: 1 if (x>=0.5) else (0 if (x<0.5) else 0.5))

	df_test =  userful_count(df_test) 
	df_test['usefulCount'] = df_test['usefulCount']/df_test['user_size']

	df_test['machine_pred'] = sub_preds
	df_test['total_pred'] = (df_test['machine_pred'] + df_test['sentiment_by_dic'])*df_test['usefulCount']
	return df_test

def recommend(cond,df_test):
    something = df_test[(df_test['condition']==cond)]
    some = something.groupby(['drugName']).agg({'total_pred':['mean']}).sort_values(by = ('total_pred','mean'),ascending = False)
    return some[:1]
