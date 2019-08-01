# conding = utf-8
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import numpy
import json
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

vectorizer = CountVectorizer(stop_words="english", decode_error="ignore")#用于feature_extraction建立词频（不能再函数中重复建立，否则之前训练的不能重复使用）

def feature_extraction(corpus, flag):#flag=0表示第一次训练词典，flag=1表示第二次直接使用词频词典

	if flag == 0:

		x_counts = vectorizer.fit_transform(corpus)#通过文本计算词典数量，下次如果再用就用vectorizer.transform
	else :
		x_counts = vectorizer.transform(corpus)
	#print(len(vectorizer.get_feature_names()))
	#print(len(X_counts.toarray()))
	tfidf_transformer = TfidfTransformer()
	x_tfidf = tfidf_transformer.fit_transform(x_counts)
	return x_tfidf
	
def naive_bayes_classifier_test():#用sklearn中自带的数据测试
	twenty_train = datasets.load_files("./data/20news-bydate/20news-bydate-train")
	twenty_test = datasets.load_files("./data/20news-bydate/20news-bydate-test")
	#print(len(twenty_train.target))
	x_tfidf = feature_extraction(twenty_train.data, 0)
	print(x_tfidf.shape)
	clf = MultinomialNB()
	clf.fit(x_tfidf, twenty_train.target)
	docs_new = ['God is love','OpenGL on the GPU is fast']
	X_new_tfidf = feature_extraction(docs_new, 1)  #计算TF-IDF
	print(X_new_tfidf.shape)
	y_pred = clf.predict(X_new_tfidf)
	for doc, category in zip(docs_new, y_pred):  #category是数字
		print(("%r => %s")%(doc, twenty_train.target_names[category]))

def naive_bayes_classifier(train_data, test_text):

	x_tfidf = feature_extraction(train_data[0], 0)
	print(x_tfidf.shape)
	clf = MultinomialNB()
	clf.fit(x_tfidf, train_data[1])
	X_new_tfidf = feature_extraction(test_text, 1)  #计算TF-IDF
	print(X_new_tfidf.shape)
	y_pred = clf.predict(X_new_tfidf)
	print(len(y_pred))
	return y_pred
		

def train():

	train_corpus = open('./data/train.tsv')
	test_corpus = open('./data/test.tsv')
	train_content = train_corpus.readline()
	test_content = test_corpus.readline()
	
	train_id = 1
	test_id = 1	
	train_text = []
	train_category = []
	test_text = []
	
	while True:#训练数据生成
		train_content = train_corpus.readline().split('\t')
		train_id = train_id+1
		if train_content == ['']:
			break
		train_text.append(train_content[2])
		train_category.append(train_content[3])
	
	while True:#测试数据生成
		test_content = test_corpus.readline().split('\t')
		test_id = test_id+1
		if test_content == ['']:
			break
		test_text.append(test_content[2])
	
	train_data = [train_text,train_category]
	test_category = naive_bayes_classifier(train_data, test_text)
	file_out = open('./data/out.tsv', 'w+')
	id = 156061
	print(type(int(test_category[0])))
	for i in range(len(test_category)):
		print("%d	%d"%(id,int(test_category[i])), file = file_out)
		id += 1
	print(id)
		
		
#feature_extraction()
#naive_bayes_classifier_test()
train()
