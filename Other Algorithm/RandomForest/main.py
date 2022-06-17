import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, mean_squared_error, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.feature_extraction import DictVectorizer

# constant
NUM_DECISIONTREE = 100
TRAIN_PERCENT = 0.80
MAX_DEPTH = 3
MAX_LEAFNODES = 2
VERBOSE = 1
#CRITERION = 'gini'
CRITERION = 'entropy'
N_JOBS = 1
RANDOM_STATE = 32

def getTrainingOrTestingData(df):
	features = df.columns.drop(df.columns[-1])
	#print(features)
	target = df.columns[-1]
	X = df.loc[:,features]
	if ' Source IP' in df.columns:
		X = X.drop(columns=[' Source IP'])
	elif ' SrcHost' in df.columns:
		X = X.drop(columns=[' SrcHost'])
		X = X.drop(X.columns[0], axis = 1)

	print(X)
	y = df.loc[:,target]
	print(y)
	return X, y

def splitFixNum(df):
	df_benigns = df[df[df.columns[-1]]==0]
	df_attackers = df[df[df.columns[-1]]==1]
	df_benigns_train = df_benigns.sample(n = 20000, random_state = 30, replace = False, axis = 0)
	df_attackers_train = df_attackers.sample(n = 20000, random_state = 30, replace = False, axis = 0)
	#features = df.columns[:,0:df.columns.size[-1]].tolist()
	df1 = df_benigns[~df_benigns.index.isin(df_benigns_train.index)]
	df2 = df_attackers[~df_attackers.index.isin(df_attackers_train.index)]
	df_test = df1.append(df2)


	df_train = df_benigns_train.append(df_attackers_train)
	X_train, Y_train = getTrainingOrTestingData(df_train)
	X_test, Y_test = getTrainingOrTestingData(df_test)
	print('split result')
	print(X_train.shape[0])
	print(X_test.shape[0])
	return X_train, X_test, Y_train, Y_test

def loadData(path):
	print(f'Loading data from : {path}')
	df = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), path)))

	# df = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), path)), 
	# 	usecols = [' SrcHost','Fwd IAT Mean', 'Fwd IAT Std', 'Label'])

	print(f'Number of Rows: {df.shape[0]}')
	print(f'Number of Columns: {df.shape[1]}')
	if ' Source IP' in df.columns:
		print(f"Number of IP address: {df[' Source IP'].nunique(dropna=False)}")
	elif ' SrcHost' in df.columns:
		print(f"Number of IP address: {df[' SrcHost'].nunique(dropna=False)}")
	df_attackers = df.loc[df[df.columns[-1]] != 0]
	print(f'Number of IP attackers: {df_attackers[df_attackers.columns[-1]].value_counts(sort=True,dropna=False)}')
	#print(df_attackers)
	print(df)

	return df

def training(df):
	print(f'>>> Attributes NaN numbers : {df.isnull().sum().sum()}')
	df.fillna(-1, inplace=True)
	size = df.columns.size
	#print(target)
	#X, y = getTrainingOrTestingData(df)

	### Split the dataset
	#X_train_fs, X_test_fs, Y_train_fs, Y_test_fs = train_test_split(X, y, train_size=TRAIN_PERCENT)
	X_train_fs, X_test_fs, Y_train_fs, Y_test_fs = splitFixNum(df)

	# SVM_clasify = svm.SVC(kernel='rbf', C=1, gamma='auto', max_iter=3000)
	classifier = RandomForestClassifier(n_estimators = NUM_DECISIONTREE, criterion = CRITERION, random_state = RANDOM_STATE, max_depth = MAX_DEPTH)
	print('[Training] RandomForest')
	start_train = datetime.now()
	print(">>> Start Time: {}".format(start_train))
	classifier.fit(X_train_fs, Y_train_fs)
	train_score = classifier.score(X_train_fs, Y_train_fs)
	print(">>> Training Time: {}".format(datetime.now() - start_train))
	print(f'>>> Training Accuracy : {train_score*100.0}')
	print("")

	return X_test_fs, Y_test_fs, classifier

def testing(X_test_fs, Y_test_fs, model):
	print('[Testing] RandomForest')
	start_predict = datetime.now()
	y_pred = model.predict(X_test_fs)
	print(y_pred)
	df_pred = pd.DataFrame({'y_pred':y_pred })
	pred_list = df_pred[(df_pred.y_pred==1)].index.tolist()
	# print(pred_list)
	#y_pred_all = SVM_clasify.predict(X)
	df_Y_test_fs = pd.DataFrame({'Y_test_fs':Y_test_fs })
	df_Y_test_fs = df_Y_test_fs.reset_index(drop=True)
	test_list = df_Y_test_fs[(df_Y_test_fs.Y_test_fs==1)].index.tolist()
	print(f'Number of IP attackers: {df_Y_test_fs.value_counts(sort=False)}')
	# pred_list = df_pred[(df_pred.y_gb_pred==1)].index.tolist()
	
	correct_count = 0;
	for i in test_list:
	  if i in pred_list:
	    correct_count+=1
	print(f'Success predicted numbers of IP attackers: {correct_count}')
	print(f'>>> Testing time: {datetime.now() - start_predict}')
	print('Testing Data')
	df_Y_test_fs = pd.DataFrame({'Y_test_fs':Y_test_fs })
	df_Y_test_fs = df_Y_test_fs.reset_index(drop=True)
	acc = accuracy_score(Y_test_fs, y_pred)
	recall = recall_score(Y_test_fs, y_pred, average="macro")
	precision = precision_score(Y_test_fs, y_pred, average="macro", zero_division=0)
	mse = mean_squared_error(Y_test_fs, y_pred)
	f1score = f1_score(y_pred, Y_test_fs, average='macro')
	print("real attacker:")
	print(f'Number of IP attackers: {df_Y_test_fs.value_counts(sort=False)}')
	print("pred attacker:")
	print(f'Number of IP attackers (pred): {df_pred.value_counts(sort=False)}')
	print(">>> Metrics")
	print(f'- Accuracy  : {acc}')
	print(f'- Recall    : {recall}')
	print(f'- Precision : {precision}')
	print(f'- MSE       : {mse}')
	print(f'- F1 Score  : {f1score}')
	#target_names = df_Y_test_fs.drop_duplicates(subset=None, keep='first', inplace=False).values.tolist()
	labels = [0,1]
	print(classification_report(Y_test_fs, y_pred, labels = labels, target_names=['attack0','attack1'], digits = 16))

if __name__ == '__main__':
	path = sys.argv[1]
	df = loadData(path)
	X_test_fs, Y_test_fs, model = training(df)
	testing(X_test_fs, Y_test_fs, model)
	
	
	### saving model to disk
	filename = 'trained_model_RF.sav'
	pickle.dump(model, open(filename, 'wb'))
	
