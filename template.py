#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/DoomchitYJ/OSS

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def load_dataset(dataset_path):
	#To-Do: Implement this function
    data = pd.read_csv(dataset_path)
    return data

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
    data = dataset_df.drop(['target'], axis=1)
    feature_cnt = len(data.columns)
    cnt_0 = len(dataset_df.loc[dataset_df['target'] == 0])
    cnt_1 = len(dataset_df.loc[dataset_df['target'] == 1])
    return feature_cnt, cnt_0, cnt_1

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
    target = dataset_df['target']
    data = dataset_df.drop(['target'], axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=testset_size, shuffle=True)
    return x_train, x_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    pred_dt = dt.predict(x_test)
    acc = metrics.accuracy_score(y_test, pred_dt)
    prec = metrics.precision_score(y_test, pred_dt)
    recall = metrics.recall_score(y_test, pred_dt)

    return acc, prec, recall    

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    pred_rf = rf.predict(x_test)
    acc = metrics.accuracy_score(y_test, pred_rf)
    prec = metrics.precision_score(y_test, pred_rf)
    recall = metrics.recall_score(y_test, pred_rf)

    return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	scaler = StandardScaler()
	scaler.fit(x_train)
	x_train_scaled = scaler.transform(x_train)
	x_test_scaled = scaler.transform(x_test)
	
	svm = SVC()
	svm.fit(x_train_scaled, y_train)
	pred_svm = svm.predict(x_test_scaled)
	acc = metrics.accuracy_score(y_test, pred_svm)
	prec = metrics.precision_score(y_test, pred_svm)
	recall = metrics.recall_score(y_test, pred_svm)
	
	return acc, prec, recall    

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)