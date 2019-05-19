# Jalankan Program ini di python lokal.
# Install libraries.

# Import libraries yang dibutuhkan dalam program  
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

# Fungsi import dataset yang digunakan 
def importdata(): 
	balance_data = pd.read_csv( 
'https://archive.ics.uci.edu/ml/machine-learning-'+
'databases/balance-scale/balance-scale.data', 
	sep= ',', header = None) 
	
	# Tampilkan info Dataset
	print ("Informasi Data\n")
	print ("Jumlah data: ", len(balance_data)) 
	print ("Dimensi data: ", balance_data.shape) 
	
	# Tampilkan Dataset pengamatan
	print ("Dataset: \n",balance_data.head()) 
	return balance_data 

# Fungsi untuk split dataset 
def splitdataset(balance_data): 

	# Pembagian variabel target 
	X = balance_data.values[:, 1:5] 
	Y = balance_data.values[:, 0] 

	# Split dataset menjadi data training dan data testing 
	X_train, X_test, y_train, y_test = train_test_split( 
	X, Y, test_size = 0.3, random_state = 100) 
	
	return X, Y, X_train, X_test, y_train, y_test 
	
# Fungsi untuk menunjukkan data training dengan GiniIndex 
def train_using_gini(X_train, X_test, y_train): 

	# Membuat objek klasifikasi 
	clf_gini = DecisionTreeClassifier(criterion = "gini", 
			random_state = 100,max_depth=3, min_samples_leaf=5) 

	# menunjukkan data training 
	clf_gini.fit(X_train, y_train) 
	return clf_gini 
	
# Fungsi untuk menunjukkan data training dengan Entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 

	# Pohon keputusan dengan entropy 
	clf_entropy = DecisionTreeClassifier( 
			criterion = "entropy", random_state = 100, 
			max_depth = 3, min_samples_leaf = 5) 

	# menunjukkan data training 
	clf_entropy.fit(X_train, y_train) 
	return clf_entropy 


# Fungsi untuk membuat prediksi
def prediction(X_test, clf_object):  
	y_pred = clf_object.predict(X_test) 
	print("Hasil prediksi:") 
	print(y_pred) 
	return y_pred 
	
# Fungsi menghitung akurasi 
def cal_accuracy(y_test, y_pred): 
	print ("Matriks: \n", confusion_matrix(y_test, y_pred)) 
	print ("Akurasi: ", accuracy_score(y_test,y_pred)*100) 
	print ("Laporan: ", classification_report(y_test, y_pred)) 

# Kode Utama 
def main(): 
	data = importdata() 
	X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
	clf_gini = train_using_gini(X_train, X_test, y_train) 
	clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
	 
	print("\nHasil hitung dengan Gini Index") 
	y_pred_gini = prediction(X_test, clf_gini) 
	cal_accuracy(y_test, y_pred_gini) 
	
	print("\nHasil hitung dengan Entropy") 
	y_pred_entropy = prediction(X_test, clf_entropy) 
	cal_accuracy(y_test, y_pred_entropy) 
		
main() 
