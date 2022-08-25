from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import sklearn
from Tools.scripts.dutree import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import recall_score, classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
ds = pd.read_csv('Csv_botanic_class.csv')
df = pd.DataFrame(ds)
df = df.drop(columns=['Column1.143'])
X = df.drop(columns=['leaf'])
Y = df['leaf']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=42)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled =ss.fit_transform(X_test)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = np.array(y_train)
gb = XGBClassifier()
pca_test = PCA(n_components=10)
pca_test.fit(X_train_scaled)
X_train_scaled_pca = pca_test.transform(X_train_scaled)
X_test_scaled_pca = pca_test.transform(X_test_scaled)
gb.fit(X_train_scaled_pca, y_train)
y_pred = gb.predict(X_test_scaled_pca)
print(accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))