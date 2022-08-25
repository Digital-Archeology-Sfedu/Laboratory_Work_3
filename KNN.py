from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import sklearn
from Tools.scripts.dutree import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import recall_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
ds = pd.read_csv('new_botanic.csv')
df = pd.DataFrame(ds)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.sample(frac=1).reset_index(drop=True)
X = df.drop(columns=['obr'])
Y = df['obr']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled =ss.fit_transform(X_test)
y_train = np.array(y_train)
knn = KNeighborsClassifier(n_neighbors=3)
pca_test = PCA(n_components=10)
pca_test.fit(X_train_scaled)
X_train_scaled_pca = pca_test.transform(X_train_scaled)
X_test_scaled_pca = pca_test.transform(X_test_scaled)
knn.fit(X_train_scaled_pca, y_train)
y_pred = knn.predict(X_test_scaled_pca)
print(classification_report(y_test, y_pred))

filename = 'models/pca_model.sav'
joblib.dump(pca_test, filename)