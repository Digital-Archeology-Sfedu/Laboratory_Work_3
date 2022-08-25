import pandas as pd
import sklearn
from Tools.scripts.dutree import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
ds = pd.read_csv('Csv_botanic.csv')
df = pd.DataFrame(ds)
df = df.drop(columns=['Column1.143'])
X = df.drop(columns=['leaf'])
Y = df['leaf']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=42)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled =ss.fit_transform(X_test)
y_train = np.array(y_train)
rfc = RandomForestClassifier()
pca_test = PCA(n_components=10)
pca_test.fit(X_train_scaled)
X_train_scaled_pca = pca_test.transform(X_train_scaled)
X_test_scaled_pca = pca_test.transform(X_test_scaled)
rfc.fit(X_train_scaled_pca, y_train)
y_pred = rfc.predict(X_test_scaled_pca)
print(classification_report(y_test, y_pred))