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
rfc = RandomForestClassifier()
pca_test = PCA(n_components=10)
pca_test.fit(X_train_scaled)
X_train_scaled_pca = pca_test.transform(X_train_scaled)
X_test_scaled_pca = pca_test.transform(X_test_scaled)
rfc.fit(X_train_scaled_pca, y_train)

y_pred = rfc.predict(X_test_scaled_pca)

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
bootstrap = [True, False]
param_dist = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rs = RandomizedSearchCV(rfc,
                        param_dist,
                        n_iter = 100,
                        cv = 3,
                        verbose = 1,
                        n_jobs=-1,
                        random_state=0)
rs.fit(X_train_scaled_pca, y_train)
print(rs.best_params_)