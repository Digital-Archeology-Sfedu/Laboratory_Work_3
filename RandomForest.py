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
df = df.drop(columns=[''])
#print(df.head())

#pca = PCA(n_components=10)
#df = preprocessing.normalize(df)
X = df.drop(columns=['leaf'])
Y = df['leaf']
#X = preprocessing.normalize(X)
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=42)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled =ss.fit_transform(X_test)
y_train = np.array(y_train)
rfc = RandomForestClassifier(n_estimators=600, min_samples_split=23, min_samples_leaf=2, max_features='sqrt', max_depth=15, bootstrap='False')
pca_test = PCA(n_components=10)
pca_test.fit(X_train_scaled)
X_train_scaled_pca = pca_test.transform(X_train_scaled)
X_test_scaled_pca = pca_test.transform(X_test_scaled)
rfc.fit(X_train_scaled_pca, y_train)
#rfc.fit(X_train_scaled_pca, y_train)
#print(rfc.score(X_train_scaled, y_train))

y_pred = rfc.predict(X_test_scaled_pca)
# print(y_test)
# feats = {}
# for feature, importance in zip(df.columns, rfc.feature_importances_):
#     feats[feature] = importance
# importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
# importances = importances.sort_values(by='Gini-Importance', ascending=False)
# importances = importances.reset_index()
# importances = importances.rename(columns={'index': 'Features'})
# sns.set(font_scale = 5)
# sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
# fig, ax = plt.subplots()
# fig.set_size_inches(30,15)
# sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
# plt.xlabel('Importance', fontsize=25, weight = 'bold')
# plt.ylabel('Features', fontsize=25, weight = 'bold')
# plt.title('Feature Importance', fontsize=25, weight = 'bold')
# #display(plt.show())
# #display(importances)
#
print(classification_report(y_test, y_pred))
#fit = pca.fit(X)
#features = fit.transform(X)
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# max_features = ['log2', 'sqrt']
# max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
# min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
# min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
# bootstrap = [True, False]
# param_dist = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# rs = RandomizedSearchCV(rfc,
#                         param_dist,
#                         n_iter = 100,
#                         cv = 3,
#                         verbose = 1,
#                         n_jobs=-1,
#                         random_state=0)
# rs.fit(X_train_scaled_pca, y_train)
# print(rs.best_params_)


# sns.set(style='whitegrid')
# plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)
# plt.show()
# evr = pca_test.explained_variance_ratio_
# cvr = np.cumsum(pca_test.explained_variance_ratio_)
# pca_df = pd.DataFrame()
# pca_df['Cumulative Variance Ratio'] = cvr
# pca_df['Explained Variance Ratio'] = evr
# pca_df.head(10)
