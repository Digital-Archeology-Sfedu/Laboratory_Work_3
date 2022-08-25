import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import joblib
ds = pd.read_csv('new_botanic.csv')
df = pd.DataFrame(ds)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.sample(frac=1).reset_index(drop=True)
X = df.drop(columns=['obr'])
Y = df['obr']
pca = PCA(n_components=10)
ss = StandardScaler()
X = ss.fit_transform(X)
fit = pca.fit(X)
features = fit.transform(X)

KMean = KMeans(n_clusters=3)
labels = KMean.fit_predict(features)

filename = 'models/kmeans_model.sav'
joblib.dump(KMean, filename)


