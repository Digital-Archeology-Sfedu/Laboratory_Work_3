import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import joblib
ds = pd.read_csv('Csv_botanic_class.csv')
df = pd.DataFrame(ds)
df = df.drop(columns=['Column1.143'])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
pca = PCA(n_components=10)
X = df.drop(columns=['leaf'])
ss = StandardScaler()
X = ss.fit_transform(X)
fit = pca.fit(X)
features = fit.transform(X)
filename = 'models/kmeans_model.sav'
kmean = joblib.load(filename)
df1 = kmean.predict(features[17047:17147])
print(df1)
