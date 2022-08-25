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
#df = df.drop(columns=['Column1.143'])
X = df.drop(columns=['obr'])
Y = df['obr']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=42)
# ss = StandardScaler()
# X_train_scaled = ss.fit_transform(X_train)
# X_test_scaled =ss.fit_transform(X_test)
# y_train = np.array(y_train)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(df)
pca_test = PCA(n_components=30)
pca_test.fit(X_train_scaled)
sns.set(style='whitegrid')
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)
plt.show()
evr = pca_test.explained_variance_ratio_
cvr = np.cumsum(pca_test.explained_variance_ratio_)
pca_df = pd.DataFrame()
pca_df['Cumulative Variance Ratio'] = cvr
pca_df['Explained Variance Ratio'] = evr
print(pca_df.head(30))