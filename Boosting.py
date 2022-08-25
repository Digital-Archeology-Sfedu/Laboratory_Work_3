from sklearn.ensemble import GradientBoostingClassifier
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
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
ds = pd.read_csv('new_botanic.csv')
df = pd.DataFrame(ds)
#df = df.drop(columns=['leaf', 'shot', 'day'])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print(df.head())
df = df.sample(frac=1).reset_index(drop=True)
X = df.drop(columns=['obr'])
Y = df['obr']
# df1 = df.loc[df['obr'] == 1]
# df2 = df.loc[df['obr'] == 10]
# df3 = df.loc[df['obr'] == 11]
# df4 = df1.head(140000)
# df4 = pd.concat([df4, df2.head(140000)], axis=0)
# df4 = pd.concat([df4, df3.head(140000)], axis=0)
# df4.reset_index(inplace=True, drop=True)
# df4.to_csv('new_botanic.csv')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7)
print(df.head())
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled =ss.fit_transform(X_test)
y_train = np.array(y_train)
gb = GradientBoostingClassifier()
pca_test = PCA(n_components=10)
pca_test.fit(X_train_scaled)
X_train_scaled_pca = pca_test.transform(X_train_scaled)
X_test_scaled_pca = pca_test.transform(X_test_scaled)
gb.fit(X_train_scaled_pca, y_train)
y_pred = gb.predict(X_test_scaled_pca)
y_pred_prob = gb.predict_proba(X_test_scaled_pca)
x_pred = gb.predict(X_train_scaled_pca)
x_pred_prob = gb.predict_proba(X_train_scaled_pca)
#print(y_pred)
#print(y_test)
print(classification_report(y_test, y_pred))

#filename = 'models/gradient_boosting_model.sav'
#joblib.dump(gb, filename)
#filename = 'models/pca_model.sav'
#joblib.dump(pca_test, filename)

def evaluate_model(predictions, probs, train_predictions, train_probs, test_labels, train_labels):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    baseline = {}

    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))], average='macro')
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))], average='macro')
    baseline['roc'] = 0.5

    results = {}

    results['recall'] = recall_score(test_labels, predictions, average='macro')
    results['precision'] = precision_score(test_labels, predictions, average='macro')
    results['roc'] = roc_auc_score(test_labels, probs, average='macro', multi_class='ovr')

    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions, average='macro')
    train_results['precision'] = precision_score(train_labels, train_predictions, average='macro')
    train_results['roc'] = roc_auc_score(train_labels, train_probs, average='macro', multi_class='ovr')

    for metric in ['recall', 'precision', 'roc']:
        print(
            f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    # Считаем ложноположительные срабатывания
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))], pos_label=2)
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Плотим обе кривых
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('ROC Curves');

evaluate_model(y_pred, y_pred_prob, x_pred, x_pred_prob,y_test, y_train)
