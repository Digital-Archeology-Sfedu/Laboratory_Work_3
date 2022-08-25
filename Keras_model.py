import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ds = pd.read_csv('Csv_botanic_class.csv')
df = pd.DataFrame(ds)
df = df.drop(columns=['Column1.143'])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
X = df.drop(columns=['leaf'])
Y = df['leaf']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=42)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled =ss.fit_transform(X_test)
y_train = np.array(y_train)
# pca_test = PCA(n_components=10)
# pca_test.fit(X_train_scaled)
# X_train_scaled_pca = pca_test.transform(X_train_scaled)
# X_test_scaled_pca = pca_test.transform(X_test_scaled)

model = keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=256, output_dim=128, input_length=1))
model.add(tf.keras.layers.Dense(32, activation='relu'))
#model.add(tf.keras.layers.Conv1D(128, 3))
#model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(tf.keras.layers.Activation(activations.relu))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.LSTM(32, return_sequences=True, return_state=True))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.summary()

batch_size = 1024
epochs = 10

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

model.fit(X_train_scaled, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


score = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])