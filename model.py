from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dropout
import keras as K
from keras import losses
from keras import optimizers
from keras import utils
from keras.regularizers import l1, l2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# read in data
train_data = pd.read_csv(filepath_or_buffer="./data/train.csv", delimiter=',').drop(['ID'], axis=1)
train_labels = train_data.pop('PAID_NEXT_MONTH')

# normalize data
train_stats = train_data.describe().transpose()
normed_train_data = (train_data - train_stats['min']) / (train_stats['max'] - train_stats['min'])


# feature - selection -> univariate selection 
feature_select = SelectKBest(score_func=chi2, k=14)
fit = feature_select.fit(normed_train_data, train_labels)
selected_features = fit.transform(train_data)

#standardize data
sc = StandardScaler()
train_data = sc.fit_transform(selected_features)
train_data_split, test_data_split, train_labels_split, test_labels_split = train_test_split(train_data, train_labels, test_size=0.2)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(14, input_dim=14, kernel_initializer='random_normal', activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(6, kernel_initializer='random_normal', activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(1, kernel_initializer='random_normal', activation='sigmoid'))

    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = create_model()

# save model to history variable for visualization
history = model.fit(train_data_split, train_labels_split, epochs=35, validation_split=0.2)

# plot training + test accuracy history
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# check model test accuracy
scores = model.evaluate(test_data_split, test_labels_split)
print(scores)

print("Save model? (1 or 0)")
save = int(input())

if(save == 1):
    mod_name = input("enter model name:")
    model_json = model.to_json()
    with open("./models/model_" + mod_name +".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./models/model_" + mod_name + ".h5")
    print("Saved model to disk")