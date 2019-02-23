# import for model base
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dropout

# imports for loss functions, optimizers, utils, and metrics
from keras import losses, optimizers, utils, metrics

# imports for numpy and pandas
import numpy as np
import pandas as pd

# imports from ScikitLearn to Standardize and split data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# imports for plotting model training history 
import matplotlib.pyplot as plt

# keras import for creating files to save models
from keras.models import model_from_json

# imports from ScikitLearn for feature selection
from sklearn.feature_selection import SelectKBest, chi2



# read in data (don't need to put into a function -> happens the same way every time)
train_data = pd.read_csv(filepath_or_buffer="./data/train.csv", delimiter=',').drop(['ID'], axis=1)
train_labels = train_data.pop('PAID_NEXT_MONTH')

# normalize data
train_stats = train_data.describe().transpose()
normed_train_data = (train_data - train_stats['min']) / (train_stats['max'] - train_stats['min'])


def feature_selection(NUM_FEATURES):
    # feature - selection -> univariate selection 
    feature_select = SelectKBest(score_func=chi2, k=NUM_FEATURES)
    fit = feature_select.fit(normed_train_data, train_labels)
    selected_features = fit.transform(train_data)

    #standardize data
    sc = StandardScaler()
    train_data = sc.fit_transform(selected_features)
    train_data_split, test_data_split, train_labels_split, test_labels_split = train_test_split(train_data, train_labels, test_size=0.25)



# After splitting data, account for class skew



def create_model():
    # create model
    model = Sequential()
    model.add(Dense(42, input_dim=18, activation='tanh', kernel_initializer='random_normal'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



# when function is called, prompts user on what variables to use for creating a new layer and returns that layer 
def create_layer(FEATURES):
    TYPE = input("What type of layer (Dense or Dropout)?")
    if(TYPE == "Dropout"):
        drp_prct = input("What percentage dropout?")
        return Dropout(int(drp_prct))
    else:
        ACT_FUNC = input("What is the activation function you would like to use?")
        NEURON_NUM = input("How many neurons does your layer have? ")
        TYPE2 = input("What type of layer is this (Input or no)? ")
        if(TYPE2 == "Input"):
            
            return DENSE(int(NEURON_NUM), input_dim=)




# Actual start of program -> generating neural network from user input
running = True
while(running):
    print("---------------------------")
    print("Starting new model training")
    print("---------------------------")
    
    EPOCHS = input("How many epochs would you like to train the model for? ")
    BATCH_NUMBER = input("How many batches would you like to use? ")
    FEATURES = input("How many features would you like to train your model on? ")
    LAYERS = input("How many layers would you like your model to have? ")
    
    layers = []
    for i in range(0, LAYERS):
        layers[i] = create_layer(FEATURES)

    model = create_model()

    # save model to history variable for visualization
    history = model.fit(train_data_split, train_labels_split, epochs=200, validation_split=0.2, batch_size=90)

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

