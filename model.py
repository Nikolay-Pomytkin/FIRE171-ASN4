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
    training_data = sc.fit_transform(selected_features)

    return train_test_split(training_data, train_labels, test_size=0.2, random_state=42)


# After splitting data, account for class skew
def deskew_classes(data, labels):
    # input_data and input_labels are numpy arrays
    input_data = data
    input_labels = labels
    input_size = len(data)

    indexes = [i for i, x in enumerate(input_labels) if x == 0]
    zero_class_size = len(indexes)
    copied_data = [input_data[x] for x in indexes]
    # copied_data = np.ndarray(copied_data)
    zero_labels = [0 for x in indexes]
    # zero_labels = np.ndarray(zero_labels)
    import copy
    add_data = copy.deepcopy(copied_data)
    merge_data = copy.deepcopy(add_data)
    add_labels = zero_labels.copy()

    num_copies = 0
    while (zero_class_size/input_size) < 0.13:
        num_copies += 1
        zero_class_size += len(indexes)

    add_labels = np.tile(add_labels, num_copies+1)

    new_labels = np.concatenate((input_labels, add_labels), axis=None)
    new_data = np.concatenate((input_data, add_data), axis=0)

    print_add_size = add_labels.shape[0]
    print_total_size = new_labels.shape[0]


    # NEED TO FIX DESKEWER

    print("--------- De-skewed data ---------")
    print("New data info:")
    print("Total one class size: " + str(print_total_size - print_add_size - len(indexes)))
    print("Total zero class size: " + str(len(indexes)+print_add_size))
    print("Total zero class percentage: " + str((print_add_size + len(indexes))/print_total_size))

    return new_data, new_labels



def create_model(layers):
    # create model
    model = Sequential()

    for l in layers:
        model.add(l)

    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



# when function is called, prompts user on what variables to use for creating a new layer and returns that layer 
def create_layer(FEATURES):
    print("---------- Creating Layer ----------")
    TYPE = input("What type of layer (Dense or Dropout)? ")
    if(TYPE == "Dropout"):
        drp_prct = input("What percentage dropout? ")
        return Dropout(float(drp_prct))
    else:
        ACT_FUNC = input("What is the activation function you would like to use? ")
        NEURON_NUM = input("How many neurons does your layer have? ")
        TYPE2 = input("What type of layer is this (1 for input)? ")
        if(TYPE2 == "1"):
            return Dense(int(NEURON_NUM), input_dim=FEATURES, activation=ACT_FUNC, kernel_initializer='random_normal')
        else:
            return Dense(int(NEURON_NUM), activation=ACT_FUNC, kernel_initializer='random_normal')




# Actual start of program -> generating neural network from user input
running = True
while(running):
    print("---------------------------")
    print("Starting new model training")
    print("---------------------------")
    
    EPOCHS = int(input("How many epochs would you like to train the model for? "))
    BATCH_NUMBER = int(input("How many batches would you like to use? "))
    FEATURES = int(input("How many features would you like to train your model on? "))
    LAYERS = int(input("How many layers would you like your model to have? "))
    
    train_data_split, test_data_split, train_labels_split, test_labels_split = feature_selection(FEATURES)

    SKEW_BOOL = int(input("Would you like to account for class skew? (1 or 0) "))
    if SKEW_BOOL == 1:
        train_data_split, train_labels_split = deskew_classes(train_data_split, train_labels_split)

    from sklearn.utils import shuffle
    train_data_split, train_labels_split = shuffle(train_data_split, train_labels_split)

    layers = []
    i = 0
    while i < LAYERS:
        layers.append(create_layer(FEATURES))
        i += 1

    model = create_model(layers)

    # save model to history variable for visualization
    history = model.fit(train_data_split, train_labels_split, epochs=EPOCHS, validation_split=0.2, batch_size=BATCH_NUMBER)

    # plot training + test accuracy history
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # check model test accuracy
    scores = model.evaluate(test_data_split, test_labels_split)
    print(scores)

    print("Save model? (1 or 0)")
    save = int(input())

    if(save == 1):
        mod_name = input("Enter model name:")
        model_json = model.to_json()
        with open("./models/model_" + mod_name +".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("./models/model_" + mod_name + ".h5")
        print("Saved model to disk")
