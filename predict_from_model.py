# import from keras to load model from json file
from keras.models import model_from_json

# import ScikitLearn functions for standardization, feature selection, and data split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# importing pandas, numpy, and csv
import pandas as pd
import numpy as np
import csv

# read model name in from terminal
model_name = input("What is the name of the model you would like to use?")

# load json and create model
json_file = open('./models/model_' + model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("./models/model_" + model_name + ".h5")
print("Loaded model from disk")

# read in data
test_data = pd.read_csv(filepath_or_buffer="./data/test.csv", delimiter=',').drop(['ID','PAID_NEXT_MONTH'], axis=1)
train_data = pd.read_csv(filepath_or_buffer="./data/train.csv", delimiter=',').drop(['ID'], axis=1)
train_labels = train_data.pop('PAID_NEXT_MONTH')

# normalize data
test_stats = test_data.describe().transpose()
train_stats = train_data.describe().transpose()
normed_test_data = (test_data - test_stats['min']) / (test_stats['max'] - test_stats['min'])
normed_train_data = (train_data - train_stats['min']) / (test_stats['max'] - test_stats['min'])

# feature - selection -> univariate selection (based on train data)
feature_select = SelectKBest(score_func=chi2, k=loaded_model.get_layer(name='dense_1').get_config()['batch_input_shape'][1])
fit = feature_select.fit(normed_train_data, train_labels)
selected_features = fit.transform(test_data)

#standardize data
sc = StandardScaler()
test_data = sc.fit_transform(selected_features)

# generate output array from model predictions
predictions = loaded_model.predict(test_data)
print("Mean: " + str(predictions.mean()))
print("Median: " + str(np.median(predictions)))

# read output file name in from terminal
submission_name = input("What would you like to call the output file?")

# print to submission csv
with open('./predictions/submission_' + submission_name + '.csv', mode='w') as submission_file:
    submission_writer = csv.writer(submission_file, delimiter=',', )
    submission_writer.writerow(["ID","PAID_NEXT_MONTH"])
    for i in range(0,3000):
        predict_val = int(round(predictions[i][0]))
        submission_writer.writerow([27001 + i, predict_val])
