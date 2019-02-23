from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# read in data
test_data = pd.read_csv(filepath_or_buffer="./data/test.csv", delimiter=',').drop(['ID','PAID_NEXT_MONTH'], axis=1)


# normalize data
test_stats = test_data.describe().transpose()
normed_test_data = (test_data - test_stats['min']) / (test_stats['max'] - test_stats['min'])


# feature - selection -> univariate selection (based on train data) 
feature_select = SelectKBest(score_func=chi2, k=14)
fit = feature_select.fit(normed_test_data, train_labels)
selected_features = fit.transform(test_data)

#standardize data
sc = StandardScaler()
test_data = sc.fit_transform(selected_features)



