import pandas
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
import random

DATA_TYPE = '.csv'
TRAIN_DATA = 'train'
TEST_DATA = 'test'
RESULT_DATA = 'results'
INPUT_PATH = 'data/original/'
OUTPUT_PATH = 'data/processed/'


dataframe = pandas.read_csv(INPUT_PATH + TRAIN_DATA + DATA_TYPE)

dataframe = dataframe.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age'], axis=1)
dataframe.replace(to_replace=dict(female=0, male=1), inplace=True)
dataframe.replace(to_replace=dict(C=0, S=0.5, Q=1), inplace=True)

# remove NaN rows
# dataframe = dataframe.dropna()
dataframe = dataframe.fillna(random.randrange(0, 1))
# dataframe = dataframe.fillna(dataframe.mean())

# normalize data
train = dataframe.values
train_min_max_scaler = preprocessing.MinMaxScaler()
train_scaled = train_min_max_scaler.fit_transform(train)
dataframe = pandas.DataFrame(train_scaled)


x_train = dataframe[dataframe.columns[2:8]].values.tolist()
y_train = dataframe[dataframe.columns[0:1]].values.tolist()


test_dt = pandas.read_csv(INPUT_PATH + TEST_DATA + DATA_TYPE)
test_dt = test_dt.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age'], axis=1)
test_dt.replace(to_replace=dict(female=0, male=1), inplace=True)
test_dt.replace(to_replace=dict(C=0, S=0.5, Q=1), inplace=True)

# remove NaN rows
# test_dt = test_dt.dropna()
test_dt = test_dt.fillna(random.randrange(0, 1))
# test_dt = test_dt.fillna(test_dt.mean())

# normalize data
test = test_dt.values
test_min_max_scaler = preprocessing.MinMaxScaler()
test_scaled = test_min_max_scaler.fit_transform(test)
test_dt = pandas.DataFrame(test_scaled)


x_test = test_dt[test_dt.columns[1:6]].values.tolist()
y_test = [0]*418


# x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.890, shuffle=False)
clf = MLPClassifier(hidden_layer_sizes=(100, 50, 100), activation='logistic', solver='lbfgs')
clf.fit(x_train, y_train)  # Fit data
prediction = clf.predict(x_test)  # Predict results for x_test
accs = accuracy_score(y_test, prediction)  # Accuracy Score
cm = confusion_matrix(y_test, prediction)  # Confusion Matrix

print(str(accs*100)+'%')
print(str(cm))

# use gender_submission csv as a refrence to create result
result_df = pandas.read_csv(INPUT_PATH + 'gender_submission' + DATA_TYPE)
series = pandas.Series(prediction)
result_df['Survived'] = series.values
result_df = result_df.astype(int)

result_df.to_csv(OUTPUT_PATH + RESULT_DATA + DATA_TYPE, index=False, sep=',', encoding='utf-8')
