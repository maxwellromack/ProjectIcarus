import numpy as np
import pandas as pd

#Read data from dataset.csv and convert into numpy arrays
data_frame = pd.read_csv("dataset.csv")
data_array = data_frame.to_numpy()
labels = data_array[:, 34].astype(str)
features = np.delete(data_array, 34, 1)

#Convert strings in labels array into ints and set as type 'float32'
#Additionally, for the purposes of this project, data that has the label 'Enrolled' will be ignored
int_labels = np.empty(3630, dtype = 'float32')
remove_list = []
label_index = 0
for i in range(4424):
    if(labels[i] == "Dropout"):
        int_labels[label_index] = 0
        label_index += 1
    elif(labels[i] == "Graduate"):
        int_labels[label_index] = 1
        label_index += 1
    elif(labels[i] == "Enrolled"):
        remove_list.append(i)
labels = int_labels
features = np.delete(features, remove_list, 0)

#Clean up features array by removing biased features, one-hot encoding, and standardization
features = np.delete(features, [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 15, 18], 1)

course_encode = np.zeros((3630, 17), dtype = 'float32')
quali_encode = np.zeros((3630, 17), dtype = 'float32')
for i in range(3630):
    course = features[i, 0]
    course_encode[i, (course - 1)] = 1
    quali = features[i, 2]
    quali_encode[i, (quali - 1)] = 1
hot_features = np.concatenate((course_encode, quali_encode), axis = 1)

bool_features = np.vstack((features[:, 1], features[:, 3], features[:, 4], features[:, 5])).astype('float32').T
unstd_features = np.delete(features, [0, 1, 2, 3, 4, 5], 1).astype('float32')
std_features = np.empty((3630, 16), dtype = 'float32')
for i in range(16):
    mean = np.mean(unstd_features[:, i])
    std = np.std(unstd_features[:, i])
    for j in range(3630):
        std_features[j, i] = (unstd_features[j, i] - mean) / std

features = np.concatenate((std_features, bool_features, hot_features), axis = 1)

#Save the cleaned features and labels array to text files
np.savetxt('clean_features.txt', features, delimiter = ',')
np.savetxt('clean_labels.txt', labels, delimiter = ',')
