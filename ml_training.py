# 
""" !conda update scikit-learn """

#import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
from os import listdir
from os import getcwd
from os.path import isfile, join
import csv

current_folder_path = getcwd()

print("+++ Start of song data +++\n")
# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('song_data.csv', header=0)    # read the file
df.head()                                 # first few lines
df.info()                                 # column details

def string_to_int(s):
    """ from string to number
          blues = 1
          classical = 2
          country = 3
          disco = 4
          hiphop = 5
          jazz = 6
          metal = 7
          pop = 8
          reggae = 9
          rock = 10


    """
    d = { 'blues':1, 'classical':2, 'country':3, 'disco':4, "hiphop":5, "jazz":6, "metal":7,"pop":8, "reggae":9, "rock":10 }
    return d[s]


def int_to_string(i):
    """ reverses string_to_int
    """
    d = { "1":"blues","2":"classical", "3":"country", "4":"disco", "5":"hiphop", "6":"jazz", "7":"metal", "8":"pop", "9":"reggae", "10":"rock" }
    return d[i]
    
# 
# this applies the function transform to a whole column
#
df['genre'] = df['genre'].map(string_to_int)  # apply the function to the column

print("\n+++ Converting to numpy arrays... +++\n")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_data_pre = df.iloc[:,2:].values         # iloc == "integer locations" of rows/cols
# print(X_data_pre)
y_data_pre = df[ 'genre' ].values       # individually addressable columns (by name)

#
# we can scramble the remaining data if we want to (we do)
# 
SIZE = len(y_data_pre)
indices = np.random.permutation(SIZE)  # this scrambles the data each time
X_data = X_data_pre[indices]
y_data = y_data_pre[indices]

#
# from the known data, create training and testing datasets
#

TRAIN_FRACTION = 0.85
TRAIN_SIZE = int(TRAIN_FRACTION*SIZE)
TEST_SIZE = SIZE - TRAIN_SIZE   # not really needed, but...
X_train = X_data[:TRAIN_SIZE]
y_train = y_data[:TRAIN_SIZE]

X_test = X_data[TRAIN_SIZE:]
y_test = y_data[TRAIN_SIZE:]

#
# it's important to keep the input values in the 0-to-1 or -1-to-1 range
#    This is done through the "StandardScaler" in scikit-learn
# 
USE_SCALER = True
if USE_SCALER == True:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)   # Fit only to the training dataframe
    # now, rescale inputs -- both testing and training
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

# scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 
#
# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(10,10,), max_iter=200, alpha=1e-4,
                    solver='sgd', verbose=True, shuffle=True, early_stopping = False, # tol=1e-4, 
                    random_state=None, # reproduceability
                    learning_rate_init=.1, learning_rate = 'adaptive')

print("\n++++++++++  TRAINING  +++++++++++++++\n")
mlp.fit(X_train, y_train)

print("\n++++++++++++  TESTING  +++++++++++++\n")
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# let's see the coefficients -- the nnet weights!
# CS = [coef.shape for coef in mlp.coefs_]
# print(CS)

# predictions:
predictions = mlp.predict(X_test)
# print("predictions")
# print(predictions)
from sklearn.metrics import classification_report,confusion_matrix
print("\nConfusion matrix:")
print(confusion_matrix(y_test,predictions))

print("\nClassification report")
print(classification_report(y_test,predictions))


def choose_song_csv(): 
    """ prompts the user to choose a song from those available in the folder""" 
    basedir_b = current_folder_path
    file_list = []
    list_of_songs = []
    prompt_list = []
    file_promt = ''
    selected_song = ''
    file_list = listdir(basedir_b)

    for i in range(len(file_list)):
        if ".csv" in file_list[i]:
            list_of_songs.append(file_list[i])

    for i in range(len(list_of_songs)):
        s = str(i)
        file_prompt = "TYPE " + s + " FOR +++ " + list_of_songs[i] + " +++ \n"
        prompt_list.append(file_prompt)

    print("choose from the files below the song you wish to be genretized!")
    print("(if you don't see your file, you need to create a csv file in it using \n the test_1_song script)\n")

    for i in prompt_list: 
        print(i)

    song_id = input()

    song_id_int = int(song_id)

    selected_song = list_of_songs[song_id_int]

    print(selected_song)

    return selected_song

def test_song(csv_name):
    """reads the csv of the features of one song and outputs the predicted genre"""

    length = int(len(csv_name))
    just_name = csv_name[-length:-4]
    # print(length)
    df = pd.read_csv(csv_name, header=0)    # read the file
    X_song = df.iloc[:,1:].values               #get x data
    
    # USE_SCALER = True
    # if USE_SCALER == True:
    #     from sklearn.preprocessing import StandardScaler
    #     scaler = StandardScaler()
    #     scaler.fit(X_song)   # Fit only to the training dataframe
    #     # now, rescale inputs -- both testing and training
    #     X_song = scaler.transform(X_song)

    predictions = mlp.predict(X_song)           #make predictions
    print("predicted genre of song " + just_name + " is:")
    key = str(predictions[0])
    print(int_to_string(key))

if False:
    L = [5.2, 4.1, 1.5, 0.1]
    row = np.array(L)  # makes an array-row
    row = row.reshape(1,4)   # makes an array of array-row
    if USE_SCALER == True:
        row = scaler.transform(row)
    print("\nrow is", row)
    print("mlp.predict_proba(row) == ", mlp.predict_proba(row))

# C = R.reshape(-1,1)  # make a column!

if __name__ == "__main__":

    print("done - uncomment rest of main for testing 1 song (run the test_1_song file first) \n")
    """ to test 1 song, use the test_1_song.py file to create a csv for that song, then uncomment the three lines of code below. """
    # song_csv = choose_song_csv()
    # print("\n\n++++++++++++ RUNNING ON ONE SONG  +++++++++++++\n\n")
    # test_song(song_csv)