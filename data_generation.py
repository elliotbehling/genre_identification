# Beat tracking example
from __future__ import print_function
import librosa
from os import listdir
from os import getcwd
from os.path import isfile, join
import csv
import pandas as pd
import statistics as st

current_folder_path = getcwd()

print('starting')
csv_columns = ["id", "genre",]

data_dict = []
temp_id = 0

# print("beginning test")
# onlyfiles = [f for f in listdir("/Users/elliotbehling/Desktop/CS35/Final_Project/data") if isfile(join("/Users/elliotbehling/Desktop/CS35/Final_Project/data", f))]
# print(onlyfiles)

filepaths = []
level1 = []
basedir = current_folder_path + "/data/genres"
level1 = listdir(basedir)
level1.remove(".DS_Store")
# print(level1)

for i in range(len(level1)):
# for i in range(len(1)):
    path = basedir + "/" + level1[i]
    genre = level1[i]
    genrefiles = listdir(path)

    for j in range(len(genrefiles)):
        temp_id += 1
        temp = genrefiles[j]
        genrefiles[j] = path + "/" + temp

        temp_dict = {} 
        temp_dict["id"] = temp_id
        temp_dict["genre"] = genre
        # print(temp_dict)
        data_dict.append(temp_dict)

    filepaths += genrefiles

print("done foldering")

# print(data_dict)
# print(filepaths)

# 1. Get the file path to the included audio example
# filename = "/Users/elliotbehling/Desktop/CS35/Final_Project/data/genres/blues/blues.00000.au"

# 2. Load the audio as a waveform `y
#    Store the sampling rate as `sr

tracker = []

for i in range(40):
    k = i+40
    l = i+80
    csv_columns.append(str(i))
    csv_columns.append(str(k))
    csv_columns.append(str(l))

# print(csv_columns)

print("counting to 1000 (generating features)")

for i in range(len(filepaths)):

    filename = filepaths[i]

    y, sr = librosa.load(filename)

    # 3. Run the default beat tracker
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    # 4. Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    for j in range(40):
        k = j + 40
        l = j + 80
        if len(beat_times) <= j:
            data_dict[i][str(j)] = 0.0
            data_dict[i][str(k)] = 0.0
        else:
            data_dict[i][str(j)] = beat_times[j]
            data_dict[i][str(k)] = beat_frames[k-40]

        if len(chroma)<= j:
            data_dict[i][str(l)] = 0.0
        else:
            data_dict[i][str(l)] = st.mean(chroma[l-80])

    print(data_dict[i]["id"])

    # data_dict[i]["bpm"] = tempo

print("just write it out")

csv_file = "song_data.csv"

with open(csv_file, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = csv_columns)
    writer.writeheader()
    for data in data_dict:
        writer.writerow(data)
print("done")

        # print('Saving output to beat_times.csv')
        # librosa.output.times_csv('beat_times.csv', beat_times)