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

# ++++++++GETTING FEATURES OF ONE SONG+++++++++++++

def choose_song(): 
    """ prompts the user to choose a song from those available in the folder""" 

    """ replace path to project folder with yours here: """

    basedir = current_folder_path
    file_list = []
    list_of_songs = []
    prompt_list = []
    file_promt = ''
    selected_song = ''
    file_list = listdir(basedir)

    for i in range(len(file_list)):
        if ".mp3" in file_list[i]:
            list_of_songs.append(file_list[i])

    for i in range(len(list_of_songs)): 
        s = str(i)
        file_prompt = "TYPE " + s + " FOR +++ " + list_of_songs[i] + " +++ \n"
        prompt_list.append(file_prompt)

    print("choose from the files below the song you wish to be genretized!")
    print("(this will only create the csv with song features. Use ml_generation for next steps)\n")

    for i in prompt_list: 
        print(i)

    song_id = input()

    song_id_int = int(song_id)

    selected_song = list_of_songs[song_id_int]

    print(selected_song)

    return selected_song

def create_csv_song(filename_s):
    """ """

    data_dict = []
    csv_columns_b = ["id"]
    basedir_b = "/Users/elliotbehling/Desktop/CS35/Final_Project"
    filepath_s = basedir_b + "/" + filename_s
    length_s = len(filename_s)
    just_song_name = filename_s[-length_s:-4]

    temp_dict = {}
    temp_dict["id"] = 0
    data_dict.append(temp_dict)

    for i in range(40):
        k = i+40
        l = i+80
        csv_columns_b.append(str(i))
        csv_columns_b.append(str(k))
        csv_columns_b.append(str(l))

    y, sr = librosa.load(filepath_s)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    for j in range(40):
        k = j + 40
        l = j + 80
        if len(beat_times) <= j: 
            data_dict[0][str(j)] = 0.0
            data_dict[0][str(k)] = 0.0
        else:
            data_dict[0][str(j)] = beat_times[j]
            data_dict[0][str(k)] = beat_frames[k-40]

        if len(chroma)<= j: 
            data_dict[0][str(l)] = 0.0
        else:
            # print(st.mean(chroma[l-80]))
            data_dict[0][str(l)] = st.mean(chroma[l-80])

    csv_file = just_song_name + ".csv"

    with open(csv_file, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_columns_b)
        writer.writeheader()
        for data in data_dict:
            writer.writerow(data)

print("done")

if __name__ == "__main__": 
    print("did this shit work?")

    song_name = choose_song()

    create_csv_song(song_name)