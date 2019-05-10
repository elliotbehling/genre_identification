# genre_identification

OVERVIEW:

Our project aimed to train scikitlearn's machine learning algorithm on a 1,000 song database categorized by genre to be able to identify a song's genre using only an mp3 file. 

COMPONENTS:

data_generation.py
this script takes in the dataset of songs and produces a set of features designed to capture the rhythmic and harmonic signature of the song. These features along with the genre of the song are then saved to a csv file for each song. 

ml_training.py
This script takes in the csv file and splits it into target data (genre) and feature data which is then split further into train and test. Then we use scikitlearn to train a machine learning algorithm on the train data and test it using the test data. This file includes in its main function the ability to test the algorithm's prediction of a single song. See below for details.

data - dataset folder of folders dividing 1000 song files into genre folders.

Other files: 

test_1_song.py generates a csv file for a single song which can then be used to test the machine learning algorithm in ml_training on a single song to predict its genre. 

song_data.csv - our pre-generated csv file with all the features pre-calculated from the 1000 song datafolder (this is here because the data_generation file takes ~20 minutes to run)

Lil Nas X - Old Town Road (feat. Billy Ray Cyrus) [Remix].mp3 - our provided 1 song test example, feel free to add more mp3's of your choosing to this folder if you wish to test them. See instructions below. 

Lil Nas X - Old Town Road (feat. Billy Ray Cyrus) [Remix].csv - the csv created by test_1_song on the example song that includes this song's features. Allows it to be run in ml_training. See instructions below. 

read_me.txt - this read_me file (contains instructions)

INSTRUCTIONS: 

Unzip zip file (contains a database of 1000 mp3 files, it might take some time)

Runing ml_training.py (without testing 1 song):
    This will provide you with the training iterations and a readout of various metrics on the success of our machine learning algorithm achieved in classifying the song_data.csv file by genre
    Note: to run, it must have the csv datafile already in the folder. This is all it needs to run as is. 

Running test_1_song.py to create csv file for 1 song: 
    1: Download a song as an mp3 and place it in the folder with the rest of these files and folders (or just use old town road again). 
    2: run script 
        This will prompt you to choose from the mp3 files available to you in this folder. One of them will be our example (old town road) and you should see your mp3 file as an option. Choose yours. This will generate the csv file for this song. It will be named <your song's name>.csv
        To test this song, follow the instructions on running ml_training.py with testing 1 song (below)

Running ml_training.py with testing 1 song: 
    1: uncomment these three lines of code (at the bottom in the __main__ function): 

    # song_csv = choose_song_csv()
    # print("\n\n++++++++++++ RUNNING ON ONE SONG  +++++++++++++\n\n")
    # test_song(song_csv)

    2: run script
        this will prompt you to choose your csv file. Choose either the example we included (old down road) or if you have run test_1_song.py, you can choose the csv you recently created that is named after your example song. 
        this will then use our machine learning algorithm to make a prediction about the genre of that song. Keep in mind, it has a ~30% accuracy rate, so it might get it wrong a few times. It's best to run it 10-20 times and see what occurs the most. 

Running data_generation.py: (only if you want to create your own version of song_data.csv which you already have)
    1: make sure the "data" folder is in the same folder as this file (it's pretty large)
    2: find a book to read (this script takes 15-20 minutes to run on our laptops)
    3: run script 
        this will begin printing the numbers 1-1000 (each represents calculating the features for a song in our database using librosa). When it gets to 1000, it's done. 

NOTES and CONCLUSIONS: 

feature generation: 
This project was made extremely challenging because of how difficult it is to generate features of a song that represent its musical signature. We used 3 main feature generating functions from Librosa, and each captured very different information, but it was not nearly enough to categorize our songs with any consistency. Of course, a larger songbank would have likely helped as well, but it would not have gotten us from our 30% success rate to the 80% success rate of the project we based ours on. We suspect that no amount of features could get one to this level of identification either, as they included a convolutional neural net in their project (and also used their own feature generation somehow). 

The functions we used from Librosa are: 

librosa.beat.beat_track which provided us with tempo which is the bpm and beat_frames which is a measure of beat "events" sampled at a number of regular time intervales throughout the song. 

librosa.frames_to_beat which provided us with beat_times which estimates the number of beat instances in a regular time interval

librosa.feature.chroma_stft which provided us with a list of the volume of pitch ranges sampled at a number of regular time intervals throughout the song. We averaged these to provide us with an estimator of the melody. 

Librosa Tutorial found here: https://librosa.github.io/librosa/tutorial.html

ml testing accuracy: 
Our ml testing accuracy varied a lot. We believe this is just due to how poor it was, and how small our sample size for testing was (150 songs) meaning if it fluctuates between properly categorizing or miscategorizing 15 songs, that's a 10% change in accuracy. Because of this, we found it best to run the file ~10 times and average the results. We found they averaged to around 30%. 

Concusions: 
Our main takeaway is that music itself is incredibly human, particularly the differences we hear are truly subtle and difficult to quantify. Even telling if something is a song or not is slightly challenging. Rhythmic identification is easiest, but it can still be tricked into thinking something is a song that's not. However, examples of older generations listening to young kid's music and saying "this isn't music!" just proves how truly subjective and wildly different our conceptions of music, good music, and categories of music are. We think this makes it clear that the level of information we take in when listening to a song is simply far beyond what we are able to extract using the tools available to us from an mp3. 

We believe implementing a convolutional neural network might have very promising results since it creates its own features which would likely be more effective than our "brute force" techniques. If we were to add to this project, this would be our next step. 

link to project we based ours on: 

Thanks for a great semester! Enjoy! 

Anthony Burre and Elliot Behling 
