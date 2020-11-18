Accomplishments
Yuying: preliminary research, contributed to writeup of phase 1 & 2, scheduled team meetings, contributed to final report and presentation

Nishith: setup a twitter scrapper that crawls twitter search for tweets related to consoles, perform analysis on the collected tweets, created visualisation and wrote the analysis for final report

Sam: preliminary research, contributed to writeup of phase 1 & 2, worked on preprocessing, trained models , contributed to final report and presentation

How to run the codebase:

sentiment_analysis.py:
install gensim, keras, sklearn, pandas, and numpy packages
update the TODOs in the file with paths to the data
positive and negative sentiment words have been provided,
but the sentiment140 dataset must be retrieved here:
https://www.kaggle.com/kazanova/sentiment140
sentiment_analysis.py encodes the sentiment140 dataset into training and test data splits
a sample training dataset is contained in the file "sample training dataset.csv"

training.py
update the TODOs in the file with paths to the data
training.py loads the encoded data from sentiment_analysis.py and trains various models
it saves the neural net training out to a file


Tweet Scrapping:

TweepyScrapper.py
Run TweepyScrapper.py. Modify start and end dates in line 46 and 47 for testing on a smaller dataset.
Warning: May result in errors if no negative tweets found in a small sample
a sample analysis dataset is contained in the file "sample analysis dataset.csv"

Tweets Analysis:
Make sure Analysis.py, Plots.py and the three csv files are in same directory. Update the 
location of keras model directory in line 116 if needed. Run analysis.py.

Analysis.py will create two files calles final.csv and comparion.csv. Open them and manually rename the first 
column from blank to 'Date'. Run Plots.py to generate analysis plots.

Results will be stored as final.csv, comparison.csv, console_analysis.png and 
console_negative.png.
