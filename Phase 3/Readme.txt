How to run the codebase:

sentiment_analysis.py:
install gensim, keras, sklearn, pandas, and numpy packages
update the TODOs in the file with paths to the data
positive and negative sentiment words have been provided,
but the sentiment140 dataset must be retrieved here:
https://www.kaggle.com/kazanova/sentiment140
sentiment_analysis.py encodes the sentiment140 dataset into training and test data splits

training.py
update the TODOs in the file with paths to the data
training.py loads the encoded data from sentiment_analysis.py and trains various models
it saves the neural net training out to a file


Tweet Scrapping:

Run TweepyScrapper.py. Make sure to first manually create csv files called 
Tweets-ps, Tweets-xbox and Tweets-all with the following index:
'id', 'username', 'retweetcount', 'text', 'tweetcreatedts', 'likes', 'hashtags',
'followers', 'location'

Tweets Analysis:
Make sure Analysis.py and the three csv files are in same directory. Update the 
location of keras model directory in line 116 if needed. Run analysis.py

Results will be stored as final.csv, comparison.csv, console_analysis.png and 
console_negative.png.
