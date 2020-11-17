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