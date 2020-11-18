import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

final = pd.read_csv('final.csv', parse_dates = ['Date'])
final.plot(x = 'Date')
plt.xlabel('Month', fontsize=16)
plt.ylabel('Avg Sentiment', fontsize=16)
plt.axvline(x = dt.datetime(2020, 6, 11), label = 'hello',ls = '--',color='r')
plt.text(dt.datetime(2020, 6, 11), 1,"11 Jun",rotation=90)
plt.axvline(x = dt.datetime(2020, 9, 7), label = 'hello',ls = '--',color='r')
plt.text(dt.datetime(2020, 9, 7), 1,"7 Sep",rotation=90)
plt.axvline(x = dt.datetime(2020, 9, 16), label = 'hello',ls = '--',color='r')
plt.text(dt.datetime(2020, 9, 16), 1,"16 Sep",rotation=90)
plt.axvline(x = dt.datetime(2020, 7, 23), label = 'hello',ls = '--',color='r')
plt.text(dt.datetime(2020, 7, 23), 1,"23 Jul",rotation=90)
plt.axvline(x = dt.datetime(2020, 8, 11), label = 'hello',ls = '--',color='r')
plt.text(dt.datetime(2020, 8, 11), 1,"11 Aug",rotation=90)
plt.axvline(x = dt.datetime(2020, 9, 21), label = 'hello',ls = '--',color='r')
plt.text(dt.datetime(2020, 9, 21), 1,"21 Sep",rotation=90)
plt.tight_layout()
plt.savefig('console_analysis.png')

comparison = pd.read_csv('comparison.csv', parse_dates = ['Date'])
comparison.plot(x = 'Date')
plt.xlabel('Month', fontsize=16)
plt.ylabel('Ratio of Negative Tweets', fontsize=16)
plt.tight_layout()
plt.savefig('console_negative.png')