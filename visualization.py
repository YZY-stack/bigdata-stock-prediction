import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
import matplotlib.dates as md

# data = pd.read_csv('stock.csv')
# # print(data.info)
# # print(data.head())
# data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
# data['Year'] = data['Date'].dt.year
# mm = ['Jan', 'Fed', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# data['Month'] = [mm[m-1] for m in data['Date'].dt.month]
# # print(data.head())
# data['Diff'] = data['Close'] - data['Open']
# data['Trend'] = ['Up' if x>=0 else 'Down' for x in data['Diff']]
# print(data.dtypes)
# for col in data.columns:
#     print(col, data[col].dtype)
# print(data.isnull().any())
'''timeticks = pd.date_range(min(data['Date']), max(data['Date']), periods=5)
fig, ax = plt.subplots(1, 2, figsize=(12, 8))
ax[0].plot(data['Date'], data['Open'], c='orangered')
ax[1].plot(data['Date'], data['Close'], c='limegreen')
# ax.plot(data['Date'], data['Open'])
# ax[0].set_xticks(timeticks)
# ax[0].set_title('Open')
# ax[1].set_xticks(timeticks)
# ax[1].set_title('Close')
fig.savefig('openclose.png')
plt.show()

fig2 = plt.figure()
hi, = plt.plot(data['Date'], data['High'], c='r')
lo, = plt.plot(data['Date'], data['Low'], c='g')
plt.legend(handles=[hi, lo], labels=['High', 'Low'])
plt.title('High/Low')
plt.ylabel('Price')
plt.show()
fig2.savefig('Highlow.png')


fig3 = plt.figure()
plt.plot(data['Date'], data['Volume'])
plt.title('Volume')
plt.show()
fig3.savefig('Volume.png')'''

'''fig4 = plt.figure()
sns.violinplot(data=data, x='Year', y='Close')
plt.title('Close')
plt.ylabel('Price')
plt.show()
fig4.savefig('Close--year.png')

fig5 = plt.figure()
sns.barplot(data=data, x='Year', hue='Trend', y='Volume', palette=['r', 'g'])
plt.title('Volume-year')
plt.show()
fig5.savefig('Volumebar.png')'''

'''fig6 = plt.figure()
plt.bar(data['Date'], data['Diff'])
plt.title('Close-Open')
plt.show()
fig6.savefig('Diff.png')

fig7 = plt.figure()
plt.plot(data['Date'], data['Adj Close'])
plt.title('Adj Close')
plt.show()
fig7.savefig('Adj.png')

fig8 = plt.figure()
hdata = data[["Year", "Month", "Volume"]].pivot_table(columns='Year',index='Month',values='Volume', aggfunc='mean')
cm = sns.diverging_palette(145, 20, as_cmap=True)
sns.heatmap(data=hdata, cmap=cm)
plt.title('Volume')
plt.show()
fig8.savefig('heatmap.png')'''

# text1 = open('metric_81.txt')
# for lines in text1:
data2 = pd.read_csv('metric_81.txt', sep=':', header=None)
# data2 = pd.DataFrame(data2)
data2 = data2.drop(data2.columns[[1]], axis=1)
data2.columns = ['comb', 'accuracy']
data2['comp1'] = [(c // 9)+1 for c in data2.index]
data2['comp2'] = [(c+1) % 9 if (c+1) % 9 != 0 else 9 for c in data2.index]
list = [1, 3, 5, 7, 10, 15, 20, 25, 30]
data2['comp1'] = [list[i-1] for i in data2['comp1']]
data2['comp2'] = [list[i-1] for i in data2['comp2']]
# print(data2)
data2 = data2.drop(data2.columns[[0]], axis=1)
data3 = data2.pivot_table(index='comp1', columns='comp2', values='accuracy')
# print(data3)
fig9 = plt.figure()
cm = sns.diverging_palette(145, 20, as_cmap=True)
cm2 = sns.color_palette('vlag', as_cmap=True)
ht = sns.heatmap(data=data3, cmap=cm, annot=True, fmt='.2f')
ht.invert_yaxis()
plt.title('Accuracy')
plt.show()
fig9.savefig('heatmap2.png')