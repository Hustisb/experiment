# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pandas import read_excel
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# read Excel
df = read_excel(r'C:\Users\withs\Desktop\20200221\Processed data.xlsx',Sheetname='Sheet1',header=0 )
print("keys of df:\n{}".format(df.keys()))
len1 = df['Weekday'].__len__()
# Weekday   7
# event1-3 :0
# event4  Duration_event4	Resource_event4	Cost_event4  2 + 1 + 2+ 1 = 6
# event5-6	Duration_event5	Resource_event5	Cost_event5  3 + 1 + 3 + 1 = 8
#           Duration_event6	Resource_event6	Cost_event6      1 + 3 + 1 = 5
# event7 Duration_event7	Resource_event7	Cost_event7  2 + 1 + 2 + 1 = 6
# 11, 17, 20, 24
# 13, 21, 26, 32
X1 = df.ix[:, ['Weekday', #7
               'event4', 'Resource_event4',
               'event5-6', 'Resource_event5',
               'Resource_event6',
               'event7', 'Resource_event7']].values
X2 = df.ix[:,['Duration_event4', 'Cost_event4','Duration_event5', 'Cost_event5', 'Duration_event6','Cost_event6', 'Duration_event7', 'Cost_event7']].values
y = df['Result'].values

#one-hot encoding
enc = OneHotEncoder()
enc.fit(X1)
X1 = enc.transform(X1).toarray()
print(enc.n_values_)
print(enc.feature_indices_)

# divide the process
res = X1[:, :11]
insert_index = [7, 11, 17, 20, 24]
for i in range(1, len(insert_index)):
    res = np.hstack((res, X2[:, 2*(i-1):2*i], X1[:, insert_index[i-1]:insert_index[i]]))
res = np.hstack((res, X2[:, 6:8]))

standardLst = [7, 13, 21, 26, 32]
for i in range(len(standardLst)):
    XXXX = res[:, :standardLst[i]]
    X_train, X_test, y_train, y_test = train_test_split( XXXX , y, random_state=0, test_size=.2 )
    print("X_train.shape: {}".format(X_train.shape))
    #decisionTree
    tree = RandomForestClassifier(random_state=0)
    tree.fit(X_train, y_train)
    print("Accuracy on train set: {:.4f}".format(tree.score(X_train, y_train)))
    print("Accuracy on test set: {:.4f}".format(tree.score(X_test, y_test)))

    fip = tree.feature_importances_
    fips = [sum(fip[:7])]

    for tmp in range(1, i+1):
        fips.append(sum(fip[:standardLst[tmp]]) - sum(fips))
    res1 = [format(x, '.2%') for x in fips]
    print("Feature importances:\n{}".format(res1))