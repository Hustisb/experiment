# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pandas import read_excel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# 读取表格数据
df = read_excel(r'C:\Users\withs\Desktop\20200221\Processed data.xlsx',Sheetname='Sheet1',header=0 )
len1 = df['Weekday'].__len__()
# step0: Weekday 7
# step1: Register Request_event1	Check Date_event2	Audit Mode_event3 3+2+2
# step2: Manual Review_event4	Review the Reason_event4	Duration_event4 Resource_event4	Cost_event4	4+3+1+2+1=11
# step3: Check Ticket_event5	Examine_event5	Duration_event5	Resource_event5  Cost_event5 3+5+1+3+1 = 13
# step4: Check Ticket_event6 Examine_event6	Duration_event6	Resource_event6	Cost_event6 3+5+1+3+1 =13
# step5: Decide_event7	Duration_event7	Resource_event7	Cost_event7 4+2+1+1 = 8
X = df.ix[:,['Weekday',  #7
             'Register Request_event1','Check Date_event2','Audit Mode_event3',  #7
             'Manual Review_event4',  'Review the Reason_event4', 'Resource_event4',  #9
             'Check Ticket_event5', 'Examine_event5', 'Resource_event5', #11
             'Check Ticket_event6', 'Examine_event6', 'Resource_event6', #11
             'Decide_event7', 'Resource_event7']].values  #6
X2 = df.ix[:,['Duration_event4','Cost_event4', 'Duration_event5','Cost_event5','Duration_event6', 'Cost_event6', 'Duration_event7','Cost_event7']].values
y = df['Result'].values
#one-hot编码
enc = OneHotEncoder()
enc.fit(X)
XX = enc.transform(X).toarray()
#划分数据集
# print(enc.n_values_)
# print(enc.feature_indices_)
arr1 = [23, 34, 45, 51]
res = XX[:, :23]
for i in range(1, len(arr1)):
    res = np.hstack((res, X2[:, 2*(i - 1): 2*i], XX[:, arr1[i-1]:arr1[i]]))
res = np.hstack((res, X2[:, 6:8]))

print("3.RandomForest + onehot + laststate:")
standardLst = [7, 10, 12, 14, 25, 38, 51, 59]
for i in range(1,len(standardLst)):
    XXXX = np.hstack((res[:, :7], res[:, standardLst[i-1]:standardLst[i]]))
    X_train, X_test, y_train, y_test = train_test_split( XXXX , y, random_state=0, test_size=.2 )
    print("X_train.shape: {}".format(X_train.shape))
    #decisionTree
    tree = RandomForestClassifier(random_state=0)
    tree.fit(X_train, y_train)
    print("Accuracy on train set: {:.4f}".format(tree.score(X_train, y_train)))
    print("Accuracy on test set: {:.4f}".format(tree.score(X_test, y_test)))