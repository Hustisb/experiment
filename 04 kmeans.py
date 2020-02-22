# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pandas import read_excel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans

# 读取表格数据
df = read_excel(r'C:\Users\withs\Desktop\20200221\Processed data.xlsx',Sheetname='Sheet1',header=0 )
len1 = df['Weekday'].__len__()
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

standardLst = [10, 12, 14, 25, 38, 51, 59]

# 2.RandomForest+ onehot +cost + n-kmeans
print("2.RandomForest + onehot +cost + n-kmeans:")
for i in range(len(standardLst)):
    XXXX = res[:, :standardLst[i]]
    print("X_train.shape: {}".format([8000, standardLst[i]]))
    n_clu = 1
    kmeans = KMeans(n_clusters= n_clu)
    kmeans.fit(XXXX)
    labels_Kmeans = np.array(kmeans.labels_).reshape(-1,1)
    data_split = []
    y_split = []
    for ii in range(n_clu):
        data_split.append(XXXX[np.where(labels_Kmeans == ii)[0]])
        y_split.append(y[np.where(labels_Kmeans == ii)[0]])
    train_scores = []
    test_scores = []
    for ii in range(len(data_split)):
        X_train, X_test, y_train, y_test = train_test_split(data_split[ii] , y_split[ii], random_state=0, test_size=.2 )
        # print("X_train.shape: {}".format(X_train.shape))
        tree = RandomForestClassifier(random_state=0)
        tree.fit(X_train, y_train)
        train_scores.append(tree.score(X_train, y_train))
        test_scores.append(tree.score(X_test, y_test))
        #f1
        # y_pred = tree.predict(X_test)
        # f1 = f1_score(y_test, y_pred, average='macro')
        # p = precision_score(y_test, y_pred, average='macro')
        # r = recall_score(y_test, y_pred, average='macro')
        # print("macro-F1: " + str(f1) + ", precision: " + str(p) + ", recall:" + str(r))
    train_score = 0
    test_score = 0
    for j in range(len(data_split)):
        train_score += train_scores[j]*len(data_split[j])
        test_score += test_scores[j]*len(data_split[j])
    train_score = train_score / len(XXXX)
    test_score = test_score / len(XXXX)

    print("Accuracy on train set: {:.4f}".format(train_score))
    print("Accuracy on test set: {:.4f}".format(test_score))