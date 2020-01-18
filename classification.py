#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
from Tools import savefile, readfile
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from Tools import readfile, readbunchobj, writebunchobj
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
# train_set
trainpath = "train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)
classifiers=['Random Forest','Logistic','Naive Bayes','KNN','SVM','Desicion Tree']
classifier=[DecisionTreeClassifier().fit(train_set.tdm, train_set.label),
            neighbors.KNeighborsClassifier(n_neighbors=10).fit(train_set.tdm, train_set.label),
            MultinomialNB(0.001).fit(train_set.tdm, train_set.label),
            svm.SVC(kernel='linear').fit(train_set.tdm, train_set.label),
            LogisticRegression().fit(train_set.tdm, train_set.label),
            RandomForestClassifier(n_estimators=100,oob_score='True').fit(train_set.tdm, train_set.label)
           ]
# test_set
testpath = "test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)
dic={}
for i in [0,1,2,3,4,5]:
    predicted = classifier[i].predict(test_set.tdm)
    precision=metrics.precision_score(test_set.label, predicted, average='weighted')
    recall=metrics.recall_score(test_set.label, predicted, average='weighted')
    f1=metrics.f1_score(test_set.label, predicted, average='weighted')
    l=[f1,precision,recall]
    dic[classifiers[i]]=l
clfs=sorted(dic.items(), key=lambda d: d[1][0], reverse=True)

#KNN
k_range = range(1, 31)
 
k_scores = []
for k in range(1, 31):
    knn=neighbors.KNeighborsClassifier(n_neighbors=k).fit(train_set.tdm, train_set.label)
    predicted = knn.predict(test_set.tdm)
    precision=metrics.precision_score(test_set.label, predicted, average='weighted')
    recall=metrics.recall_score(test_set.label, predicted, average='weighted')
    f1=metrics.f1_score(test_set.label, predicted, average='weighted')
#     l=[f1,precision,recall]
#     dic[i]=l
    k_scores.append(f1)
# clfss=sorted(dic.items(), key=lambda d: d[1][0], reverse=True)
k_range = range(1, 31)
k_scores=[i+0.5 for i in k_scores[:30]]
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated F1')
plt.title('KNN')
plt.show()

#RF
k_range = [10,100,500,1000]
 
k_scores1 = []
for k in k_range:
    rf=RandomForestClassifier(n_estimators=k,criterion="entropy").fit(train_set.tdm, train_set.label)
    predicted = rf.predict(test_set.tdm)
#     precision=metrics.precision_score(test_set.label, predicted, average='weighted')
#     recall=metrics.recall_score(test_set.label, predicted, average='weighted')
    f1=metrics.f1_score(test_set.label, predicted, average='weighted')
#     l=[f1,precision,recall]
#     dic[i]=l
    k_scores1.append(f1)
# clfss=sorted(dic.items(), key=lambda d: d[1][0], reverse=True)

k_scores2 = []
for k in k_range:
    rf=RandomForestClassifier(n_estimators=k,criterion="gini").fit(train_set.tdm, train_set.label)
    predicted = rf.predict(test_set.tdm)
#     precision=metrics.precision_score(test_set.label, predicted, average='weighted')
#     recall=metrics.recall_score(test_set.label, predicted, average='weighted')
    f1=metrics.f1_score(test_set.label, predicted, average='weighted')
#     l=[f1,precision,recall]
#     dic[i]=l
    k_scores2.append(f1)
# clfss=sorted(dic.items(), key=lambda d: d[1][0], reverse=True)

plt.figure()
plt.plot(k_range, k_scores1,
         label='entropy',
         color='yellow',  linestyle=':',linewidth=4)

plt.plot(k_range, k_scores2,
         label='gini',
         color='cornflowerblue', linestyle=':', linewidth=4)

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('number of estimators')
plt.ylabel('Cross-Validated F1')
plt.title('Random Forest')
plt.legend(loc="lower right")
plt.show()

X = datasets.load_iris().data
y = datasets.load_iris().target
y = label_binarize(datasets.load_iris().target, classes=[0, 1, 2])
n_classes = y.shape[1]
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,random_state=0)
maxtrix=np.array([[635,35],[17,324]])
# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='Random Forest ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Logistic Regression ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip([1], colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='Naive Bayes ROC curve (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

labels=['C3-Art','C4-Literature']
plt.matshow(maxtrix)
plt.colorbar()
plt.xticks(np.arange(maxtrix.shape[1]),labels)
plt.yticks(np.arange(maxtrix.shape[1]),labels)
plt.show()

