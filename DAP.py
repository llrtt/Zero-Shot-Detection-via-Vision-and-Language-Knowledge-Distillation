# from sklearn import svm
import cuml.svm as svm
from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
import numpy as np
import modules.load_data as mload
import datetime
import pickle


def trainSVM(feature, attributes, model_name, atrribute_num):
    print('-----------------------------------training begin-------------------------------------------')
    start_time = datetime.datetime.now()
    Y = np.ones(34199)
    for i in range(34199):
        # remember class is 1-50, not from 0
        Y[i] = attributes[int(feature[i, 0]-1), atrribute_num]
    clf = svm.SVC(gamma=5, kernel='rbf', C=10)
    clf.fit(feature[:34199, 1:], Y)
    endtime = datetime.datetime.now()
    print('-----------------------------------training over--------------------------------------------')
    print("total cost time: %d s" % (endtime - start_time).seconds)
    print("saving model to %s.pkl" % model_name)
    pickle.dump(clf, open(model_name+".pkl", "wb"))
    print("model saved")
    print(" ")


def find_classPoisition(feature,  classes):
    classes_pos = {}
    current_class = classes[int(feature[0, 0])]
    start_point = 0
    end_point = 0
    last = current_class
    for i in range(feature.shape[0]):
        current_class = classes[int(feature[i, 0] - 1)]
        if current_class != last or i == (feature.shape[0]-1):
            end_point = i
            classes_pos.update({last: ("[%d, %d]" % (start_point, end_point))})
            start_point = i
        last = current_class
    with open('data.txt', 'w') as f:  # 设置文件对象
        f.write(str(classes_pos))  # 将字符串写入文件中


def loadSVM(model_path):
    return pickle.load(open(model_path, "rb"))


attriName = mload.load_attriName(
    "/home/llrt/文档/Animals_with_Attributes2/predicates.txt")
feature = mload.load_featureNpy("feature.npy")
attributes = mload.load_attribute(
    "/home/llrt/文档/Animals_with_Attributes2/predicate-matrix-binary.txt")
classes = mload.load_class(
    "/home/llrt/文档/Animals_with_Attributes2/classes.txt")

for i in range(len(attriName)):
    print("model %d" % (i+1))
    print(attriName[i])
    trainSVM(feature, attributes, attriName[i], i-1)


# X, Y = make_blobs(n_samples=40, centers=2, random_state=6)
# clf = svm.SVC(gamma=5, kernel='rbf', C=10)
# clf.fit(X, Y)
# plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)

# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()

# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.decision_function(xy).reshape(XX.shape)


# # 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
# # 所以contour函数是根据Z相同来画轮廓的
# ax.contour(XX, YY, Z, colors='k')
# ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[
#            :, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
# plt.show()
