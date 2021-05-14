from sklearn import svm
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import modules.load_data as mload


def trainSVM(feature, attributes):
    Y = np.ones(2000)
    for i in range(2000):
        #remember class is 1-50, not from 0
        Y[i] = attributes[int(feature[i, 0]-1), 0]
    clf = svm.SVC(gamma=5, kernel='rbf', C=10)
    clf.fit(feature[:2000,1:], Y)

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
feature = mload.load_featureNpy("feature.npy")
attributes = mload.load_attribute(
    "/home/llrt/文档/Animals_with_Attributes2/predicate-matrix-binary.txt")
classes = mload.load_class(
    "/home/llrt/文档/Animals_with_Attributes2/classes.txt")
trainSVM(feature, attributes)
