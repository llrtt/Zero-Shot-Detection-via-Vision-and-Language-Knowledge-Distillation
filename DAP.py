from sklearn import svm
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, Y = make_blobs(n_samples=40, centers=2, random_state=6)
clf = svm.SVC(kernel='rbf')
clf.fit(X, Y)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
# 所以contour函数是根据Z相同来画轮廓的
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[
           :, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.show()
