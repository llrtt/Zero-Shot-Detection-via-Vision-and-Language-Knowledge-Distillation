import os
import pickle
import numpy as np


def load_feature(feature_path, label_path):
    r"""
    load features extracted from ResNet101 to a two demension array
    it can create an .npy file containing all the feature and it can be load at a super fast speed,so using this function onec is enough

    Parameter
    --------
    feature_path:path to feature.txt,each feature vector contains 2048 values
    label_path:labels of classes

    Returns
    --------
    sum_array:a combination of features and labels.
              the beginning of each feature vector is the label number of th class
    """
    feature_array = np.loadtxt(feature_path)
    label_array = np.loadtxt(label_path).reshape((feature_array.shape[0], 1))
    sum_array = np.concatenate((label_array, feature_array), axis=1)
    print(sum_array[2, 0:])
    np.save("feature.npy", sum_array)
    return sum_array


def load_class(class_path):
    classes = np.load(class_path)
    print(classes)


def load_featureNpy(npy_path):
    r"""
    load features array at a high speed

    Returns
    --------
    sum_array:a combination of features and labels.
              the beginning of each feature vector is the label number of th class
    """
    return np.load(npy_path)


if __name__ == '__main__':
    # load_class(
    #     "/home/llrt/文档/Animals_with_Attributes2/classes.txt")
    load_featureNpy("feature.npy")
