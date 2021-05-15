import os
import pickle
import numpy as np
from numpy.core.defchararray import array, decode


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
    r"""
    load all the class names and cut the column of class number

    Parameter
    --------
    feature_path:path to classes.txt

    Returns
    --------
    class_array:a ndarray contains all the class names.
    """
    class_array = np.loadtxt(class_path, dtype=str)
    class_array = np.delete(class_array, 0, 1)
    return class_array.reshape((50,))


def load_attribute(attribute_path):
    r"""
    load all the attributes vectors, each vector is 1x50 size. It means every class owns 50 attributes. e.g. striped, white etc

    Parameter
    --------
    feature_path:path to attributes.txt

    Returns
    --------
    class_array:a ndarray contains all the attributes vectors.
    """
    attribute_array = np.loadtxt(attribute_path)
    return attribute_array


def load_featureNpy(npy_path):
    r"""
    load features array at a high speed

    Returns
    --------
    sum_array:a combination of features and labels.
              the beginning of each feature vector is the label number of th class
    """
    return np.load(npy_path)


def load_attriName(attriName_path):
    r"""
    load attribute names to an array
    Returns
    --------
    sum_array:an array of attribute names
    """
    attriName_array = np.loadtxt(attriName_path, dtype=str)
    attriName_array = np.delete(attriName_array, 0, 1)
    return attriName_array.reshape((85,))


if __name__ == '__main__':
    # load_class(
    #     "/home/llrt/文档/Animals_with_Attributes2/classes.txt")
    # load_featureNpy("feature.npy")
    # load_attribute(
    #     "/home/llrt/文档/Animals_with_Attributes2/predicate-matrix-binary.txt")
    # print(load_attriName(
    #     "/home/llrt/文档/Animals_with_Attributes2/predicates.txt"))
    load_feature("/home/llrt/文档/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt","/home/llrt/文档/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt")
