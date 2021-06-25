import lxml
import os
import xml.dom
import numpy as np
import torchvision.transforms as transforms


class loadVOC:
    def __init__(self, filename):
        self.SegmentationClass = os.path.join(filename, 'SegmentationClass')
        self.Annotations = os.path.join(filename, 'Annotations')
        self.ImageSets = os.path.join(filename, 'ImageSets')
        self.JPEGImages = os.path.join(filename, 'JPEGImages')
        self.labels = os.path.join(filename, 'labels')
        self.imagePath = []
        self.xmlPath = []
        self.hasObject = []
        self.bbox = np.array((2, 2))

    def loadTrain(self):
        file = open(self.ImageSets+'/Main/train_train.txt', 'r+')
        lines = [each for each in file.readlines()]
        self.imagePath = [self.JPEGImages + '/' + each.split(
            " ")[0]+'.jpg' for each in lines]
        self.hasObject = [each.split(" ")[1].strip() for each in lines]
        self.xmlPath = [self.Annotations + '/' + each.split(
            " ")[0]+'.xml' for each in lines]
        file.close()

    def getBbox(self):
        self.bbox = np.array((len(self.xmlPath), 2))
        # self.bbox = np.array((self.xmlPath.size, 2))
        # for i in self.xmlPath:


            # def ListFilesToTxt(dir, file, wildcard, recursion):
            #     exts = wildcard.split(" ")
            #     files = os.listdir(dir)
            #     for name in files:
            #         fullname = os.path.join(dir, name)
            #         if(os.path.isdir(fullname) & recursion):
            #             ListFilesToTxt(fullname, file, wildcard, recursion)
            #         else:
            #             for ext in exts:
            #                 if(name.endswith('.jpg')):
            # 		            name = name.split('.')[0]
            #                     file.write(name + "\n")
            #                     break

            # def Test():
            #   dir="/home/ghz/fast-rcnn/data/VOCdevkit/VOC2007/JPEGImages"
            #   outfile="allnames.txt"
            #   wildcard = ".txt .exe .dll .lib"

            #   file = open(outfile,"w")
            #   if not file:
            #     print ("cannot open the file %s for writing" % outfile)

            #   ListFilesToTxt(dir,file,wildcard, 1)

            #   file.close()


# loadV = loadVOC("/home/llrt/文档/VOCdevkit/VOC2012")
# loadV.loadTrain()
# loadV.getBbox()
