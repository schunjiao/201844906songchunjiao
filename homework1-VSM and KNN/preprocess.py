# -*- coding: UTF-8 -*-
import os
from os import listdir,mkdir,path
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

srcPath='D:/CodeSet/pycharm/homework1/Dataset/20news-18828'#源文件夹路径
toPath='D:/CodeSet/pycharm/homework1/Dataset/preprocessData'#预处理后数据
trainPath='D:/CodeSet/pycharm/homework1/Dataset/trainSet'
testPath='D:/CodeSet/pycharm/homework1/Dataset/testSet'

def read_file():
    print 'Read data:'
    for floder in os.listdir(srcPath):
        if path.exists(toPath+'/'+floder)==False:
            mkdir(toPath+'/'+floder)
        floderPath=srcPath+'/'+floder+'/'
        for file in os.listdir(floderPath):
            filePath = floderPath + file
            tarfilePath=toPath+'/'+floder+'/'+file
            targetfile=open(tarfilePath,'w')
            for line in open(filePath,'rb').readlines():
                line = line.decode("utf-8", errors='ignore')
                text_list=re.sub("[^a-zA-Z]", " ", line).split()
                stoplist = stopwords.words('english')
                for word in text_list:
                    word = PorterStemmer().stem(word.lower())
                    if word not in stoplist:
                        targetfile.write('%s\n' % word)
            targetfile.close()



# if __name__ == '__main__':
#     read_file()
#     print 'All data are read'
