# -*- coding: UTF-8 -*-
import os
from os import listdir,mkdir,path
import shutil
trainrate=0.8

srcPath='D:/CodeSet/pycharm/homework1/Dataset/20news-18828'#源文件夹路径
toPath='D:/CodeSet/pycharm/homework1/Dataset/preprocessData'#预处理后数据路径
trainPath='D:/CodeSet/pycharm/homework1/Dataset/trainSet'#训练集数据路径
testPath='D:/CodeSet/pycharm/homework1/Dataset/testSet'#测试集数据路径

def devide():  #将预处理数据分为训练集和测试集
    for floder in os.listdir(toPath):
        if path.exists(trainPath+'/'+floder)==False:
            mkdir(trainPath+'/'+floder)
        if path.exists(testPath+'/'+floder)==False:
            mkdir(testPath+'/'+floder)
        floderPath=toPath+'/'+floder+'/'
        total=file_count(floderPath)
        count=0
        for file in os.listdir(floderPath):
            count+=1
            if count<=total*trainrate:
                shutil.copy(os.path.join(floderPath,file),trainPath+'/'+floder+'/'+file)
            else:
                shutil.copy(os.path.join(floderPath, file), testPath + '/' + floder + '/' + file)


#统计文件夹下的文件数量
def file_count(floderPath):
    count=0
    for file in os.listdir(floderPath):
        count+=1
    return count


if __name__ == '__main__':
    devide()
    print 'All data are devided'