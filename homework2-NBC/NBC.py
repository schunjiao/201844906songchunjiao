# -*- coding: utf-8 -*-
import os
import collections
import math

trainset_path='D:\\CodeSet\\pycharm\\homework2\\dataset\\trainSet' #训练集数据路径
testset_path='D:\\CodeSet\\pycharm\\homework2\\dataset\\testSet' #测试集数据路径

'''统计训练集内的文档数目（计算每个类别出现概率时用得到）'''
def count_files(trainset_path):
    count=0
    floderlist=os.listdir(trainset_path)
    for floder in floderlist:
        floderpath=trainset_path+os.path.sep+floder
        filelist=os.listdir(floderpath)
        count=count+len(filelist)
    return count

# num=count_files(trainset_path)
# print('训练集文档总数：'+str(num))

'''为训练集创建向量：每一个类别为一个向量，向量的结构为[类名，类的词典长度（即类中所有单词的总数），该类别出现的概率，该类的字典] 即[string,int,float,{}]
整个训练集为一个列表，每个元素都是一个类别的向量'''
def get_trainVectors():

    totalfiles = count_files(trainset_path)
    floderlist = os.listdir(trainset_path)

    vectorlists =[] #存放整个训练集所有向量的列表

    j = 0 #为了显示向量个数
    for floder in floderlist:

        j = j+1

        vector=[] #一个向量
        vector.append(floder) #追加类名
        totalwords = 0 #为每个类的所有单词计数
        filecount = 0.0 #统计该类中的文档数目
        wordsdict = {} #该类的词典（包含该类的所有单词,value为该单词在类中出现的总次数）

        floderpath=trainset_path+os.path.sep+floder
        filelist=os.listdir(floderpath)
        for file in filelist:
            filecount += 1
            filepath = floderpath+os.path.sep+file
            lines = open(filepath).readlines() #len(lines)即为该文档中的单词总数
            totalwords += len(lines)
            wordcountdic = collections.Counter(lines) #词典中为文档中的所有词，以及在该文档中出现的总次数
            for key, value in wordcountdic.items():
                key = key.strip()
                wordsdict[key] = wordsdict.get(key, 0)+value
        vector.append(totalwords)
        p = filecount / totalfiles
        vector.append(p)
        vector.append(wordsdict)
        # print('第'+str(j)+'个向量：')
        # print(vector)
        vectorlists.append(vector)
    return vectorlists

# lists=get_trainVectors()

'''将测试集的每一个文档表示成向量，[类名，以该文档所有单词为元素的列表]。'''
def get_testVectors():

    trainVectors=[] #存放测试集所有向量的列表
    i=0 #统计向量个数

    floderlist=os.listdir(testset_path)
    for floder in floderlist:
        floderpath = testset_path + os.path.sep +floder
        filelist=os.listdir(floderpath)
        for file in filelist:

            i+=1

            vector=[] #一个文档表示成一个向量
            vector.append(floder) #追加类名
            words=[] #存放一个文档内所有的单词

            filepath=floderpath + os.path.sep +file
            lines=open(filepath).readlines()
            wordcountdic=collections.Counter(lines)
            for key,value in wordcountdic.items():
                key=key.strip()
                words.append(key)
            vector.append(words)
            # print('第'+str(i)+'个向量：')
            # print(vector)
            trainVectors.append(vector)
    return trainVectors

# get_testVectors()

def get_allwords():
    allwords=0
    floderlist = os.listdir(trainset_path)
    for floder in floderlist:
        floderpath = trainset_path + os.path.sep + floder
        filelist = os.listdir(floderpath)
        for file in filelist:
            filepath = floderpath + os.path.sep + file
            lines = open(filepath).readlines()  # len(lines)即为该文档中的单词总数
            allwords += len(lines)
    # print allwords
    return allwords

# get_allwords()




'''实现NBC的主要操作是计算后验概率P{ci|x}(其中Ci指某一类别，x指待分类的文档)。该后验概率的计算已转换为计算乘积：P{Ci}*每个词在该类别中出现的概率
为了避免乘积结果超出计算机计算的下限，采用取对数操作，将乘法转换为加法'''
def NB_classifier():
    print('创建向量：')
    trainVectorList=get_trainVectors()
    print('已成功创建'+str(len(trainVectorList))+'个训练集向量')
    testVectorsList=get_testVectors()
    print('已成功创建' + str(len(testVectorsList)) + '个测试集向量')

    allwords=get_allwords() #allwords为数据集内单词总数
    success = 0 #统计成功次数
    failure =0 #统计失败次数

    count=0 #记录测试集文档总数

    for i in range(len(testVectorsList)): #待分类的每一个文档

        count+=1

        eachclassp = []  # 存放一个文档在每个类别中的后验概率

        for j in range(len(trainVectorList)): #每一个类别

            p = trainVectorList[j][2] #概率初值设为该类别出现的概率
            p = math.log10(p)

            for word in testVectorsList[i][1]:
                numerator = float(trainVectorList[j][3].get(word, 0)+1)
                denominator = float(trainVectorList[j][1]+allwords)
                wordp = math.log10(numerator/denominator)

                p += wordp #至此算出了一个文档属于一个类别的概率

            eachclassp.append((trainVectorList[j][0],p))
        '''得到的eachclassp的结构为[(类名1，p),(),(),.....]'''
        print('第'+str(count)+'个文档：各个类别出现概率为：')
        print(eachclassp)
        eachclassp.sort(key=lambda x:x[1],reverse=True)
        print('第' + str(count) + '个文档：各个类别出现概率（排序后）为：')
        print(eachclassp)
        judgeclass = eachclassp[0][0]
        print(judgeclass)
        if judgeclass == testVectorsList[i][0]:
            success = success+1
        else:
            failure = failure + 1
    print ('测试集文档总数：' + str(len(testVectorsList)))
    print ('分类成功文档数：'+str(success))
    print ('分类失败文档数：'+str(failure))
    successrate = float(success)/len(testVectorsList)
    print ('成功率：'+str(successrate))

NB_classifier()





