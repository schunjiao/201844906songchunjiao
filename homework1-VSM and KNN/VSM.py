# -*- coding: UTF-8 -*-
import os
from os import listdir,mkdir,path
from collections import Counter
import math
import json

trainPath='D:/CodeSet/pycharm/homework1/Dataset/trainSet'
testPath='D:/CodeSet/pycharm/homework1/Dataset/testSet'
train_vector_Path='D:/CodeSet/pycharm/homework1/Dataset/trainVector.json'
test_vector_Path='D:/CodeSet/pycharm/homework1/Dataset/testVector.json'


def file_count(setPath):
    n = 0
    floderList = listdir(setPath)
    for i in range(len(floderList)):
        floderPath = setPath + '/' + floderList[i]
        fileList = listdir(floderPath)
        n+=len(fileList)
    return n

'''计算idf值'''
def cal_idf():
    files_num= file_count(trainPath)
    word_freq_dic = {} #每个词的词频
    word_df_dic = {} #每个词的初始df
    filterDf_dic = {} #经过词频筛选后的df字典
    idf_dic={} #每个词的idf
    for floder in os.listdir(trainPath):
        floderPath = trainPath + '/' + floder + '/'
        for file in os.listdir(floderPath):
            filePath = floderPath + file
            words = open(filePath, 'r').readlines()
            tmp_freq_dic = Counter(words)
            for key,value in tmp_freq_dic.items():
                key = key.strip()
                word_freq_dic[key] =word_freq_dic.get(key, 0) + value  #每个词的词频
                word_df_dic[key] =word_df_dic.get(key, 0) + 1  #筛选前每个词的df

            # print(len(word_df_dic))

            for key, value in word_freq_dic.items():
                if value > 8:
                    filterDf_dic[key]=word_df_dic[key]
            sorted_filterDf_dic_dic = sorted(filterDf_dic.items())
            for key,value in sorted_filterDf_dic_dic:
                idf_dic[key] = math.log10(files_num/(value + 1))
    return idf_dic

''''''
def createVector(srcPath,tgtPath):
    word_idf_dic = cal_idf()
    for floder in os.listdir(srcPath):
        floderPath = srcPath + '/' + floder + '/'
        for file in os.listdir(floderPath):

            file_vector_list=[] #用来存储每个文档的向量
            file_vector_list.append(floder) #先存储类名
            all_vector_list=[] #存储每个文档的向量

            word_tfidf_dic={} #存储每个词及其tf_idf值
            word_tf_dic={} #存储每个词及其tf值
            tmp_sort_tfidf_lst=[]
            file_dic = {}  #存储一个文档中tf_idf值居于前50的所有单词及其tf_idf值

            filePath = floderPath + file
            words = open(filePath, 'rb').readlines()
            for word in words:
                word=word.strip()
                if word in word_idf_dic:
                    if word in word_tf_dic:
                        word_tf_dic[word] += 1
                    else:
                        word_tf_dic[word] = 1

            max_num = word_tf_dic[max(word_tf_dic, key=word_tf_dic.get)]

            for i in word_tf_dic:
                word_tfidf_dic[i]=(word_tf_dic[i]/max_num)*(word_idf_dic[i])

            #将tfidf字典按照值降序排列（值越大，词越重要）
            tmp_sort_tfidf_lst=sorted(word_tfidf_dic.items(),key = lambda item:item[1],reverse=True)
            #一个文档生成一个向量，向量中仅包含值最大的前50个值
            count= 0
            for key, value in tmp_sort_tfidf_lst:
                if count > 50:
                    break
                count = count + 1
                file_dic[key] = value
            file_vector_list.append(file_dic)
            all_vector_list.append(file_vector_list)
        openw = open(tgtPath, 'w')
        json.dump(file_vector_list, openw, ensure_ascii=False)
        openw.close()



if __name__ == '__main__':
    createVector(trainPath,train_vector_Path)
    print '训练集向量创建成功！'
    createVector(testPath,test_vector_Path)
    print '测试集向量创建成功！'




