
# -*- coding: UTF-8 -*-
import math
import json

train_vector_Path='D:/CodeSet/pycharm/homework1/Dataset/trainVector/vctrain.json'
test_vector_Path='D:/CodeSet/pycharm/homework1/Dataset/testVector/vctest.json'

'''计算数据集中每个向量（文档）的模，结果存储在list中'''
def cal_eachVC_len(aVC):
    sum=0
    for word,value in aVC[1].items():
        sum+=value**2
    result=math.sqrt(sum)
    return result
def cal_VClist_len(vcList):
    vc_len_list=[]
    for vc in vcList:
        vc_len_list.append(cal_eachVC_len(vc))
    return vc_len_list
'''计算两向量相似度'''
def cos_twoVC(vc1,vc2):
    multi=0
    for key in vc1[1]:
        if key in vc2[1]:
            multi+=vc1[1][key]*vc2[1][key]
    len1=cal_eachVC_len(vc1)
    len2=cal_eachVC_len(vc2)
    cos=multi/(len1*len2)
    return cos


def KNN():

    k=5

    #从json文件中读取向量
    open_tmp = open(train_vector_Path, 'r')
    train_vectors = json.load(open_tmp)
    open_tmp.close()
    # print (train_vectors)
    open_tmp = open(test_vector_Path, 'r')
    test_vectors = json.load(open_tmp)
    # print (test_vectors)
    open_tmp.close()

    for testVC in test_vectors: #testVC为测试集每一个向量，格式为[类名，{'word':tf_idf,'word':tf_idf,......}]
        all_cos_list=[] #存放一个测试集向量到训练集每个向量的[label,cos]
        for trainVC in train_vectors:
            each_cos_list=[]
            each_cos_list.append(trainVC[0])
            # print trainVC[0]
            each_cos_list.append(cos_twoVC(testVC,trainVC))
            # print each_cos_list
            all_cos_list.append(each_cos_list)
            # print all_cos_list

        sorted_cos_list=sorted(all_cos_list,key=lambda x:x[1],reverse=True)


        k_cos_dict = {}
        count = 0
        for key, value in sorted_cos_list:
            if count >= k:
                break
            count = count + 1
            k_cos_dict[key] = k_cos_dict.get(key, 0) + 1

        judge_type = ''
        max_type_count = 0

        for key,value in k_cos_dict.items():
            if value > max_type_count:
                max_type_count = value
                judge_type = key

        success = 0
        failure = 0
        if trainVC[0] == judge_type:
            success += 1
        else:
            failure += 1

    successp = (float(success)) / (float(success + failure))

    print('the performance of KNN:')
    print('total test numbers:', (success + failure))
    print('number of success:', success)
    print('number of failure:', failure)
    print('the success rate:', successp)








if __name__ == '__main__':

    KNN()
