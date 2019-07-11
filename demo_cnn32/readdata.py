# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:20:35 2019

@author: Liqi|leah0o
"""
import numpy as np 
import xlrd
import scipy.io as io
from sklearn.preprocessing import normalize as norm
from scipy.fftpack import fft


def shuru(data):
    """
    load data from xlsx
    """
    excel=xlrd.open_workbook(data);
    sheet=excel.sheets()[1];
    data=sheet.col_values(0)
    return data

def meanstd(data):    
    """
    to -1~1
    """
    for i in range(len(data)):
        datamean=np.mean(data[i])
        datastd=np.std(data[i])
        data[i]=(data[i]-datamean)/datastd
    
    return data

def sampling(data_this_doc,num_each,sample_lenth):
    """
    input:
        文件地址
        训练集的数量
        采样的长度
        故障的数量
    output:
        采样完的数据
    shuru->取长度->归一化
    """
            
    temp=shuru(data_this_doc)
#    temp=np.array(temp[1:]).reshape(-1,1)
#    temp=norm(temp)
    
    idx = np.random.randint(0, len(temp)-sample_lenth*2, num_each)
    temp_sample=[]
    for i in range(num_each):
        time=temp[idx[i]:idx[i]+sample_lenth*2]
        fre=abs(fft(time))[0:sample_lenth]
        temp_sample.append(fre) 
            
#    temp_sample=meanstd(temp_sample)
    temp_sample=np.array(temp_sample)
    temp_sample=norm(temp_sample)
#    temp_sample=temp_sample.reshape(num_each,sample_lenth)
    
    return temp_sample

class readdata():
    def __init__(self,data_doc,num_train=4000,ft=2,sample_lenth=1024):
        self.data_doc=data_doc
        self.num_train=num_train
        
        self.ft=ft
        self.sample_lenth=sample_lenth
        self.row=self.num_train//self.ft
    
    def concatx(self):
        """
        连接多个数据
        暂且需要有多少数据写多少数据
        """
        
        
        data=np.zeros((self.num_train,self.sample_lenth))
        for i,data_this_doc in enumerate(self.data_doc):
#            time=sampling(data_this_doc,self.row,self.sample_lenth*2)
#            freq=abs(fft(time))[0:self.sample_lenth]
#            data[0+i*self.row:(i+1)*self.row]=freq
            data[0+i*self.row:(i+1)*self.row]=sampling(data_this_doc,self.row,self.sample_lenth)
        return data

    def labelling(self):   #将data打包成一个元组
        """
        根据样本数和故障类型生成样本标签
        """
    
        label=np.zeros((self.num_train,self.ft))
        for i in range(self.ft):

            label[0+i*self.row:self.row+i*self.row,i]=1

        return label
       
    def output(self):
        data=self.concatx()
        
        label=self.labelling()
        size=int(float(self.sample_lenth)**0.5)
        data=data.astype('float32').reshape(self.num_train,1,size,size)
        label=label.astype('float32')
        return data,label
    

def dataset(data_name='sets',num_train=4000,sample_lenth=4096,test_rate=0.5):
    train_data_name=['/home/c/liki/EGAN/自家试验台数据/002/4.xlsx',       
                          '/home/c/liki/EGAN/自家试验台数据/010/4.xlsx',
                          '/home/c/liki/EGAN/自家试验台数据/029/4.xlsx',
                          '/home/c/liki/EGAN/自家试验台数据/053/4.xlsx',
                          '/home/c/liki/EGAN/自家试验台数据/014/4.xlsx',
                          '/home/c/liki/EGAN/自家试验台数据/037/4.xlsx',
                          '/home/c/liki/EGAN/自家试验台数据/061/4.xlsx',
                          '/home/c/liki/EGAN/自家试验台数据/006/4.xlsx',
                          '/home/c/liki/EGAN/自家试验台数据/034/4.xlsx',
                          '/home/c/liki/EGAN/自家试验台数据/057/4.xlsx'
                          ]
    test_data_name=train_data_name
    
    num_train=num_train

    if test_rate==0:
        testingset=readdata(test_data_name,
                             ft=len(train_data_name),
                             num_train=num_train,
                             sample_lenth=sample_lenth)
        
        
        x_test,y_test=testingset.output()
        io.savemat(data_name,{'x_test': x_test,'y_test': y_test,})
        return x_test,y_test
    else:
        
        trainingset=readdata(train_data_name,
                             ft=len(train_data_name),
                             num_train=num_train,
                             sample_lenth=sample_lenth)
        
        testingset=readdata(test_data_name,
                             ft=len(train_data_name),
                             num_train=int(num_train*test_rate),
                             sample_lenth=sample_lenth)
        
        x_train,y_train=trainingset.output()
        x_test,y_test=testingset.output()
        io.savemat(data_name,{'x_train': x_train,'y_train': y_train,'x_test': x_test,'y_test': y_test,})
        return x_train,y_train,x_test,y_test
