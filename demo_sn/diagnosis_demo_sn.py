#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:13:34 2019

@author: liqi
"""
from ShuffleNetV2 import ShuffleNetV2 as sfmodel
from readdata import dataset

import pandas as pd
import scipy
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import utils
#%%

class Diagnosis():
    '''
    diagnosis model
    '''
    def __init__(self,data_name='sets',n_class=10,lr=0.001,batch_size=64,num_train=100):
        print('diagnosis begin')
        
        model=sfmodel
        self.net=model(1,n_class,1.0)
        self.net.cuda()
        self.lr=lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=5,gamma = self.lr)
        self.loss_function = nn.CrossEntropyLoss()

        self.batch_size=batch_size

        self.save_dir='./'+data_name+'net_save'
        
        if os.path.exists(data_name+'.mat'):
            print('dataset is existed')
            data=scipy.io.loadmat(data_name+'.mat')
            
            self.x_train=data['x_train']
            self.x_test=data['x_test']     
            self.y_train=data['y_train']     
            self.y_test=data['y_test']
        else:
            print('new dataset')
            datasets=dataset(data_name=data_name,num_train=num_train,sample_lenth=4096,test_rate=0.5)
            self.x_train,self.y_train,self.x_test,self.y_test=datasets
        
        self.x_train=torch.from_numpy(self.x_train)
        self.y_train=torch.from_numpy(self.y_train)
        self.x_test=torch.from_numpy(self.x_test)
        self.y_test=torch.from_numpy(self.y_test)
        
        self.train_hist = {}
        self.train_hist['loss']=[]
        self.train_hist['acc']=[]
        self.train_hist['testloss']=[]
        self.train_hist['testacc']=[]
        
    def fit(self,epoches=1):
        
        print('training start!!')
        torch_dataset = TensorDataset(self.x_train,self.y_train)
        loader =DataLoader(
            dataset=torch_dataset,     
            batch_size=self.batch_size,      
            shuffle=True,               
            num_workers=2,              
            )
        
        whole_time=time.time()
        for epoch in range(epoches):
            loss_epoch=[]
            acc_epoch=[]
            correct = torch.zeros(1).squeeze().cuda()
            total = torch.zeros(1).squeeze().cuda()

            
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(loader):
                
                x_, y_ = x_.cuda(), y_.cuda()
                
                self.optimizer.zero_grad()
                y_pre = self.net(x_)
                loss = self.loss_function(y_pre, torch.max(y_, 1)[1])          
                prediction = torch.argmax(y_pre, 1)
                y_=torch.argmax(y_,1)
                correct += (prediction == y_).sum().float()
                total+=len(y_)



                loss_epoch.append(loss.item())
                loss.backward()
                self.optimizer.step()
                acc=(correct/total).cpu().detach().data.numpy()
                acc_epoch.append(acc)
                
            epoch_end_time = time.time()
            
            loss=np.mean(loss_epoch)
            acc=np.mean(acc_epoch)
            epoch_time=epoch_end_time-epoch_start_time
            self.train_hist['loss'].append(loss)         
            self.train_hist['acc'].append(acc)
            
            if epoch%10==0:
                print("Epoch: [%2d] Epoch_time:  [%8f] \n loss: %.8f, acc:%.8f" %
                              ((epoch + 1),epoch_time,loss,acc))
            self.evaluation()
        total_time=epoch_end_time-whole_time
        print('Total time: %2d'%total_time)
        print('best acc: %4f'%(np.max(self.train_hist['acc'])))
        
        if self.train_hist['testacc'][-1]>=np.max(self.train_hist['testacc']) :
        
        
            self.save()
            print(' a better model is saved')

                
                
        self.save_his()
        
        print('=========训练完成=========')

    def evaluation(self):
        
        #print('evaluation')
        
        self.net.eval()
        
        self.x_test=self.x_test.cuda()
        self.y_test=self.y_test.cuda()
        
        
        
        y_test_ori=torch.argmax(self.y_test,1)
        y_test_pre=self.net(self.x_test)
        test_loss=self.loss_function(y_test_pre,y_test_ori)
        y_tset_pre=torch.argmax(y_test_pre,1)
        

        
        correct = (y_tset_pre == y_test_ori).sum().float()
        n_samples=self.y_test.size()[0]
        acc=(correct/n_samples).cpu().detach().data.numpy()
        

        
        print("***\ntest result \n loss: %.8f, acc:%.4f\n***" %
              (test_loss.item(),
               acc))
        self.train_hist['testloss'].append(test_loss.item())

        self.train_hist['testacc'].append(acc)  
        

        self.net.train()

    def save(self):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


        torch.save(self.net.state_dict(), self.save_dir+'/net_parameters.pkl')

#        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
#            pickle.dump(self.train_hist, f)
    def save_pre(self,data,data_name='test'):
        
        data=data.cuda()
        final_pre=self.net(data)
        
        final_pre=torch.argmax(final_pre,1).cpu().detach().data.numpy()
        dic={}
        dic['prediction']=[]
        dic['prediction']=final_pre
        prediction=pd.DataFrame(dic)
        prediction.to_csv(self.save_dir+'/'+data_name+'prediction.csv')
        
    def save_his(self):
        
        data_df = pd.DataFrame(self.train_hist)
        data_df.to_csv(self.save_dir+'/history.csv')

    def load(self):
        
        self.net.load_state_dict(torch.load(self.save_dir +'/net_parameters.pkl'))
#def traintest_data():
    
def new_data_diag(data_name='extranewdata',num_train=1000):
    

    if os.path.exists(data_name+'.mat'):
        
        print('dataset is existed')
        data=scipy.io.loadmat(data_name+'.mat')
        new_x=data['x_test']
        new_y=data['y_test']
    else:
        print('new datasets')
        new_x,new_y=dataset(data_name=data_name,num_train=num_train,sample_lenth=4096,test_rate=0)
    new_x=torch.from_numpy(new_x)
    new_y=torch.from_numpy(new_y)
    diag.save_pre(new_x,data_name=data_name)

#%%    
if __name__ == '__main__':
    
    diag=Diagnosis(data_name='test',n_class=10,lr=0.001,batch_size=64,num_train=1000)
    diag.fit(1000)
    diag.evaluation()
    
    diag.load()
    diag.save_pre(diag.x_test)
#
    new_data_diag(data_name='extranewdata',num_train=1000)
    
    