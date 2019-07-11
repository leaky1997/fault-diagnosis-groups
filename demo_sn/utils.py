import os
import torch.nn as nn
import numpy as np
import scipy.misc

import matplotlib.pyplot as plt

def figplot(dic,fig_name):
    for i in dic.keys():
        plt.title('Result Analysis')
        plt.plot(dic[i],label=i)
        plt.legend() # 显示图例
        plt.xlabel('iteration times')
        plt.ylabel('rate')
    plt.savefig(fig_name+'.png',dpi=512)   
    plt.show()

def accplot(array,fig_name):
    plt.title('Result Analysis')
    plt.plot(array,label='accuracy')
    plt.legend() # 显示图例
    plt.xlabel('iteration times')
    plt.ylabel('rate')
    plt.savefig(fig_name+'.png',dpi=512)   
    plt.show()

    
def figscat(samples1,samples2,fig_name,types=10,y_=None):
    font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 10,
         }
    markerlist=['.','o','+','*']
    clist=['r','b','lightskyblue','lemonchiffon']
    
    labels=['Right_predict','Wrong_predict']
    
    x1=list(range(0,len(samples1)))
    x2=list(range(0,len(samples2)))
#    yticks=list(range(types))
    xticks=samples1.shape[0]
    yticks=['normal','out02','in02','ball02','out04','in04','ball04','out06','in06','ball06']

    plt.scatter(x1,samples1,marker=markerlist[0],c='',edgecolor=clist[0],label=labels[0])
    plt.scatter(x2,samples2,marker=markerlist[1],c='',edgecolor=clist[1],label=labels[1])

    plt.xticks([x for x in range(len(samples1)+1) if x%(len(samples1)/types)==0])
    plt.yticks([y for y in range(types)],yticks)
    plt.title(fig_name)
    plt.legend() # 显示图例
#        plt.xlabel('iteration times')
#        plt.ylabel('rate')
    plt.xlabel('sample', font1)
    plt.ylabel('faulttype', font1)
    plt.grid(True,linestyle='-')
    plt.savefig(fig_name+'.png',dpi=512)
    plt.show()


#没用到，考虑改进
def savesignal(G,ft,labels,path):
    font1 = {'family': 'Times New Roman',
     'weight': 'normal',
     'size': 3,
     }
    
    lenth=len(G)
    G=G.reshape(lenth,-1)
    ft=10
    each_sample=lenth//ft
    
    for i in range(ft):
#        plt.subplot(5,2,i+1%2)       
        plt.plot(G[i*each_sample+1][0:1024])
        plt.tick_params(labelsize=5)
        plt.title(labels[i])
        plt.savefig(path+labels[i]+'.png',dpi=512)    
        plt.show()   
        
    

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            
            nn.init.xavier_normal_(m.weight, gain=1)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight, gain=1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1)
            m.bias.data.zero_()