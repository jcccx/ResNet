# -*- coding: utf-8 -*-
# coding: utf-8
import os
import csv
import shutil
import pandas as pd
from itertools import groupby

#得到用户，数据文件目录
def get_file_list(file_path):
    file_list = os.listdir(file_path)
    if not file_list:
        return
    else:
        #dir_list = sorted(file_list,  key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        print(len(file_list))
        return file_list

#文件夹
def mkdir(upath):
    folder = os.path.exists(upath)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(upath)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        shutil.rmtree(upath)
        os.makedirs(upath) 
        print ("---  There is this folder!  ---")

#提取时间点
def time_stamp(ulist,path,dpath):
    for item in ulist:
        app_list=[]
        upath = dpath +'\\'+item[:10]+"_fa"
        mkdir(upath)
        csv_path=path+'\\'+item[:10]+"_fa.csv"
        data = pd.read_csv(csv_path,header=None,encoding='gbk')
        app_list=data[1].value_counts().index.tolist()
        print(len(app_list))
        for app in app_list:
            print(app)
            if(app!="None"):
                ser_num = data[(data[1]==app)].index.tolist()
                #print(ser_num)
                timestamp=[]
                fun = lambda x: x[1]-x[0]
                for k, g in groupby(enumerate(ser_num), fun):
                    l1 = [j for i, j in g]    # 连续数字的列表
                    timestamp.append(min(l1))   
                timestamp.sort()
                #print(timestamp)
                ts=data.iloc[timestamp]
                print(ts)
                if(len(ts)):
                    rcsv_path=upath+'\\'+item[:11]+app+"_ts.csv"
                    if not os.path.exists(rcsv_path):
                        ts.to_csv(rcsv_path,index=False,header=False,encoding='gbk')
                    else:
                        ts.to_csv(rcsv_path,mode='a',index=False,header=False,encoding='gbk')
                else:
                    with open('error.txt','a+') as f:
                        f.write(item+app+"'s csv is NULL")
                    f.close()
            else:
                print("-----------------None----------------")
                    
        
    
def main():
    path0=r'C:\Users\20180525\Desktop\compiler\data\sc_thre'
    path=r'C:\Users\20180525\Desktop\compiler\data\fa_source'
    dpath=r'C:\Users\20180525\Desktop\compiler\data\fa_time_extra'
    user_list=get_file_list(path0)
    print(user_list)
    time_stamp(user_list,path,dpath)


    
if __name__ == '__main__':
    main()
