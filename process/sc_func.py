import os
import csv
import shutil
import pandas as pd
import numpy as np
import pickle

Time_L =50000
stage = 0
func = 20

#得到用户，数据文件目录
def get_file_list(file_path):
    file_list = os.listdir(file_path)
    if not file_list:
        return
    else:
        print(len(file_list))
        return file_list


#创建文件夹
def mkdir(path0):
    folder = os.path.exists(path0)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path0)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print(path0)
        print("---  OK  ---")
    else:
        shutil.rmtree(path0)
        os.makedirs(path0) 
        print ("---  There is this folder!  ---")
        print(path0)


#生成矩阵  stage1：7-80 stage2：4-77
def to_numpy(loc,new_data):
    #构建100*88的dataframe：
    lnumpy=nu= pd.DataFrame(np.zeros((88),int),index= list(range(stage,stage+88)))
    #100行
    for i in range(100):
        chunk=new_data[(new_data[0]>=loc+i*Time_L)&(new_data[0]<loc+(i+1)*Time_L)]
        #print(chunk)
        if(len(chunk)):
            #print(chunk)
            ck=chunk[2].value_counts()
            #if(len(ck)):
                #print(ck)
            ck.name=loc+i*Time_L
            ck1=ck[(stage<=ck.index)&(ck.index<stage+88)]
            ck2=ck1.sort_index()
            #if(len(ck2)):
             #   print(ck2)
            lnumpy = pd.concat([lnumpy,ck1], axis=1)
        else:
            #nu.rename(columns={0:loc+i*Time_L})
            lnumpy = pd.concat([lnumpy,nu.rename(columns={0:loc+i*Time_L})], axis=1)
    numpy0 = pd.DataFrame(lnumpy.values.T, index=lnumpy.columns, columns=lnumpy.index)
    n_numpy=numpy0.drop(0).fillna(0).astype(int)
    return n_numpy



#从sc_thre提取信息
def get_chunk(item,path,upath1,dpath):
    dpath1=dpath +'\\'+ item[:-3]+'_func'
    mkdir(dpath1)
    path1 = path+'\\'+item[:-3]+'_sc_thre.csv'
    sc_data = pd.read_csv(path1,header=None,encoding='gbk') # 0040308099_sc_thre.csv
    user_apps=get_file_list(upath1)
    j=0
    for user_app in user_apps:
        upath2 = upath1+'\\'+user_app
        app = user_app[11:-7]
        dpath2 = dpath1 +'\\'+item[:-3]+'_'+app+'_func'
        mkdir(dpath2)
        scapp_data = pd.read_csv(upath2,header=None,encoding='gbk') # 0040308099_android_ts.csv
        k=0
        print(str(j)+"："+item[:-3]+"："+app)
        pkl_path=dpath2+'\\'+item[:-3]+'_'+app+'.pkl'
        fp = open(pkl_path, 'wb')
        for timestamp in scapp_data[0]:
            print(str(j)+"："+str(k))
            a=sc_data[(sc_data[0]>=timestamp)&(sc_data[2]==func)][:1].index.tolist()
            if(a):
                b=sc_data.iloc[a[0]][0]
                c=sc_data[(sc_data[0]>=b-1000000)&(sc_data[0]<(b+4000000))].tail(1).index.tolist()
                #################得到五秒钟数据
                new_data=sc_data.iloc[a[0]:c[0]+1]
                #print(new_data[2].value_counts)
                #################生成矩阵
                n_numpy=to_numpy(b,new_data)
                #print(n_numpy)
                #################写入pkl
                if(len(n_numpy)):
                    pickle.dump(n_numpy, fp)
                    k=k+1
                else:
                    with open('1error'+item[:-3]+'.txt','a+') as f:
                        f.write(item[:-3]+'_'+app+'_func'+'_'+str(k)+"'s csv is NULL")
                    f.close()
                #################写入csv
                # if(len(new_data)):
                #     rcsv_path=dpath2+'\\'+item[:-3]+'_'+app+'_'+str(i)+'.csv'
                #     if not os.path.exists(rcsv_path):
                #         new_data.to_csv(rcsv_path,index=False,header=False)
                #     else:
                #         new_data.to_csv(rcsv_path,mode='a',index=False,header=False)
                #     i=i+1
                # else:
                #     with open('1error'+item[:-3]+'.txt','a+') as f:
                #         f.write(item[:-3]+'_'+app+'_func'+'_'+str(i)+"'s csv is NULL")
                #     f.close()
                
            else:
                with open('error'+item[:-3]+'.txt','a+') as f:
                    f.write(item[:-3]+'_'+app+'_func'+'_'+str(k)+"'s csv is NULL")
                f.close()
        fp.close()  
        j=j+1


    

def main():
    
    path=r'C:\Users\20180525\Desktop\compiler\data1\sc_thre'
    dpath=r'C:\Users\20180525\Desktop\compiler\data1\sc_func_chunk'
    upath=r'C:\Users\20180525\Desktop\compiler\data1\fa_time_extra'
    user_list=get_file_list(upath)
    print(user_list)
    for item in user_list:
        print("item："+item)
        upath1 = upath+'\\'+item
        get_chunk(item,path,upath1,dpath)

    
if __name__ == '__main__':
    main()    
