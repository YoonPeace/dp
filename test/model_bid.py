# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:59:56 2020

@author: HIT
"""
#%%
# 3개 모델을 구현하고 경쟁 시킨다.
#%%
#####import libraries
import pandas as pd
import numpy as np
from sklearn import svm
import math
#%%
# 상수 정의
# function cg() = preprocess 참고
subset_idx = subset_idx # base column name
variable_y = variable_y # target y column name
base_lists = base_lists # base column unique value
#%%
# ML 알고리즘
# 선형/비선형 알고리즘으로 각각의 데이터 예측

# 직관에 의한 로직을 다른 곳에서 구성

#%%
# global f 
# f

#%%

class Machine :
    
    def __init__(self, train, test, base_lists, subset_idx, variable_y) :
        
        self._base_lists      = base_lists
        self._subset_idx      = subset_idx
        self._variable_y      = variable_y
        self._subset_idx_num  = None # 외부에서 사용중
        self._coef_num        = None # 외부에서 사용중
        self._variable_y_num  = None # 외부에서 사용중
        self._train           = train
        self._test            = test
        self._non_linear      = ['SVR'] # 비선형 알고리즘 <- SVR 먼저 구성 <- 나중에 configuration에서 가져오기
        self._linear          = []
        
    # create Generator
    # 나중에 preprocess 패키지에서 import 현재는 정의하는 것으로
    def cg(lists) :
        iter_range = range(len(lists))
        for i in iter_range :
            yield i
    

    # 데이터프레임의 x, y 타겟 컬럼의 자리수를 찾는다.     
    def find_x_y(self, **kwargs) : # <-- **kwarg 필요없으면 제거 / 지금은 필요없음
        subset_idx_num = []
        coef_num       = []
        variable_y_num = []
        
            
        for i in cg(df.columns) :
            if '_coef' in df.columns[i] :
                coef_num.append(i)
            if self._subset_idx in df.columns[i] :
                subset_idx_num.append(i)
            if self._variable_y in df.columns[i] :
                variable_y_num.append(i)
                
        self._coef_num        = coef_num
        self._subset_idx_num  = subset_idx_num
        self._variable_y_num  = variable_y_num
        
        # __load_data 함수 실행
    def load_data(self, seq=0):
        
        train_docX = self._train[self._train.iloc[:,self._subset_idx_num] == self._base_lists[seq]].iloc[:,self._coef_num].values
        train_docY = self._train[self._train.iloc[:,self._subset_idx_num] == self._base_lists[seq]].iloc[:,self._variable_y_num].values
        test_docX = self._test[self._test.iloc[:,self._subset_idx_num] == self._base_lists[seq]].iloc[:,self._coef_num].values
        test_docY = self._test[self._test.iloc[:,self._subset_idx_num] == self._base_lists[seq]].iloc[:,self._variable_y_num].values
        
        if 'SVR' in self._non_linear :
            print(self._base_lists[seq])
            print(train_docX) 
            if self._base_lists[seq] in self._train.iloc[:,self._subset_idx_num]  :
                print(True)
            else : 
                print(False)
    
            

#%%
t = Machine(df, df, base_lists, subset_idx, variable_y)
t.find_x_y()
t.load_data()
#%%
len(np.unique(train.prod_cd))
#%%
# 모델에 입력시킬 트레인 - test 셋의 x, y 값(count : 4)을 리턴




df[df.iloc[:,0] == 2061827817].iloc[:,t._coef_num].values
#%%
calculate = 0
train_X, train_y = _load_data(df, 0) # 0은 서브셋 기준의 집합이므로 반복의 대상임
test_X, test_y = _load_data(df, 0)   # 0은 서브셋 기준의 집합이므로 반복의 대상임
#return np.array(train_X, train_y), np.array(test_X, test_y)
train_set, test_set = find_x_y(df, df)
#%%
        
    clf = svm.SVR()
    clf.fit(train_X, train_Y) 
    predicted=clf.predict(test_X)

#target_columns
#%%

for i in range(0,len(np.unique(train.prod_cd))):
    train_set = train.loc[train['prod_cd']==np.unique(train.prod_cd)[i]]
    test_set = test.loc[test['prod_cd']==np.unique(train.prod_cd)[i]]

    train_X_set, train_Y_set = _load_data(train_set, n_prev, n_next)
    test_X_set, test_Y_set = _load_data(test_set, n_prev, n_next)
    

    #train_X
    train_X=train_X_set[:,:,25]
    #train_Y
    train_Y=train_Y_set[:,25]
    #test_X
    test_X=test_X_set[:,:,25]
    #test_Y
    test_Y=test_Y_set[:,25]

n_prev, n_next, i =24, 0, 0
#%%
# SVR 모델 구현
# 데이터 로드
#%%

def _load_data(data, n_prev, n_next):  
        docX, docY = [], []
        for i in range(len(data)-n_prev-n_next):
            docX.append(data.iloc[i:i+n_prev].as_matrix())
            docY.append(data.iloc[i+n_prev+n_next].as_matrix())
        alsX = np.array(docX)
        alsY = np.array(docY)
    
        return alsX, alsY
    
# train, test set 필요
# subset 기준으로 예측
# train, test set의 x,y 셋 분리
# x, y 셋을 정의
# x 지정 <- 원 데이터의 컬럼 기준으로 y, x_var, 시계열데이터(인덱스)만 가지고 x를 정의 하여 matrix form으로 변경
        

#%%
train_X_set
#%%
docX, docY = [], []
n_prev=24
n_next=0
for i in range(1):
    train_set = train[train['prod_cd']==2065014763]
    test_set = test[test['prod_cd']==2065014763]
    train_X_set, train_Y_set = _load_data(train_set, n_prev, n_next)
    test_X_set, test_Y_set = _load_data(test_set, n_prev, n_next)

#%%

    train_X_set, train_Y_set = _load_data(train_set, n_prev, n_next)
    test_X_set, test_Y_set = _load_data(test_set, n_prev, n_next)

#%%
train = train
#%%
train
#%%
x[0]

#%%
train
#%%
train = f_data.loc[(f_data['datetime']<40600)]
##Test : 0407~0430
test = f_data.loc[(f_data['datetime']>40523)]

#%%
def _load_data(data, n_prev, n_next):  
        docX, docY = [], []
        for i in range(len(data)-n_prev-n_next):# <-- 98319
            print(len(data)-n_prev-n_next)
            print('start',i, i+n_prev, i+n_prev+n_next)
            #docX.append(data.iloc[i:i+n_prev].as_matrix())
            #docY.append(data.iloc[i+n_prev+n_next].as_matrix())
        alsX = np.array(docX)
        alsY = np.array(docY)
    
        return alsX, alsY
x, y = _load_data(data=train, n_prev=24, n_next=0)
print(x, y)
#%%     
n_prev=24
n_next=0

i=0
#%%
x, y = _load_data(data=train, n_prev=24, n_next=0)
#%%

len(x[2][27])
#%%

train.loc[train['prod_cd']==np.unique(train.prod_cd)[0]]
#%%
predicted_result = []
test_Y_result = []
prod_cd_result = []
date_time_result = []
time_result = []
dow_result = []
tot_disc_result = []
dow_coef_result = []
time_coef_result = []
disc_coef_result = []
mean_result = []
std_result = []
sell_qty_result = []
nor_qty_result = []
both_baseline_result=[]
# 본문
for i in range(0,len(np.unique(train.prod_cd))):
    train_set = train.loc[train['prod_cd']==np.unique(train.prod_cd)[i]]
    test_set = test.loc[test['prod_cd']==np.unique(train.prod_cd)[i]]

    train_X_set, train_Y_set = _load_data(train_set, n_prev, n_next)
    test_X_set, test_Y_set = _load_data(test_set, n_prev, n_next)
    

    #train_X
    train_X=train_X_set[:,:,25]
    #train_Y
    train_Y=train_Y_set[:,25]
    #test_X
    test_X=test_X_set[:,:,25]
    #test_Y
    test_Y=test_Y_set[:,25]
        
    clf = svm.SVR()
    clf.fit(train_X, train_Y) 
    predicted=clf.predict(test_X)
    
    prod_cd_result.extend(test_Y_set[:,0].tolist())
    date_time_result.extend(test_Y_set[:,1].tolist())
    time_result.extend(test_Y_set[:,4].tolist())
    dow_result.extend(test_Y_set[:,5].tolist())
    tot_disc_result.extend(test_Y_set[:,10].tolist())
    dow_coef_result.extend(test_Y_set[:,18].tolist())
    time_coef_result.extend(test_Y_set[:,19].tolist())
    disc_coef_result.extend(test_Y_set[:,20].tolist())
    both_baseline_result.extend(test_Y_set[:,24].tolist())
    mean_result.extend(test_Y_set[:,15].tolist())
    std_result.extend(test_Y_set[:,16].tolist())
    sell_qty_result.extend(test_Y_set[:,13].tolist())
    nor_qty_result.extend(test_Y_set[:,17].tolist())
        
        
    predicted_result.extend(predicted.tolist())
    test_Y_result.extend(test_Y.tolist())
    

#%%
#append list to dataframe at once!

result=pd.DataFrame({"prod_cd":prod_cd_result,
"date_time":date_time_result,
"time":time_result,
"dow":dow_result,
"tot_disc":tot_disc_result,
"dow_coef":dow_coef_result,
"time_coef":time_coef_result,
"disc_coef":disc_coef_result,
"mean":mean_result,
"std":std_result,
"both_baseline" : both_baseline_result,
"sell_qty":sell_qty_result,
"nor_qty":nor_qty_result,
"predicted":predicted_result,
"test_Y":test_Y_result})
    
result['without_disc_sell_qty']= result['sell_qty'] / result['disc_coef']    
    #%%
    
#calc_l = result.loc[result['prod_cd']==np.unique(result.prod_cd)[0]]
    
len(np.unique(result.prod_cd))
#%%
#%%
# 판매량으로 다시 환산하는 부분
df = pd.DataFrame() 

for j in range(0,len(np.unique(result.prod_cd))):
    
    calc_l = result.loc[result['prod_cd']==np.unique(result.prod_cd)[j]]
    calc_l['nor_pred'] = calc_l['predicted'] * calc_l['dow_coef']* calc_l['time_coef'] #* calc_l['disc_coef']
    calc_l['with_disc_nor_pred'] = calc_l['predicted'] * calc_l['dow_coef']* calc_l['time_coef'] * calc_l['disc_coef']
    
    calc_l['without_disc_sell_qty']= round(calc_l['both_baseline'] * calc_l['std'] + calc_l['mean'])
    
    calc_l['with_disc_real_pred'] =round(calc_l['with_disc_nor_pred'] * calc_l['std'] + calc_l['mean'])
    calc_l['real_pred']= round(calc_l['nor_pred'] * calc_l['std'] + calc_l['mean'])
    
    calc_l.loc[calc_l['with_disc_real_pred']<0,['with_disc_real_pred']] = 0.01
    calc_l.loc[calc_l['real_pred']<0,['real_pred']] = 0.01
    
    df = df.append(calc_l,ignore_index=True)
    
    #WITH DISC
    plt.plot(calc_l['sell_qty'])
    plt.plot(calc_l['with_disc_real_pred'])
    plt.title(str(np.unique(result.prod_cd)[j]))
    plt.savefig("D://[면세]//적립금최적화_분석//image//"+str(np.unique(result.prod_cd)[j])+".png")
    
    plt.show()
   
    print (j,np.unique(result.prod_cd)[j], "RMSE : ",math.sqrt(sum((calc_l['with_disc_real_pred'] - calc_l['sell_qty'])**2/len(calc_l['sell_qty']))))
    print( j,np.unique(result.prod_cd)[j], "MAE : ", sum(abs(calc_l['with_disc_real_pred'] - calc_l['sell_qty']))/len(calc_l['sell_qty']))
    print( j,np.unique(result.prod_cd)[j], "MAPE : ", sum(abs(calc_l['sell_qty'] - calc_l['with_disc_real_pred'])/calc_l['sell_qty'])/len(calc_l['sell_qty'])*100  )
    
    
    #WITHOUT DISC
    
    #plt.plot(calc_l['without_disc_sell_qty'])
    #plt.plot(calc_l['real_pred'])
    #plt.show()
    #print (j,np.unique(result.prod_cd)[j], "RMSE : ",math.sqrt(sum((calc_l['real_pred'] - calc_l['without_disc_sell_qty'])**2/len(calc_l['without_disc_sell_qty']))))
    #print( j,np.unique(result.prod_cd)[j], "MAE : ", sum(abs(calc_l['real_pred'] - calc_l['without_disc_sell_qty']))/len(calc_l['without_disc_sell_qty']))
    #print( j,np.unique(result.prod_cd)[j], "MAPE : ", sum(abs(calc_l['without_disc_sell_qty'] - calc_l['real_pred'])/calc_l['without_disc_sell_qty'])/len(calc_l['without_disc_sell_qty'])*100  )
    
    
    
#%%
df.to_csv("D://[면세]//적립금최적화_분석//predicted_result.csv",index=False)
