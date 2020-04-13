# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:10:37 2020

@author: HIT
"""
# Linear Regression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
import numpy as np
import math

# 클래스 변수
Linear_model = ["Lasso", "ElasticNet", "Linear Regression", "Ridge"]
NonLinear_model = ['SVR']
#%%
valset
#%%
# 모델 탐색
# Linear Models
# subset 만큼 반복
for subset in range(1) :
    
    
#for subset in cg(base_lists) :

    # 메모리 내 로드 된 데이터
    tr_set = train[train[subset_idx] == base_lists[subset]]
    te_set = valset[valset[subset_idx] == base_lists[subset]]
    trainset_X, trainset_Y = tr_set.loc[:,test_colname].values, tr_set['normal_y'].values
    testset_X = te_set[test_colname].values
    
    for models in cg(Linear_model) :
        print("Predictions... Algorithm : {}".format(Linear_model[models]))
        try : 
            if 'Lasso' == Linear_model[models] : 
                model = Lasso(alpha=0.5, random_state=seed, normalize=False)
            elif 'ElasticNet' == Linear_model[models] : 
                model = ElasticNet(alpha=1.0, random_state=seed, normalize=False)
            elif 'Linear Regression' == Linear_model[models] : 
                model = Ridge(alpha=1.0, random_state=seed, normalize=False)
#            elif 'Ridge' == Linear_model[models] : 
#                model = LinearRegression(nor|malize=False)
    #        elif 'LGR' == Linear_model[models] :    
    #            model = LogisticRegression(random_state=seed)
                
            # 이외 전처리 로직 수행<수정 대상>
            def turning(data) :
                print(data)
        # 모형 fitting 
            model.fit(trainset_X, trainset_Y)
            predictions = model.predict(testset_X)
            if models == 0 :
                pred = pd.DataFrame({Linear_model[models] + '_predictions' : predictions})
            else :
                pred[Linear_model[models]  + '_predictions'] = predictions
        except Exception as e :
            print(e)

#%%
for i in range(len(iterate)) :
    validate = stats_dataframe(validate, subset_idx, variable_y, iterate[i])
validate = merge_dataframe(validate, variable_y)
#%%
y = validate[validate.prod_cd == 2040265616].normal_y.values
pred['y'] = y
#%%
#%%
import matplotlib.pyplot as plt 

%matplotlib inline
plt.rcParams["figure.figsize"] = (14,7)

for i in range(1,5,1) : 
    plt.subplot(2,2,i)
    plt.title(pred.columns[i-1])
    plt.plot(pred[['y',pred.columns[i-1]]])
plt.show()


#%%
#%%
# sell_qty 복귀
# normalize된 데이터를 sell_qty로 복귀(역산)
# train의 mean_value, std_value로 de-normalization
after_process = pd.concat([test, pred], axis=1)
#%%
after_process['denormal_y'] = (after_process['Lasso_predictions'] * after_process['stddev_value']) + after_process['mean_value']
#%%


#%%
# 에러 처리
# if pred.iloc[:,0].isna().any() == True :
for i in cg(pred.columns) :
    print("{} MAPE accuracy : {}".format(pred.columns[i], 1-abs(np.mean((abs(pred.iloc[:, i]) - abs(pred['y'])) / abs(pred['y'])))))
    print("{} RMSE accuracy : {}".format(pred.columns[i], (math.sqrt(abs(round(pred.iloc[:, i] - (pred['y']**2)).sum())) * -1) / len(pred['y'])))
    after_process['denormal_y'] = (after_process[pred.columns[i]] * after_process['stddev_value']) + after_process['mean_value']
    after_process = after_process[after_process.denormal_y.isna() == False]
    after_process['denormal_y'] = after_process['denormal_y'].astype(int)
    print(after_process)
    print(int(pd.Series(abs(after_process['sell_qty'].values - after_process['denormal_y'].values)/after_process['denormal_y'].values/len(after_process['sell_qty'].values)*100).sum()))

#%%
#%%
after_process['denormal_y'] = (after_process[pred.columns[0]] * after_process['stddev_value']) + after_process['mean_value']

after_process.denormal_y
#%%
sum(abs(pred.iloc[:, 0]) - abs(pred['y']**2))

#%%^    print("{} MAE accuracy : {}".format(pred.columns[i], 1-abs(np.mean((abs(pred.iloc[:, i]) - abs(pred['y'])) / abs(pred['y'])))))
    print (j,np.unique(result.prod_cd)[j], "RMSE : ",math.sqrt(sum((pred.iloc[:, i] - pred['y']**2/ len(pred['y']))))
    print( j,np.unique(result.prod_cd)[j], "MAE : ", sum(abs(calc_l['with_disc_real_pred'] - pred['sell_qty']))/len(calc_l['sell_qty']))


#%%
trainset.iloc[:,-1].values * 100
#%%
train['sell_qty']
train[x_var + [subset_idx] + test_colname]
#%%
# sell_qty 복귀
# normalize된 데이터를 sell_qty로 복귀
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



#%%

#%%
after_process[after_process.prod_cd == 2050605082].loc[:,['denormal_y', 'sell_qty']].plot()
#%%after_process.sell_qty


# functions concat_dataframe 
# origin 데이터프레임의 컬럼과 데이터타입을 맞춘다.
def concat_dataframe(df, sets, x_var) :
    # for i in cg(x_var) :
    for i in range(1) :
        key = 'mm'
        temp = sets[sets.key2 == key]
        temp.columns = 'mm' + "_" + sets.columns
        temp.rename(columns={'mm' + "_" + "key1" : key, 'mm' + "_" + "subset_idx" : subset_idx}, inplace=True)
        
        
        t = temp
        print(temp)
        print(temp.columns)
        # 여기서 부터 변경
        # 조인 전 데이터타입 처리
        # key1<-key2를 참조, subset_idx

        # temp_join : 독립변수별 서브셋, subset_idx 서브셋
        # origin_join : train의 독립변수 서브셋, subset_idx 서브셋

        temp_join = pd.DataFrame([temp[key].values, temp[subset_idx].values]).T
        origin_join = pd.merge(train[key].reset_index(), train[subset_idx].reset_index(), how='inner', on='index')        
        if temp_join[key].isna().any() == True :
            print(1)
                
        
        def datatype_chk(origin_col, temp_col) :
            if origin_col.dtype != temp_col.dtype :
                cd = origin_col.dtype
                temp_col = temp_col.astype(cd)
                return temp_col
            
        temp[key] = datatype_chk(origin_join[key], temp_join[key])
        temp[subset_idx] = datatype_chk(origin_join[subset_idx], temp_join[subset_idx])
        
        # 독립변수의 계수 p_value, 병합
        if i == 0 :
            merge_dataframe = pd.merge(df, temp, how='left', on=[key, subset_idx])
#            merge_dataframe = pd.merge(df, temp, left_on=[key, subset_idx], right_on = [key, subset_idx], how='left', on=[key, subset_idx])
        else :
            merge_dataframe = pd.merge(merge_dataframe, temp, how='left', on=[key, subset_idx])
    return merge_dataframe, temp, origin_join, temp, temp[key].reset_index(), temp[subset_idx]
#%%
t, temp, origin, temp_join, temp_key, temp_subset_idx = concat_dataframe(train, sets, x_var)
#%%
temp_join
#%%
pd.Dataframe(temp_key,temp_subset_idx)
#%%
temp_key.reset_index()['index'].unique()
#%%
temp_subset_idx.values
#%%
sets[sets.key2 == 'mm'].key1.values
#%%
def refact(df, sets, x_var) :
    for i in range(1) :
        key = 'mm'
        temp = sets[sets.key2 == key]
        temp.columns = 'mm' + "_" + sets.columns
        temp.rename(columns={'mm' + "_" + "key1" : key, 'mm' + "_" + "subset_idx" : subset_idx}, inplace=True)
        

        
        #temp_join = temp.drop(columns=['mm_key2']).reset_index().iloc[:,1:]
        temp_join = temp.reset_index().iloc[:,1:]
            
        #temp_join = pd.DataFrame(temp[key].values, temp[subset_idx].values).reset_index()
        
        # 검증을 위한 데이터타입 체크        
        def datatype_chk(origin_col, temp_col) :
            if origin_col.dtype != temp_col.dtype :
                cd = origin_col.dtype
                temp_col = temp_col.astype(cd)
                return temp_col
            
        temp_join[key] = datatype_chk(df[key], temp_join[key])
        temp_join[subset_idx] = datatype_chk(df[subset_idx], temp_join[subset_idx])

        if key+'_coef'
        # 독립변수의 계수 p_value, 병합
        if i == 0 :
            merge_dataframe = pd.merge(df, temp_join, left_on=[key, subset_idx], right_on = [key, subset_idx], how='left')
        else :
            merge_dataframe = pd.merge(merge_dataframe, temp_join, left_on=[key, subset_idx], right_on = [key, subset_idx], how='left')

    return merge_dataframe, temp_join

d, tj = refact(train, sets, x_var)
#%%
#%%
# 각 독립변수의 영향력을 concat 입력
def testset_process(validate, train, x_var, test_colname, subset_idx):
    # x 조인 및 학습 데이터 셋의 컬럼, temp_col = null값인 데이터
    x, tcname, bid, ts = [subset_idx] + x_var, None, None, None
    
    # validate 독립변수 지정
    valset = validate[x]
    # 계수 집계 시 median 값으로 처리
    trainset = train[x + test_colname + coef + iterate].groupby(x).median().reset_index()
    # valset 기준 병합
    valset = pd.merge(valset, trainset, left_on=x, right_on=x, how='outer')
    trainset['normal_y'] = train['normal_y']
    # 결측치 처리 
    
    # outer join으로 있는 요소들만 필터링
    #for null in cg(valset.columns) :
    #    if valset.iloc[:,null].isna().any() == True :
    #        valset.iloc[:, null] = valset.iloc[:, null].fillna(1)
    
    for i in cg(valset.columns) : 
        if valset.iloc[:, i].isna().any() == True :
            # 컬럼의 자릿수 입력
            if tcname == None :
                # 처음 1회만 실행
                var: ContextVar('None', default=42)
                ts = valset[valset.loc[:, valset.columns[i]].isna() == False]
                print(var.name)
            else :
                tcname = valset.columns[i]
                var: ContextVar(tcname, default=42)
                print(var.name)
                bid = pd.concat(ts, valset[valset.loc[:, valset.columns[i]].isna() ==False], axis=0)
                            
            #ffffff
            
    return valset
#%%    
valset = testset_process(validate, train, x_var, test_colname, subset_idx)
 #%%
from contextvars import ContextVar, Token
from contextvars import copy_context

t = []
for i in range(len(valset.columns)) : 
    if valset.iloc[:, i].isna().any() == True :
        t = valset.columns[i]

    var : ContextVar[str] = ContextVar(valset.columns[i], default=42)
    ctx: t = copy_context()
    print(type(t))

#%%
t = None
if t == None :
    var = ContextVar('None')
token = var.set('new value')
# code that uses 'var'; var.get() returns 'new value'.

token.var.name

#%%
tcname = 1
#%%
var: ContextVar[int] = ContextVar('var', default=42)
token = var.set('new value')
#%%
token.old_value
#%%
from contextvars import copy_context

ctx: tcname = copy_context()
#%%
print(list(ctx.items()))
#%%
var
#%%
var : ContextVar[str] = ContextVar(None, default=42)
