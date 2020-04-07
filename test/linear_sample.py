# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:10:37 2020

@author: HIT
"""
import numpy as np

np.random.seed(7)
m = 50
X = 2 * np.random.rand(m, 1)
y = (4+3*X+np.random.randn(m,1)).ravel()
#%%
from sklearn.svm import LinearSVR

svm_reg=LinearSVR(epsilon=1.5, random_state=7)
svm_reg.fit(X,y)
#%%
y_pred = svm_reg.predict(X)
#%%
y_pred -y
#%%

#%%
# Linear Regression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR

# 클래스 변수
Linear_model = ["Lasso", "ElasticNet", "Linear Regression", "Ridge"]
NonLinear_model = ['SVR']

# 메모리 내 로드 된 데이터
trainset_X, trainset_Y = train[test_colname].values, train['normal_y'].values
testset_X = testset[test_colname].values
# 모델 탐색
# Linear Models
for models in cg(Linear_model) :
    print("Predictions... Algorithm : {}".format(Linear_model[models]))
    try : 
        if 'Lasso' == Linear_model[models] : 
            model = Lasso(alpha=0.5, random_state=seed, normalize=False)
        elif 'ElasticNet' == Linear_model[models] : 
            model = ElasticNet(alpha=1.0, random_state=seed, normalize=False)
        elif 'Linear Regression' == Linear_model[models] : 
            model = Ridge(alpha=1.0, random_state=seed, normalize=False)
        elif 'Ridge' == Linear_model[models] : 
            model = LinearRegression(normalize=False)
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

pred['normal_y'] = test['normal_y']
#%%
# sell_qty 복귀
# normalize된 데이터를 sell_qty로 복귀(역산)
# train의 mean_value, std_value로 de-normalization
after_process = pd.concat([test, pred], axis=1)
# 에러 처리
if pred.iloc[:,0].isna().any() == True :
    print('t')
after_process['denormal_y'] = ((after_process['y'] * after_process['std_value']) + after_process['mean_value'])

#%%
for i in cg(pred.columns) :
    print("{} accuracy : {}".format(pred.columns[i], 1-abs(np.mean((abs(pred.iloc[:, i]) - abs(pred['y'])) / abs(pred['y'])))))


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
test

















