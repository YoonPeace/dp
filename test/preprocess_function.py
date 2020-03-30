# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:22:59 2020

@author: HIT
"""

path = 'C:\\Users\\HIT\\Desktop\\Default_Path\\2020년 [SaaS] Dynamic Pricing\\source\\data'
import os
import pandas as pd
import numpy as np
import re
from statsmodels.formula.api import ols

df = pd.read_csv(path+'\\'+os.listdir(path)[0], encoding='utf-8')
print(df.shape)
#%%
config ="C:\\Users\\HIT\\Desktop\\Default_Path\\2020년 [SaaS] Dynamic Pricing\\source\\data\\export"
#%%
os.listdir(path)
#%%
# create Generator
def cg(lists) :
    iter_range = range(len(lists))
    for i in iter_range :
        yield i

#%%

# train, data set 분리
# function split_train_and_test_set : train-test 데이터를 분리시킨다.
# df : 데이터 셋
# x : 기준컬럼
# rate : train, test, validate data 비율(list)
# 튜닝 대상
from sklearn.model_selection import train_test_split

def split_train_and_test_set(df, x, rate) :
    # x 기준 정렬    
    try :
        if len(rate) == 1 :
            train, test = train_test_split(df, test_size=rate[0])
    
            return train, test
        elif len(rate) >= 3 :
            probs = np.random.rapnd(len(df))
            train_mask = probs < rate[0]
            test_mask = (probs >= rate[0]) & (probs < rate[0] + rate[1])
            validation_mask = probs >= 1 - rate[2]
            
            train, test, validation = df[train_mask], df[test_mask], df[validation_mask]
            return train, test, validation
    except Exception as e :
        print(e)

#### ex 
train, test = split_train_and_test_set(df, 'datetime', [0.2])
#%%    
# Standardization 1

## function stats_dataframe : 특정 컬럼 기준- y의 mean, stddev 그룹핑을 수행한다
## parameter
### df : 데이터 셋
### y = 예측하고자 하는 값의 컬럼명
### x = 기준 컬럼
### methond = 원하는 방법
def stats_dataframe(df, x, y, method) :
    if method == 'mean' :
        colname = [x,'mean_value']
        target_dataframe = pd.DataFrame(df.groupby([x])[y].mean().reset_index())
    elif method == 'stddev' :
        colname = [x,'std_value']
        target_dataframe = pd.DataFrame(df.groupby([x])[y].std().reset_index())     
    
    target_dataframe.columns = colname
    target_dataframe = pd.merge(df, target_dataframe, on = x)
    
    return target_dataframe

#%%
# 상수는 나중에 변경
# 상수
x_var = ['tot_disc', 'dow', 'hr']
subset_idx = 'prod_cd'
variable_y = 'sell_qty'
iterate = ['mean','stddev']

for i in range(len(iterate)) :
    train = stats_dataframe(train, subset_idx, variable_y, iterate[i])

#%%    
# Standardization 2

## function merge_dataframe : test - train 데이터 표준편차, mean 값으로 scaling
## parameter
### df : 데이터 셋
### y = 종속 변수
def merge_dataframe(df, y) :
    df['normal_y'] = (df[y] - df['mean_value']) / df['std_value']        
    return df

#example
train = merge_dataframe(train, variable_y)
#%%
# OLS
## function value_store : 데이터셋 내 y와 독립변수들 간의 계수와 p_value를 뽑아 낸다
## parameter
### df : 데이터셋
### x : 서브셋 기준 독립변수
### x_var : 데이터셋
### normal_y : 표준화 된 y 
### y : 종속변수 y

def value_store (df, x, x_var, normal_y, y) :
    min_variables, variables_name, base_lists = [], [], []
    
    # parameter coef, parameter coef
    param_coef, param_pval = [], []
    formula = y + ' ~ '
    formula_set = []
    seq = 0
    
    for i in cg(set(df[x])) :
    # for i in range(2) :
        # subset 지정(prod_cd(base variable) 별 subset)
        seq += 1
        formula = y + ' ~ '
        # print('iteration formula %s' % formula) # 
        subset = df[df[x] == np.unique(df[x])[i]].iloc[:, 1:]
        base = np.unique(df[x])[i]
        for j in cg(x_var) :
            variables_name.append(x_var[j])
            min_variables.append(np.argmin(subset.groupby([x_var[j]])[normal_y].mean()))
            #np.argmin(df[df['prod_cd'] == np.unique(df['prod_cd'])[0]].iloc[:, 1:].groupby('tot_disc')['normal_y'].min()) <- 최소값 기준
            # return value 
            # x_var[i],x_var[i], x_var[i]
            # value,   value,    value
        
            # 공식 만들기
            # 독립변수 개수 만큼 creation
            if i == 0 :
                form = "C(" + x_var[j] + ", Treatment(" + str(min_variables[j]) + "))"
            else : 
                form = "C(" + x_var[j] + ", Treatment(" + str(min_variables[len(x_var) * i + j]) + "))"            
            for k in cg(x_var) :
                if j == k :
                    if k == (len(x_var)-1) :
                        formula += form
                    else : 
                        formula += form + " + "
        result = ols(formula, data=subset).fit()
        base_lists.append(base)
        param_coef.append([result.params])
        param_pval.append([result.pvalues])
        formula_set.append(formula)
    return min_variables, variables_name, base_lists, param_coef, param_pval, formula_set

min_variables, variables_name, base_lists, param_coef, param_pval, formula_set = value_store(train, subset_idx, x_var, 'normal_y', variable_y)
#%%
# function coef_extract
## 독립변수의 각 계수 산출
# min_idx : minimum coefficient, params : 변수 내 도메인, val : 계수 값, base : 독립변수 명
def coef_extract(min_idx, params_df, pval_df, x_var, seq) :
    # parameter_coef
    # idx = []
    # pval = pval_df[0].values[pd.Series(pval_df[0].index).apply(lambda x: x.split(')')[0][2:2+len(base)]) == base]
    # coef = params_df[0].values[pd.Series(params_df[0].index).apply(lambda x: x.split(')')[0][2:2+len(base)]) == base]
    text = str(params_df)
    name_p = re.compile("T\.\d+\.+\d+|T\.\d+")
    name_m = name_p.findall(text)
    names = pd.Series(name_m).apply(lambda x :x.replace('T.','')).values
    
    do_p = re.compile("C\(+\S+")
    do_m = do_p.findall(text)
    domain = pd.Series(do_m).apply(lambda x : x.replace(",","").replace("C(","")).values
    
    baseline_p = re.compile("Treatment\(\d+\.\d+|Treatment\(\d+")
    baseline_m = baseline_p.findall(text)
    baseline = pd.Series(baseline_m).apply(lambda x : x.replace("Treatment(", "")).values

    coef = np.array(params_df)
    pval = np.array(pval_df)
    
    dataframe = pd.DataFrame([names, domain, np.ravel(coef, order='C')[1:], np.ravel(pval, order='C')[1:], baseline]).T
    dataframe.columns = ['key1', 'key2', 'coef', 'pvalue', 'baseline']
    
    # for point in range(1, len(params)-1):
        # idx.append(point)
        # if point == min_idx:
        #    idx.pop(point)
    # return coef, pval, names, domain #, idx
    return dataframe
#%%
## function columns_name : value_store로 산출된 데이터를 데이터프레임화 한다.
## parameter
### x_var : 독립변수 집합
### base_lists : 서브셋 독립변수 기준
### rows : iteration 변수
    
# def columns_name(base_lists, x_var, row) :
    # base_lists_name = pd.Series(base_lists).apply(lambda x : str(x) + '_' + x_var[row])
    # base_lists_name = pd.Series(base_lists).apply(lambda x : str(x))
    # return list(base_lists_name)
#%%    
## function to_df_process : value_store로 산출된 데이터를 데이터프레임화 한다.
## parameter
### x_var : 데이터셋
### min_variables : 독립변수의 평균값 집합
### param_coef : 독립변수의 계수 집합
### subset_idx : 서브셋 기준 독립변수 리스트
# parameter 1번째의 서브셋 array위치 0 * len(base_lists)
# parameter 2번째의 서브셋 array위치 1 * len(base_lists)
# parameter 3번째의 서브셋 array위치 2 * len(base_lists)

def to_df_process(x_var, min_variables, param_coef, param_pval, subset_idx) :
    num = len(x_var)
    # coef, names, pval, domain, base_col = [], [], [], [], []
    # 서브셋 개수만큼 iteration
    print('===============================start parse data =================================')
    for seq in cg(subset_idx) :
    # print('iteration %i' % (seq + 1))
        if (seq == 0) or (len(subset_idx) == 0) :
            extract_dataframe = coef_extract(min_variables[num*seq:(num*seq)+num], param_coef[seq], param_pval[seq], x_var, seq)
            extract_dataframe['subset_idx'] = str(subset_idx[seq])
        else :
            temp = coef_extract(min_variables[num*seq:(num*seq)+num], param_coef[seq], param_pval[seq], x_var, seq)
            temp['subset_idx'] = subset_idx[seq]
            extract_dataframe = pd.concat([extract_dataframe, temp], axis=0)
        #names.append(coef_extract(min_variables[num*seq:(num*seq)+num], param_coef[seq], x_var[x], seq)[2])

    print('==============================parse data job end ================================')        
        
    return extract_dataframe
            
#    return print(subset_idx)
# 0 : coefficients, 1: pvalues, 2 : names, 3:domains
sets = to_df_process(x_var, min_variables, param_coef, param_pval, base_lists)
#%%
# function name: export_to_file
# 중간 결과 파일을 저장한다.
# parameters
# dataframe : 보낼 데이터프레임
# path : 떨굴 장소

def export_to_file(dataframe, path) :
    dataframe.to_csv(path + '\\20200324temp.csv', encoding='utf-8', index=False)

export_to_file(sets, config)
#%%
# functions concat_dataframe 
# origin 데이터프레임의 컬럼과 데이터타입을 맞춘다.
def concat_dataframe(df, sets, x_var) :
    for i in cg(x_var) :
    #for i in range(2) :
        key = x_var[i]
        temp = sets[sets.key2 == key]
        temp.columns = x_var[i] + "_" + sets.columns
        temp.rename(columns={x_var[i] + "_" + "key1" : key, x_var[i] + "_" + "subset_idx" : subset_idx}, inplace=True)
        print(temp.columns)
        # 조인 전 데이터타입 처리
        # key1<-key2를 참조, subset_idx
        temp_join1, temp_join2 = temp[key], temp[subset_idx]
        origin_join1, origin_join2 = train[key], train[subset_idx]
        
        def datatype_chk(origin_col, temp_col) :
            if origin_col.dtype != temp_col.dtype :
                cd = origin_col.dtype
                temp_col = temp_col.astype(cd)
                return temp_col
            
        temp[key] = datatype_chk(origin_join1, temp_join1)
        temp[subset_idx] = datatype_chk(origin_join2, temp_join2)
        
        # 독립변수의 계수 p_value, 병합
        if i == 0 :
            merge_dataframe = pd.merge(df, temp, left_on=[key, subset_idx], right_on = [key, subset_idx], how='left')
        else :
            merge_dataframe = pd.merge(merge_dataframe, temp, left_on=[key, subset_idx], right_on = [key, subset_idx], how='left')
    return merge_dataframe
    
#%%
# functions after_job 
# 데이터 후처리를 수행 한다. 데이터프레임의 컬럼과 데이터타입을 맞춘다.

def after_job(rate=0.05):
    df = concat_dataframe(train, sets, x_var)
    
    # 내부 데이터 정의
    coef, _pvalues, _key2 = [], [], []
    for i in cg(df.columns) :
        if '_pvalue' in df.columns[i] :
            _pvalues.append(df.columns[i])
            
        if '_coef' in df.columns[i] :
            coef.append(df.columns[i])
        
        if '_key2' in df.columns[i] :
            _key2.append(df.columns[i])
            
    # pvalue가 1(가설을 충족하는 데이터 0%)로 가정  
    # pvalue 사실 여부 확인 
    for j in cg(_pvalues) :
        tempo, coef_chk = df[_pvalues[j]], df[coef[j]]
        # pvalue를 확인하고 rate 이상인 계수는 1로 치환            
        bool_map = tempo[abs(df[_pvalues[j]]) > rate] = 1
        coef_chk[tempo == 1] = 1
        
        df[coef[j]] = coef_chk

    # NaN 값 처리
    # NaN 값 1로 채우기
    df[coef], df[_pvalues] = df[coef].fillna(1), df[_pvalues].fillna(1)
    
    # 필요하지 않는 데이터 삭제(key2)
    df.drop(_key2, axis=1, inplace=True)
    return df, coef
df, coef = after_job(0.05)
#%%

# y에 영향을 주는 각각의 독립변수의 영향력을 계산한다.
def estimate_individual_power(df, x_var, coef):
    df['all_individual_power'] = df['normal_y']
    for i in cg(coef) :
        baseline = x_var[i] + '_' + 'individual_power'
        df[baseline] = df['normal_y'] / df[coef[i]]            
        df['all_individual_power'] = df['all_individual_power'] / df[coef[i]]
     
    return df

df = estimate_individual_power(df, x_var, coef)
#%%