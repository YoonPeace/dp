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
# import sys
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
#%%
# global 변수
global seed
global x
global subset_idx
global variable_y
global iterate
seed, x, subset_idx, variable_y = 8, 'datetime', 'prod_cd', 'sell_qty'
# 시계열 값이 앙닌 독립 변수
x_var = ['tot_disc']
iterate = ['mean_value','stddev_value']
#%%
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
# train, test 모두 데이터 생성하기 위한 함수
## 2020.04.13
## 2020.04.13
## 2020.04.13
## 2020.04.13
## 2020.04.13
def search_date(df) :
    base = 14
    boolean_statement = df[x].apply(lambda x : len(str(x))).unique()[0] <= base
    
    if  boolean_statement == True :        
        datecol = base - df[x].apply(lambda x : len(str(x))).unique()[0]
        datecol = pd.DatetimeIndex(df[x].astype(str) + str('0' * datecol))

    temp = {}
    temp['year_col'] = datecol.year
    temp['quarter_col'] = datecol.month    
    temp['month_col'] = datecol.month
    temp['dayofweek_col'] = datecol.dayofweek
    temp['weekofyear_col'] = datecol.weekofyear
    temp['dayofyear_col'] = datecol.dayofweek
    temp['day_col'] = datecol.day
    temp['hour_col'] = datecol.hour
    temp['minute_col'] = datecol.minute
    temp['second_col'] = datecol.second

    temp = pd.DataFrame(temp)
    tc = temp.columns
    
    # 값이 하나만 있는 경우는 컬럼 Drop(의미가 없음)
    for i in cg(tc) :
        if len(temp[tc[i]].unique()) == 1 :
            temp.drop(columns=tc[i], inplace=True)

    df = pd.concat([df, temp], axis=1)
    df = df[set(df.columns)]
    selection_list = list(df.columns.intersection(temp.columns))
    return df[set(df.columns)], selection_list
#%%
## 2020.04.13

df, selection_list = search_date(df)
date_col = selection_list
#%%
# 여기까지 상수 및 설정 단계
# 필요한 경우 date_col를 선택가능 함
## 2020.04.13

global x_var
x_var =  x_var + date_col
#%%
# train, data set 분리
# function split_train_and_test_set : train-test 데이터를 분리시킨다.
        
def split_train_and_test_set(df, x, subset_idx, test, *rate) :
    # df : 데이터 셋
    # x : 기준컬럼
    # rate : train, test, validate data 비율(list)
    # train 데이터가 더 많은 학습 데이터를 확보할 수 있도록 함
    # x, subset_idx 기준 데이터 정렬(내림차순)
    # test 데이터 대상이 되는 데이터만 학습
    
    df = df.sort_values(by=[x, subset_idx], ascending=False)
    temp = []
    for num in rate :
        temp.append(num)
    temp.sort(reverse=True) # 오름차순(작은값 먼저)
    base_rt = temp[0]
    try :
        if (sum(np.array(rate)) >= 0.99) & (sum(np.array(rate)) <= 1.01) :  
            train, validate = train_test_split(df, train_size = base_rt, random_state=seed, shuffle=False)
            if len(rate) >= 3 :
                validate, validate = train_test_split(test, test_size = temp[1], random_state=seed, shuffle=False)
                
            return train, validate, test
            
    except Exception as e :
        print(e)
train, validate, test = split_train_and_test_set(df, x, subset_idx, None, 0.7, 0.3)
#%%
# Standardization 1

## function stats_dataframe : 특정 컬럼 기준- y의 mean, stddev 그룹핑을 수행한다
## parameter
### df : 데이터 셋
### y = 예측하고자 하는 값의 컬럼명
### x = 기준 컬럼
### methond = 원하는 방법
def stats_dataframe(df, x, y, method) :
    if method == 'mean_value' :
        colname = [x,'mean_value']
        target_dataframe = pd.DataFrame(df.groupby([x])[y].mean().reset_index())
    elif method == 'stddev_value' :
        colname = [x,'stddev_value']
        target_dataframe = pd.DataFrame(df.groupby([x])[y].std().reset_index())     
    
    target_dataframe.columns = colname
    target_dataframe = pd.merge(df, target_dataframe, on = x)
    
    return target_dataframe

#%%
for i in range(len(iterate)) :
    train = stats_dataframe(train, subset_idx, variable_y, iterate[i])
#%%    
# Standardization 2

## function merge_dataframe : test - train 데이터 표준편차, mean 값으로 scaling
## parameter
### df : 데이터 셋
### y = 종속 변수
def merge_dataframe(df, y) :
    df['normal_y'] = (df[y] - df['mean_value']) / df['stddev_value']        
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
# validate
# print(min_variables[0:6])
# print(variables_name[0:6])
# print(base_lists[0])
# print(param_coef[0])
# print(param_pval[0])
# print(formula_set[0])
#%%
# function coef_extract
## 독립변수의 각 계수 산출
# min_idx : minimum coefficient, params : 변수 내 도메인, val : 계수 값, base : 독립변수 명, seq : subset_idx 기준
def coef_extract(min_idx, params_df, pval_df, x_var, seq) :
    # parameter_coef
    # idx = []
    # pval = pval_df[0].values[pd.Series(pval_df[0].index).apply(lambda x: x.split(')')[0][2:2+len(base)]) == base]
    # coef = params_df[0].values[pd.Series(params_df[0].index).apply(lambda x: x.split(')')[0][2:2+len(base)]) == base]
    text = str(pd.DataFrame(param_coef[seq]).T.index)
    
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
    return dataframe
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

# def export_to_file(dataframe, path) :
#    dataframe.to_csv(path + '\\20200324temp.csv', encoding='utf-8', index=False)

# export_to_file(sets, config)
#%%
# functions concat_dataframe 
# origin 데이터프레임의 컬럼과 데이터타입을 맞춘다.
def concat_dataframe(df, sets, x_var) :
    for i in cg(x_var) :
    # for i in range(2) :
        key = x_var[i]
        temp = sets[sets.key2 == key]
        temp.columns = x_var[i] + "_" + sets.columns
        temp.rename(columns={x_var[i] + "_" + "key1" : key, x_var[i] + "_" + "subset_idx" : subset_idx}, inplace=True)
        
        # 조인 전 데이터타입 처리
        # key1<-key2를 참조, subset_idx
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
        
        # 독립변수의 계수 p_value, 병합
        if i == 0 :
            merge_dataframe = pd.merge(df, temp_join, left_on=[key, subset_idx], right_on = [key, subset_idx], how='left')
        else :
            merge_dataframe = pd.merge(merge_dataframe, temp_join, left_on=[key, subset_idx], right_on = [key, subset_idx], how='left')
    return merge_dataframe
#%%
# functions after_job 
# 데이터 후처리를 수행 한다. 데이터프레임의 컬럼과 데이터타입을 맞춘다.
## 2020.04.13
## 2020.04.13
## 2020.04.13
## 2020.04.13
## 2020.04.13

def after_job(rate=0.05):
    df = concat_dataframe(train, sets, x_var)

    # 내부 데이터 정의
    coef, _pvalues, _key2, _baseline = [], [], [], []
    for i in cg(df.columns) :
        if '_pvalue' in df.columns[i] :
            _pvalues.append(df.columns[i])
            
        if '_coef' in df.columns[i] :
            coef.append(df.columns[i])
        
        if '_key2' in df.columns[i] :
            _key2.append(df.columns[i])

        if '_baseline' in df.columns[i] :
            _baseline.append(df.columns[i])
            
    # pvalue가 1(가설을 충족하는 데이터 0%)로 가정  
    # pvalue 사실 여부 확인 
    for j in cg(_pvalues) :
        tempo, coef_chk = df[_pvalues[j]], df[coef[j]]
        # pvalue를 확인하고 rate 이상인 계수는 1로 치환            
        bool_map = tempo[abs(df[_pvalues[j]]) > rate] = 1
        coef_chk[tempo == 1] = 1
        
        df[coef[j]] = coef_chk
        
    # 필요하지 않는 데이터 삭제(key2)
    df.drop(_key2, axis=1, inplace=True)
    
    # functions nullable_process 
    # baseline이 있더라도 coef에서 산출되지 않는 경우 - 데이터가 한 포인트 밖에 없음
    # NaN 값 처리
    # NaN 값 1로 채우기
    # df[coef], df[_pvalues] = df[coef].fillna(1), df[_pvalues].fillna(1)
    def nullable_process() :
        target = coef + _pvalues + _baseline
        for i in cg(target) :
            if df[target[i]].isna().any() == True :
                df[target[i]] = df[target[i]].fillna(1)
            # baseline 값 채우기 <- 원래 키값의 데이터로 입력
            if (target[i] in _baseline) & (df[target[i]].isna().any() == True) :
                print(target[i])
        return df
    
    # 모든 p_value 값이 0인 경우 그 컬럼은 삭제
    def nullable_column_drop(df):
        for columns in cg(_pvalues) :
            if len(df[df[_pvalues[columns]] == 1]) == len(df) :
                
                # coef,baseline, pvalues 삭제
                df.drop(columns=_pvalues[columns], inplace=True)
                df.drop(columns=_baseline[columns], inplace=True)
                df.drop(columns=coef[columns], inplace=True)
                coef.pop(columns)
        return df
    
    # 최종 null값 체크 : 나중에는 삭제
    for check in cg(df.columns) :
        try :
            if df[df.columns[check]].isna().any() == True :
                break
        except Exception as e :
            print(e)
            
    df = nullable_process()
    df = nullable_column_drop(df)
    return df, coef

train, coef = after_job(0.125)
#sample, coef = after_job(0.125)
#%%
# y에 영향을 주는 각각의 독립변수의 영향력을 계산한다.
## 2020.04.13
## 2020.04.13
## 2020.04.13
## 2020.04.13
## 2020.04.13

def estimate_individual_power(dset, x_var, coef):
    dset['all_individual_power'] = dset['normal_y']
    colname = ['all_individual_power']
    for i in cg(coef) :
        baseline = x_var[i] + '_' + 'individual_power'
        colname.append(baseline)
        print(baseline)
        dset[baseline] = dset['normal_y'] / dset[coef[i]]       
        dset['all_individual_power'] = dset['all_individual_power'] / dset[coef[i]]
     
    return dset, colname

sample, test_colname = estimate_individual_power(train, x_var, coef)
# train, test_colname = estimate_individual_power(sample, x_var, coef)
#%%
# 각 독립변수의 영향력을 concat 입력
# 없는 경우 Aggregation : EMA로 최신 값을 적용
def testset_process(validate, train, x_var, test_colname, subset_idx):
    try :            
        x = [subset_idx] + x_var
        # validate 독립변수 지정
        valset = validate[x]
        # 계수 집계 시 median 값으로 처리
        trainset = train[x + test_colname + coef + iterate].groupby(x).median().reset_index()
        # valset 기준 병합
        valset = pd.merge(valset, trainset, left_on=x, right_on=x, how='left')
        trainset['normal_y'] = train['normal_y']
        # 결측치 처리 
        # outer join으로 있는 요소들만 필터링
        #for null in cg(valset.columns) :
        #    if valset.iloc[:,null].isna().any() == True :
        #        valset.iloc[:, null] = valset.iloc[:, null].fillna(1)
    except Exception as e : 
        print(e)
        pass
                
    return valset
#%%    
valset = testset_process(validate, train, x_var, test_colname, subset_idx)