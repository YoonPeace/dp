# -*- coding: utf-8 -*-

# 모델에 데이터를 넣기 전 독립 변수를 선택한다
import helper
# import
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols

# train, data set 분리
# function split_train_and_test_set : train-test 데이터를 분리시킨다.
# df : 데이터 셋
# x : 기준컬럼
# rate : train, test, validate data 비율(list)
# default rate = 8 : 2

def split_train_and_test_set(df, x, rate=[0.8,0.2]):
    # x 기준 정렬
    df.sort_values(by=x, ascending=True)
    try:
        if len(rate) <= 2:
            train, test = train_test_split(df, test_size=rate[1], random_state=7)
            return train, test
        elif len(rate) >= 3:
            probs = np.random.rapnd(len(df))
            train_mask = probs < rate[0]
            test_mask = (probs >= rate[0]) & (probs < rate[0] + rate[1])
            validation_mask = probs >= 1 - rate[2]

            train, test, validation = df[train_mask], df[test_mask], df[validation_mask]
            return train, test, validation
    except Exception as e:
        print(e)
        raise

# Standardization 1
## function stats_dataframe : 특정 컬럼 기준- y의 mean, stddev 그룹핑을 수행한다
## parameter
### df : 데이터 셋
### y = 예측하고자 하는 값의 컬럼명
### x = 기준 컬럼
### methond = 원하는 방법(mean, stddev)
def stats_dataframe(df, x, y, method = ['mean', 'stddev']) :
    def stats_method(df, x, y, method):
        if method == 'mean':
            colname = [x, 'mean_value']
            target_dataframe = pd.DataFrame(df.groupby([x])[y].mean().reset_index())
        elif method == 'stddev':
            colname = [x, 'std_value']
            target_dataframe = pd.DataFrame(df.groupby([x])[y].std().reset_index())

        target_dataframe.columns = colname
        target_dataframe = pd.merge(df, target_dataframe, on=x)
        return target_dataframe
    for i in range(len(method)):
        df = stats_method(df, 'prod_cd', 'sell_qty', method[i])
    return df

# Standardization 2
## function merge_dataframe : test - train 데이터 표준편차, mean 값으로 scaling
## parameter
### df : 데이터 셋
### y = 종속 변수
def scale_dataframe(df, y):
    df['normal_y'] = (df[y] - df['mean_value']) / df['std_value']
    return df

#%%
# function coef_extract
## 독립변수의 각 계수 산출
# min_idx : minimum coefficient, params : 변수 내 도메인, val : 계수 값, base : 독립변수 명
def coef_extract(min_idx, params, base) :
    # parameter_coef
    params, names = params[0].values[pd.Series(params[0].index).apply(lambda x: x.split(')')[0][2:2+len(base)]) == base], []
    for point in range(len(params) + 1):
        names.append(point)
        if point == min_idx:
            names.pop(point)
    return params, names


## function columns_name : value_store로 산출된 데이터를 데이터프레임화 한다.
## parameter
### x_var : 독립변수 집합
### base_lists : 서브셋 독립변수 기준
### rows : iteration 변수

def columns_name(base_lists, x_var, row):
    base_lists_name = pd.Series(base_lists).apply(lambda x: str(x) + '_' + x_var[row])
    return list(base_lists_name)

# 상수
x_var = ['tot_disc', 'dow', 'hr']

# OLS
## function value_store : 데이터셋 내 y와 독립변수들 간의 계수와 p_value를 뽑아 낸다
## parameter
### df : 데이터셋
### x : 서브셋 기준 독립변수
### x_var : 데이터셋
### normal_y : 표준화 된 y
### y : 종속변수 y

def value_store(df, x, x_var, normal_y, y):
    min_variables, variables_name, base_lists = [], [], []

    # parameter coef, parameter coef
    param_coef, param_pval = [], []
    formula = y + ' ~ '
    seq = 0
    for i in range(len(set(df[x]))):
        # for i in range(2) :
        # subset 지정(prod_cd(base variable) 별 subset)
        seq += 1
        formula = y + ' ~ '
        # print('iteration formula %s' % formula) #
        subset = df[df[x] == np.unique(df[x])[i]].iloc[:, 1:]
        base = np.unique(df[x])[i]
        for j in range(len(x_var)):
            variables_name.append(x_var[j])
            min_variables.append(np.argmin(subset.groupby([x_var[j]])[normal_y].mean()))
            # np.argmin(df[df['prod_cd'] == np.unique(df['prod_cd'])[0]].iloc[:, 1:].groupby('tot_disc')['normal_y'].min()) <- 최소값 기준
            # return value
            # x_var[i],x_var[i], x_var[i]
            # value,   value,    value

            # 공식 만들기
            # 독립변수 개수 만큼 creation
            if i == 0:
                form = "C(" + x_var[j] + ", Treatment(" + str(min_variables[j]) + "))"
            else:
                form = "C(" + x_var[j] + ", Treatment(" + str(min_variables[len(x_var) * i + j]) + "))"
            for k in range(len(x_var)):
                if j == k:
                    if k == (len(x_var) - 1):
                        formula += form
                    else:
                        formula += form + " + "
        result = ols(formula, data=subset).fit()
        base_lists.append(base)
        param_coef.append([result.params])
        param_pval.append([result.pvalues])
        print(formula)
    return min_variables, variables_name, base_lists, param_coef, param_pval


min_variables, variables_name, base_lists, param_coef, param_pval = value_store(train, 'prod_cd', x_var, 'normal_y',
                                                                                'sell_qty')


# %%
## function to_df_process : value_store로 산출된 데이터를 데이터프레임화 한다.
## parameter
### x_var : 데이터셋
### min_variables : 독립변수의 평균값 집합
### param_coef : 독립변수의 계수 집합
### subset_idx : 서브셋 기준 독립변수 리스트
# parameter 1번째의 서브셋 array위치 0 * len(base_lists)
# parameter 2번째의 서브셋 array위치 1 * len(base_lists)
# parameter 3번째의 서브셋 array위치 2 * len(base_lists)

def to_df_process(x_var, min_variables, param_coef, subset_idx):
    num = len(x_var)
    val, idx = [], []
    # 독립변수 만큼 iteration
    for x in range(num):
        print('===============================start parse data =================================')
        for seq in range(len(subset_idx)):
            # for seq in range(4) :
            print('iteration %i' % (seq + 1))
            print(x_var[x], coef_extract(min_variables[num * seq:(num * seq) + num], param_coef[seq], x_var[x])[0])
            # print(coef_extract(min_variables, param_coef, seq, x_var[x]))
            # val.append((coef_extract(min_variables[num*seq:(num*seq)+num], param_coef[seq], x, x_var[x])[0]))
            val.append(coef_extract(min_variables[num * seq:(num * seq) + num], param_coef[seq], x_var[x])[0])
            idx.append(coef_extract(min_variables[num * seq:(num * seq) + num], param_coef[seq], x_var[x])[1])

            # idx.append((coef_extract(min_variables[0], param_coef, x, x_var[x])[1]))

        print('==============================parse data job end ================================')

    return np.array([val, idx])

#    return print(subset_idx)
coefficients = to_df_process(x_var, min_variables, param_coef, base_lists)
pvalues = to_df_process(x_var, min_variables, param_pval, base_lists)

# %%
## function subset_to_df : subset 별 pvalue, coef 값을 데이터프레임화 한다.
## parameter
### x_var : 독립변수 집합
### coefficients : 독립변수의 계수 집합
### base_lists : 서브셋 기준 독립변수 리스트

# function seubset to df
# parameter
# base_lists : 서브셋 기준 독립변수 리스트
# x_var : 독립변수 집합
# coefficients : 독립변수의 계수 집합
def subset_to_df(base_lists, x_var, coefficients):
    columns = []
    #  columns += columns_name(base_lists, x_var, row)
    for i in range(len(variables_name)):
        if i == 0:
            subset_to_df = pd.Series(coefficients[0][i])
        else:
            subset_to_df = pd.concat([subset_to_df, pd.Series(coefficients[0][i])], axis=1)

        if i % len(base_lists) == 0:
            columns += columns_name(base_lists, x_var, int(i / len(base_lists)))
    subset_to_df.columns = columns
    return subset_to_df


# %%
# function garbage_p_value_filter
# 일정 유의 구간에 계수만 추출한다
# parameter
# base_lists : 서브셋 val : p_value 값(기본 0.05)
# subset : 서브셋

def garbage_p_value_filter(subset, val):
    # NaN 값인 경우 데이터가 없으므로 의미가 없다는 가정을 취함
    # p_value와 coefficients를 1로 강제 치환
    subset = subset[(subset <= val) & (subset >= -(val))]
    subset = subset.fillna(1)
    return subset


# %%
# merge_baseline
# function merge baselikne
# 각 서브셋의 베이스라인을 산출한다
# parameter
# x_var : 독립변수 집합
# variables :  변수 셋
# min_variables : 독립변수의 베이스라인 값(min값 기준)
# base_lists : 서브셋

def merge_baseline(x_var, variables, min_variables, base_lists):
    merge = pd.DataFrame([base_lists * len(x_var), variables, min_variables]).T.set_index(0)
    return merge


# %%
# subset_spread
def subset_spread(df=merge_baseline(x_var, variables_name, min_variables, base_lists), val=0.05):
    coef = subset_to_df(base_lists, x_var, coefficients)
    p = garbage_p_value_filter(subset_to_df(base_lists, x_var, pvalues), val)
    coefmap = coef[p != 1].T
    pmap = p[p != 1].T
    return coefmap, pmap


def sub(base_lists, coefficients, x_var):
    coefmap, pmap = subset_spread(0.05)
    # for i in range(len(variables_name)) :

    return coefmap, pmap


coefmap, pmap = sub(base_lists, coefficients, x_var)
