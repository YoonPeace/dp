# -*- coding: utf-8 -*-
# Dynamic Pricing 모델 동작을 수행한다.
import filerwx
import preprocess
import helper
import sys
import time

# Function Name : init_job
# 외부에서 파라미터를 받아와 모델을 실행할 수 있게 한다.
# parameter : param : init 0/1 (실행 / 비실행), file_yn (파일여부)
def init_job(param, file_yn) :
    def file_pass(file_yn) :
        if file_yn == 1 :
           return filerwx.file_import(1)
    return file_pass(file_yn)


# Function Name : bootstraping
# Function Descrption : process를 실행한다
# parameter : param : init 0/1 (실행 / 비실행), file_yn 0/1(파일여부), flag = file or db(1/2)
# default = 실행, 파일, 파일(1, 1, 1)
def bootstraping(param=1, file_yn=1, flag=1) :
    if flag == 1 :
        #################################### job initiaition ####################################
        df = init_job(param, file_yn)

        # 데이터의 메타 정보 저장
        # 일단 패스
        # helper.export_describe(df)
        # 저장될때 까지 sleep
        # time.sleep(3)
        #########################################################################################

        ####################################  preprocessing  ####################################
        # train-test data set 분리
        tr, ts = preprocess.split_train_and_test_set(df, 'datetime', [0.8, 0.2])
        ### print(train)

        # Standardization 하기 위한 train/test 데이터별 mean, stddev 계산
        # mean, stddev 계산 후 y scaling 수행
        # train
        tr = preprocess.stats_dataframe(tr, 'prod_cd', 'sell_qty')
        tr = preprocess.scale_dataframe(tr, 'sell_qty')
        # test
        ts = preprocess.stats_dataframe(ts, 'prod_cd', 'sell_qty')
        ts = preprocess.scale_dataframe(ts, 'sell_qty')

        # preprocess.coef_extract()

        #########################################################################################
    elif flag == 2 :
        init_job(param, file_yn, flag)

    return tr, ts

#test
train , test = bootstraping()
print(test)
# Function Name : bootstraping
# Function Descrption : 작업이 끝나면 프로세스를 종료한다.
def finish_job() :
    return print('end of job')
