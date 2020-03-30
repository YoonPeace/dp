# -*- coding: utf-8 -*-
# -*- 데이터를 불러 오거나 내보낸다.
import os, sys
import pandas as pd
import helper
import time

# 원천 데이터 Import
def file_import(yn):
    if yn == 1 :
        help = helper.helper()
        help.properties()
        config = help.config
        # 가장 값이 큰 이름이 앞으로
        lists = os.listdir(config['file_source_location'])
        lists.sort(reverse=True)
        this_file = lists[0]
        current_date = this_file
        try:
            print(config['file_source_location'] + '\\' + this_file)
            df = pd.read_csv(config['file_source_location'] + '\\' + this_file, encoding='utf-8', index_col=False)
            # dataframe 반환
            return df
            #>> preprocess 로 이동

        except Exception as e:
            print(e)
            raise e

def file_name_generate(current_date=time.localtime()) :
    # 파일 이름 생성
    name = str(current_date.tm_year) + str(current_date.tm_mon) + str(current_date.tm_mday) + str(current_date.tm_hour) + str(current_date.tm_min) + str(current_date.tm_sec)
    return name

# 데이터 후처리
def file_export(self, df, name):
    try:
        # 파일 내보내기
        return df.to_csv(config['file_to_location'] + '\\' + file_source_location + name, encoding='utf-8', index=False)

    except Exception as se:
        # 에러 등록
        error_log()
        print(e)
        raise e

def connect_db(db_param, flag) :
    if flag == 1 :
        if db_param == 'postgres' :
           print('hello')

def export_db(db_param, flag) :
    if flag == 2 :
        if db_param == 'postgres' :
            print('hello')
