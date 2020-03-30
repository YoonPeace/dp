# -*- coding: utf-8 -*-

# Dynamic Pricing 모델이 수행할 수 있도록 지원 한다
#####import libraries
import os, sys
import pandas as pd
import numpy as np
import json
import warnings
import statsmodels.api as sm
import platform
from statsmodels.formula.api import ols
import time

import filerwx

class helper :
    osname = platform.system()
    def __init__(self):
        self.path = os.getcwd()
        self.config = ''

    # properties 파일 열기
    def properties(self):
        ppath = self.path
        with open('../etc/properties.json', 'r', encoding='utf-8') as f :
            config = json.load(f)
        self.config = config

def os_chk(cmd) :
    # bootstrap 에서 처리하는 각종 명령어를 처리한다.
    if platform.system().upper() == 'WINDOWS' :
        # windows
        os.system(str(cmd))
    else :
        # others
        print(os.system(str(cmd)))

# 데이터 명세 export - 미완료
def export_describe(df):
    try :
        # 변수목록
        # 메타 데이터 저장
        # 중복하여 호출(재설계 대상)
        help = helper()
        help.properties()
        config = help.config

        # name 호출
        name = filerwx.file_name_generate()
        sys.stdout = open(config['file_to_location'] + '\\' + name + '.txt', 'w', encoding='utf-8')
        print('dataframe columns : %s \n' % list(df.columns))
    except Exception as e:
        print(e)

# Error 처리 - 미완료
def error_log(self):
    try :
        pass
    except Exception as e :
        log_path = config['default_path'] + config['log_file']
        print('rr')
