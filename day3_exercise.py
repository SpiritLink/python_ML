import pandas as pd
import os
import openpyxl
import requests
import json
import numpy as np

# 1. dataset 자료 불러오기
os.chdir('C:/Users/USER/Downloads')
excel_data = pd.read_excel('dataset.xlsx')

# 2. 20130601~20130630 시간당 기상 자료 불러오기
url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?'
key = 'serviceKey=amZyo3vOJtsNCiW0LO9Gha12rCV83HHteCYsvVOT35lyuvbBLEqQFfy%2BOfAvC4wZrR8KNeQ%2B4lDz4V4nxqEfFA%3D%3D'
param = '&dataType=JSON&numOfRows=999&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=159&endDt=20130620&endHh=23&startHh=01&startDt=20130601'

url = url + key + param
response = requests.get(url)

# HTTP 응답값을 json 형태로 파싱한 뒤, data frame 형태로 재구성
dic = json.loads(response.text)
doc = dic['response']['body']['items']['item']
api_data = pd.DataFrame(doc)
api_data.info()

for i in api_data.columns:
    try:
        api_data[i] = api_data[i].astype('int')
    except:
        try:
            api_data[i] = api_data[i].astype('float')
        except:
            continue

api_data['tm'] = pd.to_datetime(api_data['tm'])
api_data.dropna(axis = 1, thresh = api_data.shape[0]/2, inplace = True)
excel_data['발생일시'] = pd.to_datetime(excel_data['발생일시'])
data = pd.merge(excel_data, api_data, how = 'right',
         left_on = '발생일시', right_on = 'tm')

data['사고여부'].replace(np.nan, 0, inplace = True)
data.drop(['발생일시', 'tm', 'stnNm', 'stnId', 'clfmAbbrCd'], axis = 1, inplace = True)

# 3. data set 과 기상자료 (부산) 결합 (변수추가방법)
# excel data 와 api data를 머지해야 한다.
# pd.merge(df_left, df_right, how =)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test


from sklearn.imput import KNNImputer
ki = KNNImputer(n_neighbors=5)
ki.fit(X_train)
X_train_imp = ki.transform(X_train)
X_test_imp = ki.transform(X_test)

from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors=5)
X_train_re, y_train_re = sm.fit_resample(X_train_imp, y_train_imp)

from sklearn.model_selection import GridSearchCV
grid = {'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5]}
gs = GridSearchCV(estimator = RandomForestClassify, param_grid = grid)
gs.fit(X_train_re)
rf = gs.best_estimator
rf.fit(X_train_re)
rf.predict(X_train_re)
re.predict(X_test)
a = 3