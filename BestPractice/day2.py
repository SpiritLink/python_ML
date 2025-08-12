import seaborn as sns
df = sns.load_dataset('mpg')
df.info()
df.drop('name', axis = 1, inplace=True)

import pandas as pd
df1 = pd.get_dummies(df, columns = ['origin'], dtype = 'int', 
               drop_first = True)

# KNN Imputer
from sklearn.impute import KNNImputer
ki = KNNImputer(n_neighbors = 5)
ki.fit(df1)
ki.transform(df1)

df1 = pd.DataFrame(ki.transform(df1), columns = df1.columns)

# 주성분 분석(PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, random_state = 0)
pca.fit(df1)
=> 데이터 형태: 2d array, DataFrame 형태
=> 독립데이터만 입력(종속데이터 제외)
=> train자료를 대입해서 관계식 형성
pca.transform(df1)
=> 2d array 형태로 주어진 주성분 값 계산 출력
=> DataFrame 형태로 변환 필요
df_pca = pd.DataFrame(pca.transform(df1), columns = ['PC1', 'PC2'])


from sklearn.decomposition import PCA
pca = PCA(n_components = df1.shape[1], random_state = 0)
pca.fit(df1)
pca.transform(df1)
pca.explained_variance_ratio_
Out[28]: 
array([9.97535863e-01, 2.06411676e-03, 3.55634814e-04, 3.21488464e-05,
       7.62072957e-06, 3.92585247e-06, 3.81937654e-07, 2.29485731e-07,
       7.82045594e-08])

variance = pca.explained_variance_ratio_
import matplotlib.pyplot as plt
plt.plot(range(1,10), variance, ls = '--', marker = 'o',
         mfc = 'red')


4Fs6k7F3R5MpSUXwJG5kJkI95Z7Q66N8HeCFaR5xVC0R%2FxcjPxbPLNCOe1TQYyIY6nCVhsdaTykCtdwY%2FOAn6g%3D%3D



















import requests

url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?'
key = 'serviceKey=4Fs6k7F3R5MpSUXwJG5kJkI95Z7Q66N8HeCFaR5xVC0R%2FxcjPxbPLNCOe1TQYyIY6nCVhsdaTykCtdwY%2FOAn6g%3D%3D&'
params = 'numOfRows=10&dataType=JSON&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20200310&endHh=01&startHh=01&startDt=20190120'
url = url+key+params

url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?'
params ={'serviceKey' : '4Fs6k7F3R5MpSUXwJG5kJkI95Z7Q66N8HeCFaR5xVC0R%2FxcjPxbPLNCOe1TQYyIY6nCVhsdaTykCtdwY%2FOAn6g%3D%3D', 
         'pageNo' : '1', 
         'numOfRows' : '10', 
         'dataType' : 'JSON', 
         'dataCd' : 'ASOS', 
         'dateCd' : 'HR', 
         'startDt' : '20100101', 'startHh' : '01', 
         'endDt' : '20100601', 'endHh' : '01', 
         'stnIds' : '108' }

response = requests.get(url)
print(response.content)

import json
dic = json.loads(response.text)
doc = dic['response']['body']['items']['item']
data = pd.DataFrame(doc)
data.info()



url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?'
key = 'serviceKey=4Fs6k7F3R5MpSUXwJG5kJkI95Z7Q66N8HeCFaR5xVC0R%2FxcjPxbPLNCOe1TQYyIY6nCVhsdaTykCtdwY%2FOAn6g%3D%3D'
param = '&dataType=JSON&numOfRows=10&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20250811&endHh=23&startHh=01&startDt=20250811'
url = url + key + param

import requests
response = requests.get(url)

import json
dic = json.loads(response.text)
doc = dic['response']['body']['items']['item']
data = pd.DataFrame(doc)
data.info()
data['stnId'] = data['stnId'].astype('float')
data['tm'] = pd.to_datetime(data['tm'])
data['tm'].dt.year
data['tm'].dt.weekday
0: 월요일 ~ 6: 일요일

import numpy as np
data.replace('', np.nan, inplace = True)
data['taQcflg'].astype('float')




url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList?'
key = 'serviceKey=4Fs6k7F3R5MpSUXwJG5kJkI95Z7Q66N8HeCFaR5xVC0R%2FxcjPxbPLNCOe1TQYyIY6nCVhsdaTykCtdwY%2FOAn6g%3D%3D'
param = '&dataType=JSON&numOfRows=11&pageNo=1&dataCd=ASOS&dateCd=DAY&startDt=20250801&endDt=20250811&stnIds=114'
url = url + key + param

import requests
response = requests.get(url)

import json
dic = json.loads(response.text)
doc = dic['response']['body']['items']['item']
data = pd.DataFrame(doc)

import numpy as np
data.replace('', np.nan, inplace=True)

for name in data.columns:
    try:
        data[name] = data[name].astype('int')
    except:
        try:
            data[name] = data[name].astype('float')
        except:
            continue

import pandas as pd
data['tm'] = pd.to_datetime(data['tm'])
data.info()

pip install selenium
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
pip install webdriver-manager
from webdriver_manager.chrome import ChromeDriverManager

driver=webdriver.Chrome(service = Service(ChromeDriverManager().install()), 
                        options = Options())
driver = webdriver.Chrome()
driver.get('https://www.naver.com')





