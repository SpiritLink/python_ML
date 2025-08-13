import seaborn as sns
df = sns.load_dataset('mpg')
df.info()
df.drop('name', axis = 1, inplace=True)

import pandas as pd
df1 = pd.get_dummies(df, columns = ['origin'], dtype = 'int', 
               drop_first = True)

# KNN Imputer- KNN을 이용하여 결측자료 대체
# from sklearn.impute import KNNImputer
# ki = KNNImputer(n_neighbors = 정수)
# - 옵션: n_neighbors : KNN 수행 시 판단을 위한 주변 데이터 갯수 지정
# ki.fit(데이터)
# ki.transform(데이터)
# => 출력결과: 2d array 형태이므로 데이터프레임 변환 필요

from sklearn.impute import KNNImputer
ki = KNNImputer(n_neighbors = 5)
ki.fit(df1)
ki.transform(df1)

df1 = pd.DataFrame(ki.transform(df1), columns = df1.columns)

# 주성분 분석(PCA)
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 주성분 갯수, random_state = 난수번호)
# - 스크리도표 작성하여 주성분 갯수를 파악하고자 할때는 주성분의 갯수를 변수갯수로 지정
# => 데이터.shape[1]
# pca.fit(데이터)
# => 데이터 형태: 2d array, DataFrame 형태
# => 독립데이터만 입력(종속데이터 제외)
# => train자료를 대입해서 관계식 형성
# pca.transform(데이터)
# => 2d array 형태로 주어진 주성분 값 계산 출력
# => DataFrame 형태로 변환 필요
# df_pca = pd.DataFrame(pca.transform(데이터), columns = ['PC1', 'PC2'])
# pca.explained_variance_ratio_ # 주성분 별 설명력 출력


from sklearn.decomposition import PCA
pca = PCA(n_components = df1.shape[1], random_state = 0)
pca.fit(df1)
pca.transform(df1)
pca.explained_variance_ratio_
# Out[28]: 
# array([9.97535863e-01, 2.06411676e-03, 3.55634814e-04, 3.21488464e-05,
#        7.62072957e-06, 3.92585247e-06, 3.81937654e-07, 2.29485731e-07,
#        7.82045594e-08])

# 스크리도표 작성
variance = pca.explained_variance_ratio_
import matplotlib.pyplot as plt
plt.plot(range(1,10), variance, ls = '--', marker = 'o',
         mfc = 'red')


# 공공데이터포털의 open api 데이터 불러오기
# - open api 활용 명세서 확인하여 요청 주소 예시 참고

import requests

# 요청 url 입력
url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?'
# 개인 키 입력
key = '개인키'
# 요청 변수들 입력
params = 'numOfRows=10&dataType=JSON&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20200310&endHh=01&startHh=01&startDt=20190120'

# 요청 url 결합
url = url+key+params

# 데이터 요청하기
response = requests.get(url)
print(response.content)

# 요청된 데이터(json)를 데이터프레임으로 변환
import json
dic = json.loads(response.text)
doc = dic['response']['body']['items']['item']
data = pd.DataFrame(doc)
data.info()
# 주의!! 불러온 데이터는 모두 문자형으로 입력되어 있음
# => 변수별 데이터 유형 변환 필요
# 주의!! 결측의 경우 ''으로 문자 처리 되어 있으므로 결측으로 변환 필요
# import numpy as np
# 데이터.replace('', np.nan, inplace=True)

# 기상청 asos 데이터(시간측정 데이터) 불러오기
url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?'
key = '개인키'
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

# 요일 데이터로 변환
data['tm'] = pd.to_datetime(data['tm'])
# 요일데이터에서 연도 추출
data['tm'].dt.year
# 요일 데이터에서 요일 추출
data['tm'].dt.weekday
# 0: 월요일 ~ 6: 일요일

import numpy as np
data.replace('', np.nan, inplace = True)
data['taQcflg'].astype('float')



# 기상청 asos 데이터(일측정 데이터) 불러오기
url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList?'
key = '개인키'
param = '&dataType=JSON&numOfRows=11&pageNo=1&dataCd=ASOS&dateCd=DAY&startDt=20250801&endDt=20250811&stnIds=114'
url = url + key + param

import requests
response = requests.get(url)

import json
dic = json.loads(response.text)
doc = dic['response']['body']['items']['item']
data = pd.DataFrame(doc)

# 결측자료 결측으로 전환
import numpy as np
data.replace('', np.nan, inplace=True)

# 데이터 유형들 일괄 변환
for name in data.columns:
    try:
        data[name] = data[name].astype('int')
    except:
        try:
            data[name] = data[name].astype('float')
        except:
            continue

# 시간 데이터를 datetime 유형으로 변환
import pandas as pd
data['tm'] = pd.to_datetime(data['tm'])
data.info()


# 웹크롤링
pip install selenium # 웹크롤링에 필요한 라이브러리 설치
import selenium
from selenium import webdriver
driver = webdriver.Chrome()

# 위의 webdriver.Chrome()이 실행이 안되는 경우 아래의 코드 수행(최초 한번만)
# pip install webdriver-manager # chrome 조작을 위한 라이브러리 설치
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
driver=webdriver.Chrome(service = Service(ChromeDriverManager().install()), 
                        options = Options())

# 대선결과 득표수 크롤링
driver.get('http://info.nec.go.kr/main/showDocument.xhtml?electionId=0000000000&topMenuId=VC&secondMenuId=VCCP09') #페이지 이동
from selenium.webdriver.common.by import By
# ID 속성을 이용하여 해당 위치 클릭
driver.find_element(By.ID, 'electionType1').click() 
# ID 속성을 이용하여 드롭박스 형태로 나타나는 옵션 값 선택
driver.find_element(By.ID, 'electionName').send_keys('제20대') 

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# 화면전환에 따른 속성값이 나타날때까지 기다림
WebDriverWait(driver, 10).\
    until(EC.presence_of_element_located((By.ID, 'electionCode'))) 
driver.find_element(By.ID, 'electionCode').send_keys('대통령선거')

# ID, TAG_NAME 등을 파악하기 힘든 경우 XPATH 복사를 이용해 위치 선택
sido_list_raw = driver.find_element(By.XPATH, '''//*[@id="cityCode"]''') 
sido_list = sido_list_raw.find_elements(By.TAG_NAME, 'option')
sido_names_values = [option.text for option in sido_list][2:]

# 아래부터는 크롤링 하고자 하는 페이지에 맞춘 데이터 전처리 과정
def move_sido(name):
    element = driver.find_element(By.ID, 'cityCode')
    element.send_keys(name)
    WebDriverWait(driver, 10).\
        until(EC.presence_of_element_located((By.ID, 'searchBtn')))
    driver.find_element(By.ID, 'searchBtn').click()

def get_num(tmp):
    return float(tmp.replace(',', ''))


def append_data(rows, sido_name, data):
    for i, row in enumerate(rows[5:], start = 2):
        if i % 2 == 0:
            cells = row.find_elements(By.TAG_NAME, 'td')
            data['광역시도'].append(sido_name)
            data['시군'].append(cells[0].text.strip())
            data['전체'].append(get_num(cells[2].text.strip()))
            data['문'].append(get_num(cells[3].text.strip()))
            data['홍'].append(get_num(cells[4].text.strip()))
            data['안'].append(get_num(cells[5].text.strip()))

election_result_raw = {'광역시도': [], '시군': [], '전체': [], 
                       '문': [], '홍':[], '안':[]}

for each_sido in sido_names_values:
    move_sido(each_sido)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'table01')))
    table = driver.find_element(By.TAG_NAME, 'table')
    rows = table.find_elements(By.TAG_NAME, 'tr') # row 자료들을 가지고 옴

    append_data(rows, each_sido, election_result_raw)

import pandas as pd
election_result = pd.DataFrame(election_result_raw)


# 오피넷 페이지에서 구별 주유소 가격 크롤링 (엑셀과 같은 데이터 다운 형태)
driver = webdriver.Chrome()
driver.get('https://www.opinet.co.kr/searRgSelect.do')

sido_select_element = driver.find_element(By.ID, 'SIDO_NM0')

from selenium.webdriver.support.ui import Select
sido_select = Select(sido_select_element).select_by_visible_text('서울')

gu_list_raw = driver.find_element(By.XPATH, '''//*[@id="SIGUNGU_NM0"]''')
gu_list = gu_list_raw.find_elements(By.TAG_NAME, 'option')
gu_names = [option.text for option in gu_list][1:]

import time
for gu in gu_names:
    element = driver.find_element(By.ID, 'SIGUNGU_NM0')
    element.send_keys(gu)
    time.sleep(2) # 프로그램 자체를 강제로 멈추게 함. 괄호안은 초(second)
    driver.find_element(By.ID, 'searRgSelect').click()
    time.sleep(2)
    element_get_excel = driver.find_element(By.XPATH, 
                    '''//*[@id="templ_list0"]/div[7]/div/a''').click()
    time.sleep(2)

import pandas as pd
df1 = pd.read_excel('file_name1.xls')
df2 = pd.read_excel('file_name2.xls')
pd.concat([df1, df2], axis = 0/1, join = 'outer'/'inner')

# 다운받은 폴더내에 파일명의 키워드를 이용하여 다운받은 파일 이름 불러오기
import os
os.chdir('C:/Users/USER/Downloads')
import glob
file_names = glob.glob('*주유소*.xls')

# xls 파일 불러오기의 경우, 경우에 따라 라이브러리 업그레이드 필요
# pip install --upgrade xlrd

# 불러온 파일을 하나의 데이터프레임으로 결합
final_data = pd.DataFrame()
for name in file_names:
    raw_data = pd.read_excel(name, header = 2)
    final_data = pd.concat([final_data, raw_data], 
                           axis = 0, join = 'outer')

import numpy as np
final_data.replace('-', np.nan, inplace=True)

for var in final_data.columns:
    try:
        final_data[var] = final_data[var].astype('float')
    except:
        continue

# 주소에서 구를 추출 - 구별 이름 정보를 이용하여 지도위에 정보 plot을 위함
final_data['구'] = final_data['주소'].str.split(' ').str.get(1)
final_data['구'].unique()


import pandas as pd
import numpy as np
# index에 지정한 변수에 따라 values에 지정한 변수의 값을 aggfunc 함수에 적용하여 결과 출력
gu_data = pd.pivot_table(final_data, index = ['구'],
                         values = ['휘발유'], aggfunc = np.mean)

# json 파일 불러오기 - 위경도가 포함된 지리정보 데이터
import json
geo_str = json.load(open('skorea_municipalities_geo_simple.json', 
                         encoding='utf-8'))

# 지도 그리기 위한 라이브러리 설치
# pip install folium
import folium
oil_map = folium.Map(location = [37.5502, 126.982], 
                     zoom_start = 12)
folium.Choropleth(geo_data = geo_str, data = gu_data,
                  columns = [gu_data.index, '휘발유'],
                  key_on = 'feature.id',
                  fill_color = 'PuBu', fill_opacity=0.7,
                  line_opacity= 0.2).add_to(oil_map)
oil_map.save('지도.html')


