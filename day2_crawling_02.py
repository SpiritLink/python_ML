import pandas as pd
import requests
import json
import numpy as np

# def auto_convert_df_types(df, datetime_cols=None, int_cols=None, na_fill_for_int=None):
#     """
#     DataFrame 컬럼 타입을 자동 변환하는 함수.
#     - 모든 object/문자열 컬럼에 대해 숫자 변환 가능하면 숫자로, 날짜 컬럼은 날짜로 변환
#     - int 컬럼은 NaN이 있으면 Int64 (nullable int)로 변환
#     - 필요 시 특정 int 컬럼은 NaN을 0으로 채운 뒤 int 변환
#
#     Parameters:
#         df (pd.DataFrame): 변환 대상 DataFrame
#         datetime_cols (list): 날짜로 변환할 컬럼 목록 (자동 감지 대신 지정)
#         int_cols (list): 정수(int)로 변환할 컬럼 목록
#         na_fill_for_int (int/float): int 변환 시 NaN 대체 값 (예: 0)
#     """
#
#     df = df.copy()
#     # 1. 빈 문자열을 NaN으로 처리
#     df.replace('', np.nan, inplace=True)
#
#     # 2. 날짜 변환 (사용자가 지정한 경우)
#     if datetime_cols:
#         for col in datetime_cols:
#             df[col] = pd.to_datetime(df[col], errors='coerce')
#
#     # 3. 숫자 변환 가능한 컬럼 처리
#     for col in df.columns:
#         if col in (datetime_cols or []):  # 날짜 컬럼은 건너뛰기
#             continue
#         if df[col].dtype == 'object':
#             converted = pd.to_numeric(df[col], errors='coerce')
#             # 변환 결과에서 NaN이 아닌 값이 1개 이상이면 숫자 컬럼으로 처리
#             if converted.notna().sum() > 0:
#                 df[col] = converted
#
#     # 4. 지정된 int 컬럼 처리
#     if int_cols:
#         for col in int_cols:
#             if na_fill_for_int is not None:
#                 df[col] = df[col].fillna(na_fill_for_int).astype(int)
#             else:
#                 df[col] = df[col].astype('Int64')  # NaN 허용 정수 타입
#
#     return df
#
# url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList?'
# key = 'serviceKey=amZyo3vOJtsNCiW0LO9Gha12rCV83HHteCYsvVOT35lyuvbBLEqQFfy%2BOfAvC4wZrR8KNeQ%2B4lDz4V4nxqEfFA%3D%3D'
# param = '&dataType=JSON&numOfRows)=10&pageNo=1&dataCd=ASOS&dateCd=DAY&startDt=20250801&endDt=20250811&stnIds=114'
#
# url = url + key + param
# response = requests.get(url)
#
# # JSON 파싱
# dic = json.loads(response.text)
# doc = dic['response']['body']['items']['item']
# data = pd.DataFrame(doc)
#
# data = auto_convert_df_types(
#     data,
#     datetime_cols=['tm'],        # 날짜 컬럼
#     int_cols=['stnId', 'minTaHrmt'],  # 반드시 정수로 쓰고 싶은 컬럼
#     na_fill_for_int=0            # NaN은 0으로 채움
# )
#
# data.info()

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.ie.webdriver import WebDriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
driver = webdriver.Chrome() #(service = Service(ChromeDriverManager().install())
driver.get('http://info.nec.go.kr/main/showDocument.xhtml?electionId=0000000000&topMenuId=VC&secondMenuId=VCCP09')
driver.find_element(By.ID, 'electionType1').click()
element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'electionName')))
driver.find_element(By.ID, 'electionName').send_keys(' 제19대')
driver.find_element(By.ID, 'electionName').click()


WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'electionCode')))

driver.find_element(By.ID, 'electionCode').send_keys('대통령선거')

sido_list_raw = driver.find_element(By.XPATH, '''//*[@id="cityCode"]''')
sido_list = sido_list_raw.find_elements(By.TAG_NAME, 'option')
sido_names_values = [option.text for option in sido_list][2:]

def move_sido(name):
    element = driver.find_element(By.ID, 'cityCode')
    element.send_keys(name)
    WebDriverWait(driver, 10).\
        until(EC.presence_of_element_located((By.ID, 'searchBtn')))
    driver.find_element(By.ID, 'searchBtn').click()

def get_num(tmp):
    return float(tmp.replace(',', ''))

# 시작되는 코드들이 tr / td 로 시작된다.
cells = driver.find_elements(By.TAG_NAME, 'td')


def append_data(rows, sido_name, data):

    for i, row in enumerate(rows[5:], start = 2): #이렇게 할 경우 ,데이터가 2번 부터 시작함
        if i % 2 == 0:
            cells = row.find_elements(By.TAG_NAME, 'td')

            data['광역시도'].append(sido_name)
            data['시군'].append(cells[0].text.strip())
            data['전체'].append(get_num(cells[2].text.strip()))
            data['문'].append(get_num(cells[3].text.strip()))
            data['홍'].append(get_num(cells[4].text.strip()))
            data['안'].append(get_num(cells[5].text.strip()))

election_result_raw= {'광역시도' : [], '시군': [], '전체': [],
                      '문': [], '홍':[], '안':[]}

for each_sido in sido_names_values:
    move_sido(each_sido)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located(By.TAG_NAME, 'table'))
    table = driver.find_element(By.TAG_NAME, 'table')
    rows = table.find_elements(By.TAG_NAME, 'tr')
    append_data(rows, each_sido, election_result_raw)

import pandas as pd
election_result = pd.DataFrame(election_result_raw)
a = 4