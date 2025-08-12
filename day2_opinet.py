import time
import xlrd
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
driver = webdriver.Chrome() #(service = Service(ChromeDriverManager().install())

# 주유소 기름값 조회 사이트
driver.get('https://www.opinet.co.kr/searRgSelect.do')
sido_select_element = driver.find_element(By.ID, 'SIDO_NM0')
Select(sido_select_element).select_by_visible_text('서울')

gu_list_raw = driver.find_element(By.XPATH, '''//*[@id="SIGUNGU_NM0"]''')
gu_list = gu_list_raw.find_elements(By.TAG_NAME, 'option')
gu_names = [option.text for option in gu_list][1:]
# 지역 서울 선택, 구 이름 목록 추출
# 조회해서 나오면, 엑셀 저장 클릭

# for gu in gu_names:
#     element = driver.find_element(By.ID, 'SIGUNGU_NM0')
#     element.send_keys(gu)
#     time.sleep(2) #초
#     driver.find_element(By.ID, 'searRgSelect').click()
#     time.sleep(2)
#     element_get_excel = driver.find_element(By.XPATH, '''//*[@id="templ_list0"]/div[7]/div/a/span''').click()
#     time.sleep(2)

import pandas as pd

import os
os.chdir('C:/Users/USER/Downloads')
import glob
file_names = glob.glob('*주유소*.xls')

final_data = pd.DataFrame()

for name in file_names:
    raw_data = pd.read_excel(name, header = 2)
    final_data = pd.concat([final_data, raw_data], axis=0, join='outer')

final_data.replace('-', np.nan, inplace=True)

for var in final_data.columns:
    try:
        final_data[var] = final_data[var].astype('float')
    except:
        continue

a= 5;