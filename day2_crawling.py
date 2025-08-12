import pandas as pd
import requests
import json

url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?'
key = 'serviceKey=amZyo3vOJtsNCiW0LO9Gha12rCV83HHteCYsvVOT35lyuvbBLEqQFfy%2BOfAvC4wZrR8KNeQ%2B4lDz4V4nxqEfFA%3D%3D'
param = '&dataType=JSON&numOfRows)=10&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20250811&endHh=23&startHh=01&startDt=20250811'

url = url + key + param
response = requests.get(url)
print(response.content)

dic = json.loads(response.text)
doc = dic['response']['body']['items']['item']
data = pd.DataFrame(doc)
data.info()

# 결측 데이터 0 처리
import numpy as np
data.replace('', np.nan, inplace=True)

data['tm'] = pd.to_datetime(data['tm'])
data['rnum'] = data['rnum'].astype('int')
data['stnId'] = data['stnId'].astype('float')
data['ta'] = data['ta'].astype('float')
data['ws'] = data['ws'].astype('float')
data['wd'] = data['wd'].astype('int')
data['hm'] = data['hm'].astype('int')
data['pv'] = data['pv'].astype('float')
data['td'] = data['td'].astype('float')
data['pa'] = data['pa'].astype('float')
data['ps'] = data['ps'].astype('float')
data['ss'] = data['ss'].astype('float')
data['ssQcflg'] = data['ssQcflg'].fillna(0).astype(int)
data['ssQcflg'] = data['ssQcflg'].astype('int')
data['icsr'] = data['icsr'].astype('float')
data['dc10Tca'] = data['dc10Tca'].fillna(0).astype(int)
data['dc10Tca'] = data['dc10Tca'].astype('int')

a = 4

