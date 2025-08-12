import pandas as pd
import requests
import json
import numpy as np

def auto_convert_df_types(df, datetime_cols=None, int_cols=None, na_fill_for_int=None):
    """
    DataFrame 컬럼 타입을 자동 변환하는 함수.
    - 모든 object/문자열 컬럼에 대해 숫자 변환 가능하면 숫자로, 날짜 컬럼은 날짜로 변환
    - int 컬럼은 NaN이 있으면 Int64 (nullable int)로 변환
    - 필요 시 특정 int 컬럼은 NaN을 0으로 채운 뒤 int 변환

    Parameters:
        df (pd.DataFrame): 변환 대상 DataFrame
        datetime_cols (list): 날짜로 변환할 컬럼 목록 (자동 감지 대신 지정)
        int_cols (list): 정수(int)로 변환할 컬럼 목록
        na_fill_for_int (int/float): int 변환 시 NaN 대체 값 (예: 0)
    """

    df = df.copy()
    # 1. 빈 문자열을 NaN으로 처리
    df.replace('', np.nan, inplace=True)

    # 2. 날짜 변환 (사용자가 지정한 경우)
    if datetime_cols:
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3. 숫자 변환 가능한 컬럼 처리
    for col in df.columns:
        if col in (datetime_cols or []):  # 날짜 컬럼은 건너뛰기
            continue
        if df[col].dtype == 'object':
            converted = pd.to_numeric(df[col], errors='coerce')
            # 변환 결과에서 NaN이 아닌 값이 1개 이상이면 숫자 컬럼으로 처리
            if converted.notna().sum() > 0:
                df[col] = converted

    # 4. 지정된 int 컬럼 처리
    if int_cols:
        for col in int_cols:
            if na_fill_for_int is not None:
                df[col] = df[col].fillna(na_fill_for_int).astype(int)
            else:
                df[col] = df[col].astype('Int64')  # NaN 허용 정수 타입

    return df

url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList?'
key = 'serviceKey=amZyo3vOJtsNCiW0LO9Gha12rCV83HHteCYsvVOT35lyuvbBLEqQFfy%2BOfAvC4wZrR8KNeQ%2B4lDz4V4nxqEfFA%3D%3D'
param = '&dataType=JSON&numOfRows)=10&pageNo=1&dataCd=ASOS&dateCd=DAY&startDt=20250801&endDt=20250811&stnIds=114'

url = url + key + param
response = requests.get(url)

# JSON 파싱
dic = json.loads(response.text)
doc = dic['response']['body']['items']['item']
data = pd.DataFrame(doc)

data = auto_convert_df_types(
    data,
    datetime_cols=['tm'],        # 날짜 컬럼
    int_cols=['stnId', 'minTaHrmt'],  # 반드시 정수로 쓰고 싶은 컬럼
    na_fill_for_int=0            # NaN은 0으로 채움
)

data.info()

a = 4