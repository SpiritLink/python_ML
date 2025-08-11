import pandas
import seaborn as sns

df = sns.load_dataset('mpg')
df.drop(['horsepower', 'origin', 'name'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, random_state=0)

print(train.shape)
print(test.shape)

# Scaling 데이터가 2차원 Array 형태로 변경된다.
from sklearn.preprocessing import StandardScaler

# 평균0, 표준편차 1이 되는 형태 (둘다 True 일 경우)
sc = StandardScaler(with_mean=True, with_std=True)
sc.fit(train)  # 평균, 표준편차 계산
# 주의 : 입력되는 데이터는 반드시 2차원 어레이 형태여야 한다. (데이터 프레임)
# df['mpg'] 시리즈 형태의 데이터
# df[['mpg']] 데이터프레임 (2차원 형태)의 데이터
train_sc = sc.transform(train)
test_sc = sc.transform(test)

import pandas as pd

train_sc = pd.DataFrame(train_sc, columns=train.columns)
test_sc = pd.DataFrame(test_sc, columns=test.columns)

sc = StandardScaler(with_mean=True, with_std=True)
sc.fit(train[['displacement', 'weight']])
train[['dis_sc', 'wei_sc']] = sc.transform(train[['displacement', 'weight']])

# Simple Imputer (데이터 결합)
import seaborn as sns

df = sns.load_dataset("titanic")
df.info()

df_num = df.select_dtypes(include='number')  # 선택하고자 하는 유형
df_cat = df.select_dtypes(include='object')  # 제외하고자 하는 유형

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')
si.fit(df_num)  # 식 완성, train 자료 입력
si.transform(df_num) # 계산 완성, 결과는 2차원 형태
df_num = pd.DataFrame(si.transform(df_num), columns = df_num.columns)

from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy = 'most_frequent')
si.fit(df_cat)
df_cat = pd.DataFrame(si.transform(df_cat), columns = df_cat.columns)
print(df_cat)
df_si = pd.concat([df_num, df_cat], axis = 1)
print(df_si)

# 주어진 자료가 문자든, 숫자든 상관없이 다 진행되었다.
# 자료를 변환하기전에 채울 수 있는 방식이라 편하다.