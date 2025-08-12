# 실행: F9
# import 라이브러리 as 별칭

import seaborn as sns
df = sns.load_dataset('titanic')
# 결측여부 확인
df.shape[0] - df.count()
df.isnull().sum()

# Hold-Out 방식
# train : test = 비율1 : 비율2
# from sklearn.model_selection import train_test_split
# train_test_split(데이터1, 데이터2,
#                  test_size = 비율,
#                  shuffle = True(무작위)/False(차례로),
#                  random_state = 난수번호,
#                  stratify = 데이터['층화변수'])
# => [데이터1 train, 데이터1 test, 데이터2 train, 데이터2 test]
# => 각각의 데이터를 객체명에 할당
# X_train, X_test, y_train, y_test

# K-Fold 방식
# from sklearn.model_selection import KFold
# 객체 = KFold(n_splits = 분할갯수, 
#            shuffle = False(차례로)/True(무작위),
#            random_state = 난수번호)
# 객체.split(데이터)
# for train_idx, test_idx in 객체.split(데이터):
#     train, test = df.iloc[train_idx], df.iloc[test_idx]
    

from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5, 
              shuffle = True, 
              random_state = 0)
kfold.split(df)
# Out[16]: <generator object _BaseKFold.split at 0x0000025BD1704A60>

for train_idx, test_idx in kfold.split(df):
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    print(train.index) # train의 index 번호
    print(test.index) # test의 index 번호


# 불균형 데이터 처리
# 라이브러리 설치
# pip install imbalanced-learn

# 주의!! sklearn 라이브러리의 버전 
# pip install scikit-learn==1.2.2

import seaborn as sns
df = sns.load_dataset('titanic')
df['survived'].value_counts()

y = df['survived']
X = df.drop('survived', axis = 1)

# 샘플링 - 언더샘플링
# from imblearn.under_sampling import RandomUnderSampler, TomekLinks
# rus = RandomUnderSampler(random_state= 0)
# tl = TomekLinks()
# rus.fit_resample(X, y)
# tl.fit_resample(X, y)
# => 출력결과: X_resampled, y_resampled

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state= 0)
rus.fit_resample(X, y)

# 주의!! 주어진 자료에 문자가 있으면 안됨
#        결측이 있어도 상관없음

y = df['survived']
X = df[['pclass', 'age', 'fare']]
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state= 0)
X_resampled, y_resampled = rus.fit_resample(X, y)

y.value_counts()
y_resampled.value_counts()

# 언더샘플링 - TomekLinks
# 주의!! 결측이 있으면 안됨

y = df['survived']
X = df[['pclass', 'fare']]
from imblearn.under_sampling import TomekLinks
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X, y)
y_resampled.value_counts()
# Out[34]: 
# survived
# 0    547
# 1    342

# 오버샘플링
# from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
# ros = RandomOverSampler(random_state = 0)
# sm = SMOTE(k_neighbors=갯수, random_state = 0)
# bsm = BorderlineSMOTE(k_neighbors=갯수, random_state = 0)
# ros.fit_resample(X, y)
# sm.fit_resample(X, y)
# bsm.fit_resample(X, y)
# => 출력결과: X_resampled, y_resampled

# 오버샘플링 - 랜덤오버샘플링
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state = 0)
X_resampled, y_resampled = ros.fit_resample(X, y)
y_resampled.value_counts()
# Out[41]: 
# survived
# 0    549
# 1    549

# 오버샘플링 - SMOTE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
sm = SMOTE(k_neighbors=5, random_state = 0)
X_resampled, y_resampled = sm.fit_resample(X, y)
y_resampled.value_counts()

# 실습. 
# 종속변수: survived
# 독립변수: sex(문자, 범주, 'male'), age(연속, 결측), 
#          fare(연속), class(문자, 범주, 'Third')
# => 결측(데이터.dropna()), 문자 처리(pd.get_dummies())
# 데이터.dropna(inplace=True)
data = df[['survived', 'sex', 'age', 'fare', 'class']]
data.dropna(inplace = True)
data.info()

# pd.get_dummies(데이터, columns = ['var'], dtype='int').drop(['var'], axis=1)
import pandas as pd
data = pd.get_dummies(data, columns = ['sex', 'class'], 
               dtype = 'int').drop(['sex_male', 'class_Third'], axis = 1)

# 독립변수, 종속변수 지정
y = data['survived']
X = data.drop('survived', axis = 1)

# Hold-Out 으로 train:test = 7:3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                 test_size = 0.3, random_state = 0,
                 stratify = y)

# 오버샘플링(SMOTE)
from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors = 5, random_state = 0)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)


# 모델링: raw data와 resampled data
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty=None, max_iter=1000)
lr.fit(X_resampled, y_resampled)
y_test_pred1 = lr.predict(X_test)
y_test_prob1 = lr.predict_proba(X_test)[:, 1]
lr.fit(X_train, y_train)
y_test_pred2 = lr.predict(X_test)
y_test_prob2 = lr.predict_proba(X_test)[:, 1]

# 모델링 결과 확인
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred1))
print(classification_report(y_test, y_test_pred2))

# roc 곡선
# from sklearn.metrics import roc_curve, auc
# roc_curve(실제값, 예측확률)
# => 출력결과: [fpr, tpr, threshold]


from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_test, y_test_prob1)

# 결과 DataFrame으로 만들기
dic= {'0_reacll': 1-fpr,
      '1_recall': tpr,
      'thresh': threshold}
roc = pd.DataFrame(dic)

# ROC 곡선 그리기
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, ls = '--', c = 'red')
plt.ylabel('FPR')
plt.xlabel('TPR')
plt.title("ROC Curve")


# 자료의 표준화 - 자료는 연속형
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler(with_mean = True, with_std = True)
# sc.fit(train)
# 주의!! 입력되는 데이터는 2차원 형태(데이터프레임)
# df['mpg'] # 시리즈형태의 데이터
# df[['mpg']] # 데이터프레임(2차원 형태)의 데이터
# train_sc = sc.transform(train)
# test_sc = sc.transform(test)
# => 변환된 결과는 2d array 형태로 출력
# => DataFrame으로 변환 필요

import seaborn as sns
df = sns.load_dataset('mpg')
df.drop(['horsepower', 'origin', 'name'], axis = 1,
        inplace = True)

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.3,
                               random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean = True, with_std = True)
sc.fit(train)
train_sc = sc.transform(train)
test_sc = sc.transform(test)

import pandas as pd
train_sc = pd.DataFrame(train_sc, columns = train.columns)
test_sc = pd.DataFrame(test_sc, columns = test.columns)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean = True, with_std = True)
sc.fit(train[['displacement', 'weight']])
train[['dis_sc', 'wei_sc']] = sc.transform(train[['displacement', 'weight']])
test[['dis_sc', 'wei_sc']] = sc.transform(test[['displacement', 'weight']])


# 대체(imputation) - SimpleImputer
# from sklearn.impute import SimpleImputer
# si = SimpleImputer(strategy = 'mean'/'median'/'most_frequent')
# si.fit(df_num) # 식 완성, train 자료 입력
# si.transform(df_num) # 계산 완성, 결과는 2차원 형태
# df_num = pd.DataFrame(si.transform(df_num), 
#                       columns = df_num.columns)

import seaborn as sns
df = sns.load_dataset('titanic')
df.info()

df_num = df.select_dtypes(include = 'number')
df_cat = df.select_dtypes(include = 'object')

from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy = 'mean')
si.fit(df_num) # 식 완성, train 자료 입력
si.transform(df_num) # 계산 완성, 결과는 2차원 형태
df_num = pd.DataFrame(si.transform(df_num), 
                      columns = df_num.columns)

from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy = 'most_frequent')
si.fit(df_cat)
df_cat = pd.DataFrame(si.transform(df_cat),
                      columns = df_cat.columns)

df_si = pd.concat([df_num, df_cat], axis = 1)






