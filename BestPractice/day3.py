랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators, max_depth,
                            n_jobs=-1, random_state)
rf.fit(독립_train, 종속_train)
y_train_pred = rf.predict(독립_train)
y_test_pred = rf.predict(독립_test)

rf.feature_importances_

import seaborn as sns
df = sns.load_dataset('titanic')
df.info()
종속변수: survived
독립변수: sex(문자), age(숫자, 결측), fare(숫자), 
         class(문자)

import pandas as pd
df = pd.get_dummies(df, columns = ['sex', 'class'],
                    dtype = 'int', drop_first = True)
df['age'].fillna(df['age'].mean(), inplace=True)

y = df['survived']
X = df[['age', 'fare', 'sex_male', 
        'class_Second', 'class_Third']]

Hold-Out
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                 X, y,
                 test_size = 0.3,
                 random_state = 0,
                 stratify = y)
y_train.value_counts()
Out[16]: 
survived
0    384
1    239
# resampling - SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors = 5, random_state = 0,
           n_jobs = -1)
X_train_re, y_train_re = sm.fit_resample(X_train, 
                                         y_train)
y_train_re.value_counts()
Out[15]: 
survived
0    384
1    384

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100,
                            max_depth = 3,
                            random_state = 0,
                            n_jobs = -1)
rf.fit(X_train_re, y_train_re)
y_train_pred = rf.predict(X_train_re)
y_test_pred = rf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_train_re, y_train_pred))
print(classification_report(y_test, y_test_pred))

cross-validation
Hold-out + KFold
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test

from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator, param_grid, scoring, cv)
estimator: 적용하고자하는 모델 함수(주요파라미터 제외)
예) RandomForestClassifier(random_state=0, n_jobs=-1)
param_grid : 찾고자 하는 파라미터 값들을 입력
예) {'n_estimators':[50, 100, 200, 300], 
    'max_depth':[2, 3, 4, 5]}
scoring : 최적파라미터 결정 기준
=> 'accuracy', 'f1', 'precision', 'recall', 'roc_auc'
cv : kfold의 갯수 (랜덤하게)
 => StratifiedKFold, KFold의 객체를 입력
예) cv = kfold
gs.fit(X_train, y_train)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
kfold = KFold(n_splits= 분할갯수, 
              shuffle = False/True,
              random_state = 난수번호)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
grid = {'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6]}
gs = GridSearchCV(estimator=RandomForestClassifier(random_state=0,
                                                n_jobs=-1), 
             param_grid = grid,
             scoring = 'accuracy', cv = 5)
gs.fit(X_train_re, y_train_re)
gs.best_params_
Out[37]: {'max_depth': 5, 'n_estimators': 300}
rf1 = gs.best_estimator_
=> RandomForestClassifier(max_depth=5, n_estimators=300,
                          random_state = 0, n_jobs=-1)
rf1.fit(X_train_re, y_train_re)
y_train_pred1 = rf1.predict(X_train_re)
y_test_pred1 = rf1.predict(X_test)


KFold + KFold => 중첩교차검증

from sklearn.model_selection import GridSearchCV
=> 파라미터 찾기(inner loop 역할)
from sklearn.model_selection import cross_val_score
=> 테스트 결과 확인(outer loop 역할)
cross_val_score(estimator= GridSearch 객체,
                독립_full, 종속_full,
                scoring = 'accuracy',
                cv = kfold 갯수)

grid = {'max_depth': [4, 5, 6, 7],
        'n_estimators': [100, 200, 300, 400]}
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator=RandomForestClassifier(
    random_state = 0, n_jobs=-1), 
    param_grid = grid, cv = 5)

from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = gs,
                        X=X, y=y,
                        scoring = 'accuracy',
                        cv = 5, n_jobs = -1)
# Out[54]: array([0.81564246, 0.83146067, 0.8258427 , 0.81460674, 0.84831461])
import numpy as np
np.mean(score)
np.std(score)

from sklearn.model_selection import cross_validate
score = cross_validate(estimator = gs,
                        X=X, y=y,
                        scoring = 'accuracy',
                        cv = 5, n_jobs = -1, 
                        return_estimator=True)
for i, est in enumerate(score['estimator']):
    print(i, est.best_params_)
0 {'max_depth': 7, 'n_estimators': 100}
1 {'max_depth': 7, 'n_estimators': 300}
2 {'max_depth': 4, 'n_estimators': 100}
3 {'max_depth': 7, 'n_estimators': 300}
4 {'max_depth': 7, 'n_estimators': 100}

AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(
    estimator = DecisionTreeClassifier(max_depth = 1),
    n_estimators = 학습횟수, learning_rate = 학습률)
learning_rate : 학습률 ( 0 ~ 1 ) 작을 수록 학습을 더디게
n_estimators : 학습횟수, 학습률이 낮으면 학습횟수를 늘려야 함

ada.fit(독립_train, 종속_train)
y_train_pred = ada.predict(독립_train)
y_test_pred = ada.predict(독립_test)

X_train_re, X_test, y_train_re, y_test
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(estimator = None, 
                         n_estimators = 100,
                         learning_rate = 0.1, 
                         random_state = 0)
ada.fit(X_train_re, y_train_re)
y_train_pred = ada.predict(X_train_re)
y_test_pred = ada.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_train_re, y_train_pred))
print(classification_report(y_test, y_test_pred))

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier as GBC
grid = {'learning_rate': [0.001, 0.01, 0.1, 1],
        'n_estimators': [300, 400, 500, 600]}
gs = GridSearchCV(estimator = GBC(random_state = 0),
             param_grid = grid,
             scoring = 'accuracy', cv = 5)
gs.fit(X_train_re, y_train_re)
gs.best_params_
Out[77]: {'learning_rate': 0.01, 'n_estimators': 500}
gbm = gs.best_estimator_
gbm.fit(X_train_re, y_train_re)
y_train_pred = gbm.predict(X_train_re)
y_test_pred = gbm.predict(X_test)
print(classification_report(y_train_re, y_train_pred))
print(classification_report(y_test, y_test_pred))

XGBoost
pip install xgboost
from xgboost import XGBClassifier
xg = XGBClassifier(n_estimators = 100,
                   eta = 0.3,
                   max_depth = 3,
                   reg_lambda = 0.7,
                   reg_alpha = 0.3,
                   random_state = 0)
xg.fit(X_train_re, y_train_re)
y_train_pred = xg.predict(X_train_re)
y_test_pred = xg.predict(X_test)

print(classification_report(y_train_re, y_train_pred))
print(classification_report(y_test, y_test_pred))

실습
1)데이터 구축
# 1. dataset.xlsx 자료 불러오기(변수 없는 부분 삭제)
pd.read_excel('file.xlsx')
# 2. 20130601~20130831 시간당 기상 자료 불러오기
# 3. dataset과 기상자료(부산) 결합(변수추가방법)
pd.merge(df_left, df_right, how = , left_on,
         right_on)
# 4. 최종 데이터셋: 20130601~20130831 시간당 자료
# => 변수: 날짜 및 시간, 날씨, 사고여부(1:사고, 0: 미사고)
# 2)전처리
# 기상자료부분 => 실수 및 정수, 날짜유형
# 문자형 => 삭제
# 결측자료 => KNN imputation 이용(train, test 진행 후)
# 3) hold-out + kfold=> 파라미터 튜닝, 검증자료결과 확인
# 모델: randomforest

import pandas as pd
df = pd.read_excel('dataset.xlsx')
df.info()
df['발생일시'] = pd.to_datetime(df['발생일시'])

url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?'
key = 'serviceKey=4Fs6k7F3R5MpSUXwJG5kJkI95Z7Q66N8HeCFaR5xVC0R%2FxcjPxbPLNCOe1TQYyIY6nCVhsdaTykCtdwY%2FOAn6g%3D%3D'
param = '&dataType=JSON&numOfRows=999&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=159&endDt=20130831&endHh=23&startHh=00&startDt=20130601'
url = url + key  + param

import requests
response = requests.get(url)

import json
dic = json.loads(response.text)
ros = dic['response']['body']['items']['item']
df1 = pd.DataFrame(ros)
import numpy as np
df1.replace('', np.nan, inplace = True)

for i in df1.columns:
    try:
        df1[i] = df1[i].astype('int')
    except:
        try:
            df1[i] = df1[i].astype('float')
        except:
            continue
df1['tm'] = pd.to_datetime(df1['tm'])

df1.info()

df1.dropna(axis = 1, thresh = df1.shape[0]/2,
           inplace = True)

data = pd.merge(df, df1, how = 'right',
         left_on = '발생일시', right_on = 'tm')

data['사고여부'].replace(np.nan, 0, inplace = True)
data.drop(['발생일시', 'tm', 'stnNm', 
           'stnId', 'clfmAbbrCd'], 
          axis = 1, inplace = True)

y = data['사고여부']
X = data.drop('사고여부', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                 stratify= y, random_state = 0)

from sklearn.impute import KNNImputer
ki = KNNImputer(n_neighbors=5)
ki.fit(X_train)
X_train_imp = ki.transform(X_train)
X_test_imp = ki.transform(X_test)

X_train_imp = pd.DataFrame(X_train_imp, columns = X_train.columns)
X_test_imp = pd.DataFrame(X_test_imp, columns = X_test.columns)

from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors=5)
X_train_re, y_train_re = sm.fit_resample(X_train_imp, y_train)

from sklearn.model_selection import GridSearchCV
grid = {'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5]}
gs = GridSearchCV(estimator = RandomForestClassifier(random_state = 0, n_jobs = -1), 
             param_grid = grid)
gs.fit(X_train_re)
rf = gs.best_estimator_
rf.fit(X_train_re)
rf.predict(X_train_re)
re.predict(X_test)