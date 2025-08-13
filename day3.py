# 랜덤 포레스트
# 분류모델 : Classifier 연속자료예측 : Regressor
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)

# n_estimators	생성할 트리(결정나무)의 개수 (보통 100 이상 사용)
# max_depth	각 트리의 최대 깊이
# 두 파라미터 수치 조절을 잘 해야, 모델의 성능이 향상된다.

# rf.fit(독립_train, 종속_train)
# y_train_pred = rf.predict(독립데이터)
# y_test_pred = rf.predict(독립테스트)
#
# rf.feature_importances_

import seaborn as sns
df = sns.load_dataset('titanic')
df.info()
# 종속변수: survived (독립변수에 의해 결정되는 값)
# 독립변수: sex, age, fare, class (모델이 예측에 사용하는 입력 데이터, 원인이나 족너에 해당하는 변수)

import pandas as pd
df = pd.get_dummies(df, columns = ['sex', 'class'], dtype = 'int', drop_first = True)
df['age'].fillna(df['age'].mean(), inplace = True)

y = df['survived']
X = df[['age', 'fare', 'sex_male',
        'class_Second', 'class_Third']]

# Hold-Out 방법을 이용한 데이터 분리
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(
           X, y,
                 test_size = 0.3,
                 random_state = 0,
                 stratify = y)

y_train.value_counts()

# resampling - SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors = 5, random_state = 0,
      n_jobs = -1)

X_train_re, y_train_re = sm.fit_resample(X_train, y_train) #train 자료에 대해서만 resampling 해야한다.
y_train_re.value_counts()

rf.fit(X_train_re, y_train_re)
y_train_pred = rf.predict(X_train_re)
y_test_pred = rf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_train_re, y_train_pred))
print(classification_report(y_test_pred, y_test_pred))

# cross-validation
# Hold-out + KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold # 분류모델

from sklearn.model_selection import GridSearchCV
GridSearchCV(estimator, param_grid, scoring, cv)
estimator: 적용하고자 하는 모델 함수 (주요 파라미터 제외)
예) RandomForestClassifier(random_state=0, n_jobs=-1)
param_grid : 찾고자 하는 파라미터 입력
예) {'n_estimators':[50, 100, 200, 300], 'max_depth':[2, 3, 4, 5]}
scoring : 최적파라미터 결정 기준
=> 'accuracy', 'f1', 'precision', 'recall', 'roc_auc'

cv : kfold의 갯수 (랜덤하게)
gs.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
GridSearchCV(estimator = RandomForestClassifier(),
             param_grid = grid,
             scoring = 'accuracy',
             cv = 5)
gf.fit(X_train_re, y_train_re)
gs.best_params
gs.best_estimator
=> RandomForestClassifier(max_depth=5, n_estimators=300,
                          random_state = 0, n_jobs=-1)

rf1.fit(X_train_re, y_train_re)
y_train_pred1 = rf1.predict(X_train_re)
y_test_pred1 = rf1.predict(X_test)

# KFold + KFold => 중첩 교차 검증

from sklearn.model_selection import GridSearchCV
=> 파라미터 찾기(inner loop 역할)
from sklearn.model_selection import cross_val_score
=> 테스트 결과 확인(outer loop 역할)

cross_val_score(estimator= GridSearch 결과,
                독립_full, 종속_full,
                scoring='accuracy',
                cv= kfold 갯수)

gs = GridSearchCV(estimator=RandomForestClassifier(
    random_state = 0, n_jobs = -1),
    param_grid = grid, cv = 5)

from sklearn.model_selection import cross_val_score
cross_val_score(estimator = gs,
                X=X, y=y,
                scoring='accuracy',
                cv = 5, n_jobs = -1)

import numpy as np
np.mean(score)
np.std(score)

for i, est in enumerate(score['estimator']):
    print(i, est.best_params_)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 1),
                   n_estimators = 학습횟수, learning_rate = 학습률)
learning_rate : 학습률 (0 ~ 1) 작을 수록 학습을 더디게
# 몇회 반복할 것인가. learning_rate와 trade off
n_estimators : 학습횟수, 학습률이 낮으면 학습횟수를 늘려야 함

ada.fit(독립_train, 종속_train)
y_train_pred = ada.predict(독립_train)
y_test_pred = ada.predict(독립_test)

X_train_re, X_test, y_train_re, y_test
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(estimator = None,
                         n_estimators = 50,
                         learning_rate = 1,
                         random_state = 0)
ada.fit(X_train_re, y_train_re)
y_train_pred = ada.predict(X_train_re)
y_test_pred = ada.predict(X_test)

#깊이는 가능한 3이상 넘어가지 않게금 설정 (3 넘을 경우 과적합 발생)
# 시간이 오래 걸리는 부분을 보완한, 다른 부스팅 계열이 존재 (빅분기 경우 Adaboost 쓰면 떨어진다.) XG 부스트만 쓴다.
from sklearn.metrics import classification_report
classification_report(y_train_re, y_train_pred)
classification_report(y_test, y_test_pred)

## GBM
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
grid = {'learning_rate': [0.001, 0.01, 0.1, 1],
        'n_estimators': [300, 400, 500, 600]}

GridSearchCV(estimator = GBC(random_state = 0),
             param_grid = grid,
             scoring = 'accuracy', cv = 5)

gs.fit(X_train_re, y_train_re)
gs.best_params_
GBC(learning_rate = 0.1)
gbm = gs.best_estimator_
gbm.fit(X_train_re, y_train_re)
y_train_read = gbm.predict(X_train_re)
y_test_pred = gbm.predict(X_test)
print(classification_report(y_train_re, y_train_pred))
print(classification_report(y_test_re, y_test_pred))

# XG Boost (핵심)
from xgboost import XGBClassifier
XGBClassifier

XGBClassifier(n_estimators = 100,
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

# light gbm이 성능이 제일 좋게 나왔다. (빅분기 채점)
from lightgbm import LGBMClassifier
LGBMClassifier(boosting_type= 'gbdt')

