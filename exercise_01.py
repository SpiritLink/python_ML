# TODO 실습 (exercise 01)
# 종속변수 : survived
# 독립변수 : sex(문자, 범주, 'male'),
#           age(연속, 결측),
#           fare(연속),
#           class(문자, 범주, Third)
# => Hold-Out 으로 train:test = 7:3
# => 결측, 문자 전처리
# => 오버 샘플링 (SMOTE)
import pandas as pd
import seaborn as sns

# 종속변수 : survived
df = sns.load_dataset('titanic')

# 1. 결측, 문자 전처리 (sex가 문자로 처리되어 있다)
data = df[['survived', 'sex', 'age', 'fare', 'class']]
# 결측 자료 제거
data.dropna(inplace=True)
# 문자 전처리
data = pd.get_dummies(data, columns=['sex', 'class'], dtype='int').drop(['sex_male', 'class_Third'], axis=1)

from tabulate import tabulate

print(tabulate(data, headers='keys'))

# 종속변수
y = data['survived']
# 독립변수, 열 기준으로 삭제
x = data.drop('survived', axis=1)

# 2. Hold-Out 으로 train:test = 7:3
from sklearn.model_selection import train_test_split

# stratify : 종속 변수의 비율에 맞춰 train 과 test 가 나뉘어 진다.
X_train, X_test, Y_train, Y_test = train_test_split(x, y,
                                                    test_size=0.3, random_state=0,
                                                    stratify=y)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# 3. 오버 샘플링 (SMOTE)
from imblearn.over_sampling import SMOTE

# 몇개의 데이터를 가지고, 합성 데이터를 만들것인지?
sm = SMOTE(k_neighbors=5, random_state=0)
# test 자료가 아닌, Train 자료를 넣어야 하는것에 주의
x_resampled, y_resampled = sm.fit_resample(X_train, Y_train)
print(y_resampled.value_counts())

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty=None, max_iter=1000)
lr.fit(x_resampled, y_resampled)
y_test_pred1 = lr.predict(X_test)
lr.fit(X_train, Y_train)
y_test_pred2 = lr.predict(X_test)