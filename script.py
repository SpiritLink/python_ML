import pandas
import seaborn as sns

df = sns.load_dataset('titanic')
print(df.shape[0] - df.count())
print('---------------')
print(df.isnull().sum())

# 누락 데이터 치환 : 데이터.fillna
print('---------------')
print(pandas.get_dummies(df['pclass']))
print('---------------')
# labels = ['문자'] -> 더미 변수 유지

# python 에서 pandas 에서 데이터를 변경하는 명령어 사용시, 저장하진 않음
# 새로운 객체를 만들어서 저장해야함

# Hold-Out 방법
# train : test = 특정한 비율로 나누어 주는 방식, (비율 1 : 비율 2)

from sklearn.model_selection import train_test_split
# train_test_split(데이터1, 데이터2,
# test_size = 비율,
# shuffle = True(무작위) / False (차례로),
# random_state = 난수번호,
# stratify = 데이터['층화변수'] (종속변수))

# 출력 결과 데이터1 train, 데이터1 test, 데이터2 train, 데이터2 test
# 데이터 셋을 받을 수 있는 각각의 객체명을 할당해야함
# X_train, X_test, Y_train, Y_test 각각의 결과를 받을 수 있는 공간 할당

# 대부분 True 시계열 자료일 경우, 무작위로 추출하면 안된다.

from sklearn.model_selection import KFold

# 객체 = KFold(n_splits= 분할갯수,
#               shuffle = False(차례로) / True(무작위),
#               random_state = 난수번호)

# 차례대로 트레인 자료와 테스트 자료의 인덱스를 생성하는것
for train_idx, test_idx in KFold(n_splits=5, shuffle=True).split(df):
    train, test = df.iloc[train_idx], df.iloc[test_idx]

print(train)
print(test)

from sklearn.model_selection import KFold

kFold = KFold(n_splits=5, shuffle=True, random_state=0)
kFold.split(df)

for (train_idx, test_idx) in kFold.split(df):
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    # print(train.index) # train의 인덱스 번호
    # print(test.index) # test의 인덱스 번호

# LOOCV 방식 (Leave One Out 방식)

# Nested Cross 방식
# Train Data는 샘플링을 해도 되지만, Test data는 샘플링 하면 정확도가 안맞는 문제 발생

y = df['survived']
x = df.drop('survived', axis=1)

# data frame을 시각화 하기 위한 라이브러리, df를 표 형태로 콘솔에 출력한다.
from tabulate import tabulate

print(tabulate(df, headers='keys'))

# Random Under Sampling
# 주어진 자료에 문자가 있으면 안됨
# 결측이 있어도 상관 없다.
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0)
x_resampled, y_resampled = rus.fit_resample(x, y)
print(y.value_counts())
print(y_resampled.value_counts())

# Tomek Link
# 주의 ! 결측이 있으면 안된다.
from imblearn.under_sampling import TomekLinks

tl = TomekLinks()  # n_jobs=None 일 경우, CPU Core를 1개 사용하겠다는 의미
# x_resampled2, y_resampled2 = tl.fit_resample(x, y) #Input contains Nan 에러가 ㅂ라생함

# x_resampled, y_resampled = tl.fit_resample(train, test)
# print(x_resampled)
# print(y_resampled)
# y resampled 는 1대 1 비율로 만들어지지 않는다.

# 오버샘플링 - 랜덤 오버 샘플링
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
ros.fit_resample(x, y)
print(x_resampled.value_counts())
print(y_resampled.value_counts())

# 오버샘플링 - SMOTE (합성 데이터, KNN)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

sm = SMOTE(k_neighbors=5, random_state=0)  # 몇개의 데이터를 가지고, 합성 데이터를 만들것인지?
x_resampled, y_resampled = sm.fit_resample(x, y)
y_resampled.value_counts()