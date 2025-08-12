import seaborn as sns

df = sns.load_dataset('mpg')
df.info()
df.drop('name', axis=1, inplace=True)
print(df.origin)

# 문자 이기 때문에, 거리 측정이 불가능하다, 더미 변수를 활용해야 한다.
import pandas as pd
df1 = pd.get_dummies(df, columns = ['origin'], dtype = 'int', drop_first = True)
print(df1)

from sklearn.impute import KNNImputer
ki = KNNImputer(n_neighbors=5)
ki.fit(df1) # impute 할때는, train / test 데이터를 나누고 train 만 만들어야 한다.
filled_array = ki.transform(df1)

# transform 의 결과로 filled_array 가 반환되는데 이를 이용해서 결측값이 채워진 값으로 대체한다.
df1 = pd.DataFrame(ki.transform(df1), columns = df1.columns)

# 주성분 분석 (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, random_state = 0)
pca.fit(df1)
# 데이터 형태 : 2d array, DataFrame 형태
# 독립 데이터만 입력 (종속 데이터 제외)
# train 자료를 대입해서 관계식 형성
pca.transform(df1)
# 2d array 형태로 주어진 주성분 값 계산 출력
# DataFrame 형태로 변환 필요

df2 = pd.DataFrame(pca.transform(df1), columns = ['PC1', 'PC2'])

from sklearn.decomposition import PCA
pca = PCA(n_components = df1.shape[1], random_state=0)
pca.fit(df1)
pca.transform(df1)
print(pca.explained_variance_ratio_)

variance = pca.explained_variance_ratio_
import matplotlib.pyplot as plt
plt.plot(list(range(1, 10)), variance, ls = '--', marker = 'o', mfc = 'red')
# plt.show()
a = 4
