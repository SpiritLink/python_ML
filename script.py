import seaborn as sns
df = sns.load_dataset('titanic')
print(df.shape[0] - df.count())
print('---------------')
print(df.isnull().sum())
