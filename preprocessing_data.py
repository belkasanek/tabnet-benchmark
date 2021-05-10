import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# to have same structure as train
df_test['target'] = 0

CATEGORICAL_COLUMNS = [col for col in df.columns if col.endswith('bin') or col.endswith('cat')]
NUMERIC_COLUMNS = [col for col in df.columns if  not col.endswith('bin') and not col.endswith('cat')]

# missing values mark as -1
for col in CATEGORICAL_COLUMNS:
    l_enc = LabelEncoder()
    df[col] = l_enc.fit_transform(df[col].values)
    df_test[col] = l_enc.transform(df_test[col].values)
    
df.to_csv('data/processed_train.csv', index=False)
df_test.to_csv('data/processed_test.csv', index=False)