import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# to have same structure as train
df_test['target'] = 0

CATEGORICAL_COLUMNS = [col for col in df.columns if col.endswith('bin') or col.endswith('cat')]
NUMERIC_COLUMNS = [col for col in df.columns if  not col.endswith('bin') and not col.endswith('cat')]

# missing values mark as -1
# add 2 so every categorical value start from 1
for col in CATEGORICAL_COLUMNS:
    df[col] = df[col] + 2
    df_test[col] = df_test[col] + 2
    
df.to_csv('data/processed_train.csv', index=False)
df_test.to_csv('data/processed_test.csv', index=False)