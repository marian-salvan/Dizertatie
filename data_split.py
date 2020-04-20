import pandas as pd
import matplotlib.pyplot as plt

TRAIN_DATA_PATH = "./data/unchanged_data/train/data.csv"
TEST_DATA_PATH = "./data/unchanged_data/test/data.csv"

df = pd.read_csv("./data/data.log")
df_row_nr = len(df.index)

train_df = pd.DataFrame()
test_df = pd.DataFrame()

print(list(df.columns.values))

for i in range(0, int(df_row_nr / 600)):
    temp_df = df.head(600)
    temp_train_df = temp_df.head(480)
    temp_test_df = temp_df.tail(120)

    train_df = train_df.append(temp_train_df, ignore_index=True)
    test_df = test_df.append(temp_test_df, ignore_index=True)

    df = df.iloc[600:]  # delete first 600 rows
    print(i)

print(train_df.shape)
print(test_df.shape)

train_df.to_csv(TRAIN_DATA_PATH, index=False)
test_df.to_csv(TEST_DATA_PATH, index=False)
print('done...')

