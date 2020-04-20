import pandas as pd  # pandas is a dataframe library
import numpy as np
from sklearn.preprocessing import MinMaxScaler

PAST_DATA_SAMPLE_NUMBER = 20
FUTURE_DATA_SAMPLE_NUMBER = 9

# generate new columns names
def generate_columns_names(nr, type='inlet_server_past'):
    new_columns_names = []
    for i in range(1, nr + 1):
        new_columns_names.append(type + '_' + str(i))

    return new_columns_names


def create_past_data_columns(df, nr, df_column_name, new_columns_names):
    column_to_process = df[df_column_name].values
    result = {}

    for i in range(0, nr):
        shifted_array = np.roll(column_to_process, i + 1)
        result[new_columns_names[i]] = list(shifted_array)

    return result


def create_future_data_columns(df, nr, df_column_name, new_columns_names):
    column_to_process = df[df_column_name].values
    result = {}

    for i in range(0, nr):
        shifted_array = np.roll(column_to_process, -(i + 1))
        result[new_columns_names[i]] = list(shifted_array)

    return result

# display all columns
pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = pd.read_csv("./data/data.log")

col_names = ['t_room_initial', 't_air_input', 'air_flow', 'heat_generation_rate_server_1',
             'heat_generation_rate_server_2', 'inlet_probe_1_rack_1']

# drop the other columns
df.drop(df.columns.difference(col_names), 1, inplace=True)

# create the past feature data columns
past_columns_names = generate_columns_names(PAST_DATA_SAMPLE_NUMBER)
past_columns = create_past_data_columns(df, PAST_DATA_SAMPLE_NUMBER, col_names[5], past_columns_names)

# create the future output data columns
future_columns_names = generate_columns_names(FUTURE_DATA_SAMPLE_NUMBER, type='inlet_server_future')
future_columns = create_future_data_columns(df, FUTURE_DATA_SAMPLE_NUMBER, col_names[5], future_columns_names)

# create the new df with past and future data
past_df = pd.DataFrame(data=past_columns)
future_df = pd.DataFrame(data=future_columns)

dataset = pd.concat([past_df, df, future_df], axis=1)

# remove the rows with shifted data - first and last
dataset.drop(df.head(PAST_DATA_SAMPLE_NUMBER).index, inplace=True)  # drop first n rows
dataset.drop(df.tail(FUTURE_DATA_SAMPLE_NUMBER).index, inplace=True)  # drop last n rows

print(dataset.shape)

dataset.to_csv("./data/ten_seconds_data/inlet_probe_1_rack_1.csv", index=False)
print("done")
