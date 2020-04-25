import pandas as pd  # pandas is a dataframe library
import numpy as np
from sklearn.preprocessing import MinMaxScaler

PAST_DATA_SAMPLE_NUMBER = 40
FUTURE_DATA_SAMPLE_NUMBER = 19

INLET_TEMPERATURES = ['inlet_probe_1_rack_1', 'inlet_probe_2_rack_1', 'inlet_probe_3_rack_1', 'inlet_probe_4_rack_1',
                      'inlet_probe_1_rack_2', 'inlet_probe_2_rack_2', 'inlet_probe_3_rack_2', 'inlet_probe_4_rack_2']
OUTLET_TEMPERATURES = ['outlet_probe_1_rack_1', 'outlet_probe_1_rack_2', 'outlet_probe_1_room',
                       'outlet_probe_2_room', 'outlet_probe_3_room', 'outlet_probe_4_room']


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


def create_one_second_dataset(column_list, type='inlet_server', folder="one_second_data"):
    past_data_sample_number = 4

    for column in column_list:
        # display all columns
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None

        df = pd.read_csv("./data/data.log")

        col_names = ['t_room_initial', 't_air_input', 'air_flow', 'heat_generation_rate_server_1',
                     'heat_generation_rate_server_2', column]

        # drop the other columns
        df.drop(df.columns.difference(col_names), 1, inplace=True)

        # create the past feature data columns
        past_columns_names = generate_columns_names(past_data_sample_number, type=type + '_past')
        past_columns = create_past_data_columns(df, past_data_sample_number, col_names[5], past_columns_names)

        # create the new df with past
        past_df = pd.DataFrame(data=past_columns)

        dataset = pd.concat([past_df, df], axis=1)

        # remove the rows with shifted data - first and last
        dataset.drop(df.head(past_data_sample_number).index, inplace=True)  # drop first n rows

        print(dataset.shape)

        dataset.to_csv("./data/" + folder + "/" + column + ".csv", index=False)
        print("done: " + column)


def create_dataset(column_list, type='inlet_server', folder="ten_seconds_data"):
    for column in column_list:
        # display all columns
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None

        df = pd.read_csv("./data/data.log")

        col_names = ['t_room_initial', 't_air_input', 'air_flow', 'heat_generation_rate_server_1',
                     'heat_generation_rate_server_2', column]

        # drop the other columns
        df.drop(df.columns.difference(col_names), 1, inplace=True)

        # create the past feature data columns
        past_columns_names = generate_columns_names(PAST_DATA_SAMPLE_NUMBER, type=type + '_past')
        past_columns = create_past_data_columns(df, PAST_DATA_SAMPLE_NUMBER, col_names[5], past_columns_names)

        # create the future output data columns
        future_columns_names = generate_columns_names(FUTURE_DATA_SAMPLE_NUMBER, type=type + '_future')
        future_columns = create_future_data_columns(df, FUTURE_DATA_SAMPLE_NUMBER, col_names[5], future_columns_names)

        # create the new df with past and future data
        past_df = pd.DataFrame(data=past_columns)
        future_df = pd.DataFrame(data=future_columns)

        dataset = pd.concat([past_df, df, future_df], axis=1)

        # remove the rows with shifted data - first and last
        dataset.drop(df.head(PAST_DATA_SAMPLE_NUMBER).index, inplace=True)  # drop first n rows
        dataset.drop(df.tail(FUTURE_DATA_SAMPLE_NUMBER).index, inplace=True)  # drop last n rows

        print(dataset.shape)

        dataset.to_csv("./data/" + folder + "/" + column + ".csv", index=False)
        print("done: " + column)


create_dataset(INLET_TEMPERATURES, type="inlet_server", folder="twenty_seconds_data")
# create_one_second_dataset(INLET_TEMPERATURES, type="inlet_server")
print("Done inlet")

create_dataset(OUTLET_TEMPERATURES, type="outlet_server", folder="twenty_seconds_data")
# create_one_second_dataset(OUTLET_TEMPERATURES, type="outlet_server")
print("Done outlet")
