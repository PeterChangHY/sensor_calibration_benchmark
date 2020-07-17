#!/usr/bin/env python
"""this module cotains the method to match two datas using their timestamp"""
import pandas as pd


def match_time_and_fill(test_data_frame, target_data_frame, match_time_tolerance_in_second, default_value=False):
    """ sample the test_data with target_data's timestamp, fill the test_data if matched otherwise fill the default value
    """

    # matched_data_list = []
    print(test_data_frame['Parsed_value'])
    print(target_data_frame['Parsed_value'])

    """
    because I know the time is exactly the same so hack here

    """

    matched_data_frame = test_data_frame
    matched_data_frame = matched_data_frame.drop(columns=['Raw_value'])
    matched_data_frame = matched_data_frame.rename(index=str, columns={"Parsed_value": "test_value"})
    target_data_frame = target_data_frame.drop(columns=['Raw_value', 'Unix_Time'])
    target_data_frame = target_data_frame.rename(index=str, columns={"Parsed_value": "target_value"})
    #matched_data_frame = matched_data_frame.merge(target_data_frame, left_on='test_value', right_on='target_value')
    matched_data_frame = pd.concat([matched_data_frame, target_data_frame], axis=1, sort=False)
    print("matched_data_frame")
    print(matched_data_frame)

    """
    for _, target_row in target_data_frame.iterrows():
        target_value = target_row['Parsed_value']
        sample_time = target_row['Unix_Time']
        time_diff_df = (test_data_frame['Unix_Time'] - sample_time).abs()
        sorted_indices = time_diff_df.argsort()
        if len(time_diff_df) > 0 and time_diff_df[sorted_indices[0]] < match_time_tolerance_in_second:
            matched_data_list.append([sample_time, test_data_frame.iloc[sorted_indices[0]]['Parsed_value'], target_value])
        else:
            matched_data_list.append([sample_time, default_value, target_value])
    """
    # matched_data_frame = pd.DataFrame(matched_data_list, columns=['Unix_Time', 'test_value', 'target_value'])
    return matched_data_frame
