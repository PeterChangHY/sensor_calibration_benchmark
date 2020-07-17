#!/usr/bin/env python
"""this module define the class of Data object """

import os

import pandas as pd
import numpy as np

# extract data from a tsv file and parse to useful value
#
# Input format(bag_name.tsv):
# Unix_Time \t Key \t \t Raw_value
#
# Output fromat:
# Data.bag: str
# Data.key: str
# Data.value_type: str('boolean' or  'scalar' or 'matrix')
# Data.data_frame: a panda data frame contain columns
# 1. Unix_Time                  float64
# 2. Raw_value                  object(string)
# 3. Parsed_value               object(boolean or scalar(float) or matrix(np.array))
# (optional)Datetime    datetime64[ns] //e.g 1970-01-01 00:00:00.000000000)


class Data(object):
    """ Data class that contains:
        bag: str
        key: str
        value_type: parsed_value type
        data_frame: panda data_frame
    """

    def __init__(self, bag=None, key=None, data_frame=None, value_type=None):
        self.bag = bag
        self.key = key
        self.data_frame = data_frame
        self.value_type = value_type


class DataCreatorInterface(object):
    """ Data object creator interface"""

    def create(self, bag, key, filename, adding_datetime=False):
        """ Create Data Object """
        raise NotImplementedError


class DataCreatorBase(DataCreatorInterface):
    """ The class provide the common function implement of Data class """

    def __init__(self):
        self._bag = None
        self._key = None
        self._data_frame = None
        self._value_type = None

    def create(self, key, filename, adding_datetime=False):
        self._check_file(filename)
        self._bag = os.path.splitext(os.path.basename(filename))[0]
        self._key = key
        self._set_value_type()
        valid = self._read_tsv_to_panda(filename)
        if adding_datetime:
            self._add_datetime()
        return Data(self._bag, self._key, self._data_frame, value_type=self._value_type), valid

    def _check_file(self, filename):
        if os.path.isfile(filename):
            return True
        else:
            raise ValueError('file: {} is not a file'.format(filename))

    def _read_tsv_to_panda(self, filename):
        column_names = ['Unix_Time', 'Key', 'Raw_value']
        self._data_frame = pd.read_csv(filename, header=0, names=column_names, sep='\t',
                                       dtype={'Unix_Time': np.float64,
                                              'Key': str,
                                              'Raw_value': str})
        self._get_key_slice()
        valid = self._parse_values()
        # sort by timestamps
        self._data_frame.sort_values('Unix_Time', inplace=True)
        return valid

    def _get_key_slice(self):
        # For meaningful (and simpler) comparison, it's easier to chop the df by the key
        self._data_frame = self._data_frame.loc[(self._data_frame['Key'] == self._key), [
            'Unix_Time', 'Raw_value']].reset_index(drop=True)

    def _parse_values(self):
        '''parse the df['Raw_value'] and create data['Parsed_value'] '''
        raise NotImplementedError

    def _add_datetime(self):
        self._data_frame['Datetime'] = pd.to_datetime(self._data_frame['Unix_Time'], origin='unix')

    def _set_value_type(self):
        raise NotImplementedError


class BooleanDataCreator(DataCreatorBase):
    """build boolean data """

    def _set_value_type(self):
        self._value_type = 'boolean'

    def _parse_values(self):
        parsed_values = []
        for _, row in self._data_frame.iterrows():
            value_in_string = str(row['Raw_value'])
            try:
                parsed_values.append(self._parse_string_to_boolean(value_in_string))
            except ValueError:
                return False
        self._data_frame['Parsed_value'] = parsed_values
        return True

    @ staticmethod
    def _parse_string_to_boolean(value_in_string):
        if value_in_string.lower() == 'true':
            return True
        elif value_in_string.lower() == 'false':
            return False
        else:
            raise ValueError(
                "Can not parse the value: \'{}\' to a boolean".format(value_in_string))


class ScalarDataCreator(DataCreatorBase):
    """create scalar data"""

    def _set_value_type(self):
        self._value_type = 'scalar'

    def _parse_values(self):
        parsed_values = []
        for _, row in self._data_frame.iterrows():
            value_in_string = str(row['Raw_value'])
            parsed_values.append(self._parse_string_to_scalar(value_in_string))
        self._data_frame['Parsed_value'] = parsed_values

    @ staticmethod
    def _parse_string_to_scalar(value_in_string):
        try:
            return float(value_in_string)
        except ValueError:
            print(
                "Can not parse the value: \'{}\' to a scalar(float)".format(value_in_string))


class MatrixDataCreator(DataCreatorBase):
    """ create matrix data """

    def _set_value_type(self):
        self._value_type = 'matrix'

    def _parse_values(self):
        parsed_values = []
        for _, row in self._data_frame.iterrows():
            value_in_string = str(row['Raw_value'])
            parsed_values.append(self._parse_string_to_matrix(value_in_string))
        self._data_frame['Parsed_value'] = parsed_values

    @ staticmethod
    def _parse_string_to_matrix(value_in_string):
        data_list = value_in_string.split(',')
        fail_convert_msg_base = 'Cannot parse the value: {} to a matrix'
        if len(data_list) < 3:
            raise ValueError(fail_convert_msg_base.format(value_in_string) +
                             ', need at least tree elements')
        row = int(data_list[0])
        col = int(data_list[1])
        try:
            matrix_data = [float(v) for v in data_list[2:]]
        except ValueError:
            raise ValueError(
                fail_convert_msg_base.format(value_in_string) +
                ', because I cannot convert \'{}\' to a scalar(float)'.format(v))
        try:
            return np.array([matrix_data]).reshape((row, col), order='C')
        except ValueError:
            raise ValueError(
                fail_convert_msg_base.format(value_in_string) +
                ', cannot reshape array of size {} into shape ({},{})'.format(len(matrix_data), row, col))


def generate_boolean_data(key, tsv_file):
    """generate boolean data"""
    boolean_creator = BooleanDataCreator()
    print(tsv_file)
    return boolean_creator.create(key, tsv_file)


def generate_scalar_data(key, tsv_file):
    """generate scalar data"""
    scalar_creator = ScalarDataCreator()
    return scalar_creator.create(key, tsv_file)


def generate_matrix_data(key, tsv_file):
    """generate matrix data"""
    matrix_creator = MatrixDataCreator()
    return matrix_creator.create(key, tsv_file)


def main():
    ''' main function '''
    test_data_tsv = '/home/peterchang/work/sensor_calibration_benchmark/test/gt_tsv_dir/fake_data.tsv'
    boolean_data = generate_boolean_data('SENSOR_CALIB_EXTRINSIC_PARAM_IMU_OFF', test_data_tsv)
    print('boolean_data.key:')
    print(boolean_data.key)
    print('boolean_data.data_frame:')
    print(boolean_data.data_frame.dtypes)
    print(boolean_data.data_frame)
    scalar_data = generate_scalar_data('HEIGHT', test_data_tsv)
    print('scalar_data.key:')
    print(scalar_data.key)
    print('scalar_data.data_frame:')
    print(scalar_data.data_frame.dtypes)
    print(scalar_data.data_frame)
    matrix_data = generate_matrix_data('TR_CAM_TO_IMU', test_data_tsv)
    print('matrix_data.key:')
    print(matrix_data.key)
    print('matrix_data.data_frame:')
    print(matrix_data.data_frame.dtypes)
    print(matrix_data.data_frame)


if __name__ == '__main__':
    main()
