import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import Dataset

from utils.tools import StandardScaler

warnings.filterwarnings('ignore')


def _get_time_features(dt, timebed):
    if timebed == 'year':
        return np.stack([
            dt.hour.to_numpy(),  # hour of day
            dt.dayofweek.to_numpy(),  # day of week
            dt.day.to_numpy(),  # day of month
            dt.dayofyear.to_numpy(),  # day of year
            dt.month.to_numpy(),  # month of year
            dt.weekofyear.to_numpy(),  # week of year
        ], axis=1).astype(np.float)
    elif timebed == 'year_min':
        return np.stack([
            dt.minute.to_numpy(),  # minute of hour
            dt.hour.to_numpy(),  # hour of day
            dt.dayofweek.to_numpy(),  # day of week
            dt.day.to_numpy(),  # day of month
            dt.dayofyear.to_numpy(),  # day of year
            dt.month.to_numpy(),  # month of year
            dt.weekofyear.to_numpy(),  # week of year
        ], axis=1).astype(np.float)
    elif timebed == 'hour':
        return np.stack([
            dt.hour.to_numpy(),
        ], axis=1).astype(np.float)
    else:
        print('invalide time embedding')
        exit(-1)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', timebed='hour', data_path='ETTh1.csv',
                 target='OT', criterion='Standard',
                 forecasting_form='End_to-end', block_shift=4, aug_num=4, jitter=0.2,
                 angle=60, group_pred=1, group_num=0):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.timebed = timebed
        assert timebed in ['None', 'hour', 'year', 'year_min']
        type_bed = {'None': 0, 'hour': 1, 'year': 6, 'year_min': 7}
        self.set_bed = int(type_bed[timebed])
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.criterion = criterion
        self.forecasting_form = forecasting_form
        self.block_shift = block_shift
        rand_x = np.rint(np.random.rand(1) * self.label_len // self.block_shift)
        self.random_shift = int(rand_x[0])
        self.aug_num = aug_num
        self.jitter = jitter
        self.group_pred = group_pred
        self.group_num = group_num
        self.feature_num = 1
        self.angle = angle if self.group_pred == 1 else 1  # if using respective group forecasting, then relation_mat
        # will not be activated (making angle = 1 deg will be approximate to this situation)
        self.Relation_Mat = None
        self.__read_data__()

    def __read_data__(self):
        if str(self.criterion) == 'Standard':
            self.scaler = StandardScaler()
            self.scaler3 = StandardScaler()
        else:
            self.scaler = MaxAbsScaler()
            self.scaler3 = MaxAbsScaler()

        self.scaler2 = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.label_len, 12 * 30 * 24 + 4 * 30 * 24 - self.label_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        if self.features == 'M':
            if self.group_pred == 1:
                cols_data = df_raw.columns[1:]
            else:
                group_len = df_raw.columns[1:].shape[0] // self.group_pred
                if self.group_num != self.group_pred - 1:
                    cols_data = df_raw.columns[1 + group_len * self.group_num:
                                               1 + group_len * (self.group_num + 1)]
                else:
                    cols_data = df_raw.columns[1 + group_len * self.group_num:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        df_value = df_data.values

        datastamp = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col='date', parse_dates=True)
        if self.set_bed:
            dt_embed = _get_time_features(datastamp.index, self.timebed)

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.scaler2.fit(train_data)
        data = self.scaler.transform(df_value)
        data_R = self.scaler2.transform(df_value)

        # Relation Matrix of Variables through cosine matrix
        training_data = data_R[border1s[0]:border2s[0]]
        self.Relation_Mat = np.matmul(training_data.T, training_data)
        RMS = np.sqrt(np.sum(training_data ** 2, axis=0, keepdims=True))
        RMS_dot = np.matmul(RMS.T, RMS)
        self.Relation_Mat = self.Relation_Mat / RMS_dot
        zero_mat = np.zeros_like(self.Relation_Mat)
        self.Relation_Mat = np.where(self.Relation_Mat < np.cos(self.angle * np.pi / 180), zero_mat, self.Relation_Mat)
        if self.set_type == 0:
            print('Variable Relation Matrix is:')
            print(self.Relation_Mat)

        # time embedding
        if self.set_bed:
            train_data_stamp = dt_embed[border1s[0]:border2s[0]]
            self.scaler3.fit(train_data_stamp)
            data_stamp = self.scaler3.transform(dt_embed)
            data = np.concatenate([data, data_stamp], axis=-1)

        self.feature_num = data.shape[-1]
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        if self.forecasting_form == 'End-to-end':
            r_begin = index
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]

            return seq_x, self.set_bed, self.Relation_Mat
        elif self.forecasting_form == 'Self-supervised':
            r_begin = index * self.label_len // self.block_shift + self.random_shift
            r_end = r_begin + self.label_len
            seq_x = self.data_x[r_begin:r_end]

            seq_x = np.expand_dims(seq_x, axis=0)
            seq_x = seq_x.repeat(self.aug_num, axis=0)
            _, L, D = seq_x.shape
            for i in range(self.aug_num - 1):
                rand_aug = int(np.rint(np.random.rand(1) * 3)[0])
                if 0 <= rand_aug < 1:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(L, D) - 0.5) * self.jitter
                elif 1 <= rand_aug < 2:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(1) - 0.5)[0] * self.jitter
                else:
                    seq_x[i + 1, :, :] += (np.random.rand(L, D) - 0.5) * self.jitter
            return seq_x, self.Relation_Mat
        else:
            print('Invalid forecasting form')
            exit(-1)

    def __len__(self):
        bs = self.block_shift
        ll = self.label_len
        if self.forecasting_form == 'End-to-end':
            return len(self.data_x) - self.label_len - self.pred_len + 1
        elif self.forecasting_form == 'Self-supervised':
            return bs * (len(self.data_x) - (1 + bs) * ll // bs) // ll
        else:
            print('Invalid forecasting form')
            exit(-1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def standard_transformer(self, data):
        return self.scaler2.transform(data)

    def target_feature(self):
        return self.feature_num


class Dataset_ETT_min(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', timebed='hour', data_path='ETTm1.csv',
                 target='OT', criterion='Standard',
                 forecasting_form='End_to-end', block_shift=4, aug_num=4, jitter=0.2,
                 angle=60, group_pred=1, group_num=0):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.timebed = timebed
        assert timebed in ['None', 'hour', 'year', 'year_min']
        type_bed = {'None': 0, 'hour': 1, 'year': 6, 'year_min': 7}
        self.set_bed = int(type_bed[timebed])
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.criterion = criterion
        self.forecasting_form = forecasting_form
        self.block_shift = block_shift
        rand_x = np.rint(np.random.rand(1) * self.label_len // self.block_shift)
        self.random_shift = int(rand_x[0])
        self.aug_num = aug_num
        self.jitter = jitter
        self.group_pred = group_pred
        self.group_num = group_num
        self.feature_num = 1
        self.angle = angle if self.group_pred == 1 else 1  # if using respective group forecasting, then relation_mat
        # will not be activated (making angle = 1 deg will be approximate to this situation)
        self.Relation_Mat = None
        self.__read_data__()

    def __read_data__(self):
        if str(self.criterion) == 'Standard':
            self.scaler = StandardScaler()
            self.scaler3 = StandardScaler()
        else:
            self.scaler = MaxAbsScaler()
            self.scaler3 = MaxAbsScaler()

        self.scaler2 = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.label_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.label_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        if self.features == 'M':
            if self.group_pred == 1:
                cols_data = df_raw.columns[1:]
            else:
                group_len = df_raw.columns[1:].shape[0] // self.group_pred
                if self.group_num != self.group_pred - 1:
                    cols_data = df_raw.columns[1 + group_len * self.group_num:
                                               1 + group_len * (self.group_num + 1)]
                else:
                    cols_data = df_raw.columns[1 + group_len * self.group_num:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        df_value = df_data.values

        datastamp = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col='date', parse_dates=True)
        if self.set_bed:
            dt_embed = _get_time_features(datastamp.index, self.timebed)

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.scaler2.fit(train_data)
        data = self.scaler.transform(df_value)
        data_R = self.scaler2.transform(df_value)

        # Relation Matrix of Variables through cosine matrix
        training_data = data_R[border1s[0]:border2s[0]]
        self.Relation_Mat = np.matmul(training_data.T, training_data)
        RMS = np.sqrt(np.sum(training_data ** 2, axis=0, keepdims=True))
        RMS_dot = np.matmul(RMS.T, RMS)
        self.Relation_Mat = self.Relation_Mat / RMS_dot
        zero_mat = np.zeros_like(self.Relation_Mat)
        self.Relation_Mat = np.where(self.Relation_Mat < np.cos(self.angle * np.pi / 180), zero_mat, self.Relation_Mat)
        if self.set_type == 0:
            print('Variable Relation Matrix is:')
            print(self.Relation_Mat)

        # time embedding
        if self.set_bed:
            train_data_stamp = dt_embed[border1s[0]:border2s[0]]
            self.scaler3.fit(train_data_stamp)
            data_stamp = self.scaler3.transform(dt_embed)
            data = np.concatenate([data, data_stamp], axis=-1)

        self.feature_num = data.shape[-1]
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        if self.forecasting_form == 'End-to-end':
            r_begin = index
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]

            return seq_x, self.set_bed, self.Relation_Mat
        elif self.forecasting_form == 'Self-supervised':
            r_begin = index * self.label_len // self.block_shift + self.random_shift
            r_end = r_begin + self.label_len
            seq_x = self.data_x[r_begin:r_end]

            seq_x = np.expand_dims(seq_x, axis=0)
            seq_x = seq_x.repeat(self.aug_num, axis=0)
            _, L, D = seq_x.shape
            for i in range(self.aug_num - 1):
                rand_aug = int(np.rint(np.random.rand(1) * 3)[0])
                if 0 <= rand_aug < 1:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(L, D) - 0.5) * self.jitter
                elif 1 <= rand_aug < 2:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(1) - 0.5)[0] * self.jitter
                else:
                    seq_x[i + 1, :, :] += (np.random.rand(L, D) - 0.5) * self.jitter
            return seq_x, self.Relation_Mat
        else:
            print('Invalid forecasting form')
            exit(-1)

    def __len__(self):
        bs = self.block_shift
        ll = self.label_len
        if self.forecasting_form == 'End-to-end':
            return len(self.data_x) - self.label_len - self.pred_len + 1
        elif self.forecasting_form == 'Self-supervised':
            return bs * (len(self.data_x) - (1 + bs) * ll // bs) // ll
        else:
            print('Invalid forecasting form')
            exit(-1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def standard_transformer(self, data):
        return self.scaler2.transform(data)

    def target_feature(self):
        return self.feature_num


class Dataset_ECL(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', timebed='hour', data_path='ECL.csv',
                 target='MT_320', criterion='Standard',
                 forecasting_form='End_to-end', block_shift=4, aug_num=4, jitter=0.2,
                 angle=60, group_pred=1, group_num=0):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.timebed = timebed
        assert timebed in ['None', 'hour', 'year', 'year_min']
        type_bed = {'None': 0, 'hour': 1, 'year': 6, 'year_min': 7}
        self.set_bed = int(type_bed[timebed])
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.criterion = criterion
        self.forecasting_form = forecasting_form
        self.block_shift = block_shift
        rand_x = np.rint(np.random.rand(1) * self.label_len // self.block_shift)
        self.random_shift = int(rand_x[0])
        self.aug_num = aug_num
        self.jitter = jitter
        self.group_pred = group_pred
        self.group_num = group_num
        self.feature_num = 0
        self.Relation_Mat = None
        self.angle = angle if self.group_pred == 1 else 1  # if using respective group forecasting, then relation_mat
        # will not be activated (making angle = 1 deg will be approximate to this situation)
        self.__read_data__()

    def __read_data__(self):
        if str(self.criterion) == 'Standard':
            self.scaler = StandardScaler()
            self.scaler3 = StandardScaler()
        else:
            self.scaler = MaxAbsScaler()
            self.scaler3 = MaxAbsScaler()

        self.scaler2 = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.label_len, len(df_raw) - num_test - self.label_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            if self.group_pred == 1:
                cols_data = df_raw.columns[1:]
            else:
                group_len = df_raw.columns[1:].shape[0] // self.group_pred
                if self.group_num != self.group_pred - 1:
                    cols_data = df_raw.columns[1 + group_len * self.group_num:
                                               1 + group_len * (self.group_num + 1)]
                else:
                    cols_data = df_raw.columns[1 + group_len * self.group_num:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        datastamp = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col='date', parse_dates=True)
        if self.set_bed:
            dt_embed = _get_time_features(datastamp.index, self.timebed)

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.scaler2.fit(train_data)
        data = self.scaler.transform(df_value)
        data_R = self.scaler2.transform(df_value)

        # Relation Matrix of Variables through cosine matrix
        training_data = data_R[border1s[0]:border2s[0]]
        self.Relation_Mat = np.matmul(training_data.T, training_data)
        RMS = np.sqrt(np.sum(training_data ** 2, axis=0, keepdims=True))
        RMS_dot = np.matmul(RMS.T, RMS)
        self.Relation_Mat = self.Relation_Mat / RMS_dot
        zero_mat = np.zeros_like(self.Relation_Mat)
        self.Relation_Mat = np.where(self.Relation_Mat < np.cos(self.angle * np.pi / 180), zero_mat, self.Relation_Mat)
        if self.set_type == 0:
            print('Variable Relation Matrix is:')
            print(self.Relation_Mat)

        # time embedding
        if self.set_bed:
            train_data_stamp = dt_embed[border1s[0]:border2s[0]]
            self.scaler3.fit(train_data_stamp)
            data_stamp = self.scaler3.transform(dt_embed)
            data = np.concatenate([data, data_stamp], axis=-1)

        self.feature_num = data.shape[-1]
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        if self.forecasting_form == 'End-to-end':
            r_begin = index
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]

            return seq_x, self.set_bed, self.Relation_Mat
        elif self.forecasting_form == 'Self-supervised':
            r_begin = index * self.label_len // self.block_shift + self.random_shift
            r_end = r_begin + self.label_len
            seq_x = self.data_x[r_begin:r_end]

            seq_x = np.expand_dims(seq_x, axis=0)
            seq_x = seq_x.repeat(self.aug_num, axis=0)
            _, L, D = seq_x.shape
            for i in range(self.aug_num - 1):
                rand_aug = int(np.rint(np.random.rand(1) * 3)[0])
                if 0 <= rand_aug < 1:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(L, D) - 0.5) * self.jitter
                elif 1 <= rand_aug < 2:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(1) - 0.5)[0] * self.jitter
                else:
                    seq_x[i + 1, :, :] += (np.random.rand(L, D) - 0.5) * self.jitter
            return seq_x, self.Relation_Mat
        else:
            print('Invalid forecasting form')
            exit(-1)

    def __len__(self):
        bs = self.block_shift
        ll = self.label_len
        if self.forecasting_form == 'End-to-end':
            return len(self.data_x) - self.label_len - self.pred_len + 1
        elif self.forecasting_form == 'Self-supervised':
            return bs * (len(self.data_x) - (1 + bs) * ll // bs) // ll
        else:
            print('Invalid forecasting form')
            exit(-1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def standard_transformer(self, data):
        return self.scaler2.transform(data)

    def target_feature(self):
        return self.feature_num


class Dataset_WTH(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', timebed='hour', data_path='WTH.csv',
                 target='WetBulbCelsius', criterion='Standard',
                 forecasting_form='End_to-end', block_shift=4, aug_num=4, jitter=0.2,
                 angle=60, group_pred=1, group_num=0):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.timebed = timebed
        assert timebed in ['None', 'hour', 'year', 'year_min']
        type_bed = {'None': 0, 'hour': 1, 'year': 6, 'year_min': 7}
        self.set_bed = int(type_bed[timebed])
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.criterion = criterion
        self.forecasting_form = forecasting_form
        self.block_shift = block_shift
        rand_x = np.rint(np.random.rand(1) * self.label_len // self.block_shift)
        self.random_shift = int(rand_x[0])
        self.aug_num = aug_num
        self.jitter = jitter
        self.group_pred = group_pred
        self.group_num = group_num
        self.feature_num = 0
        self.angle = angle
        self.Relation_Mat = None
        self.__read_data__()

    def __read_data__(self):
        if str(self.criterion) == 'Standard':
            self.scaler = StandardScaler()
            self.scaler3 = StandardScaler()
        else:
            self.scaler = MaxAbsScaler()
            self.scaler3 = MaxAbsScaler()

        self.scaler2 = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.label_len, len(df_raw) - num_test - self.label_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            if self.group_pred == 1:
                cols_data = df_raw.columns[1:]
            else:
                group_len = df_raw.columns[1:].shape[0] // self.group_pred
                if self.group_num != self.group_pred - 1:
                    cols_data = df_raw.columns[1 + group_len * self.group_num:
                                               1 + group_len * (self.group_num + 1)]
                else:
                    cols_data = df_raw.columns[1 + group_len * self.group_num:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        datastamp = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col='date', parse_dates=True)
        if self.set_bed:
            dt_embed = _get_time_features(datastamp.index, self.timebed)

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.scaler2.fit(train_data)
        data = self.scaler.transform(df_value)
        data_R = self.scaler2.transform(df_value)

        # Relation Matrix of Variables through cosine matrix
        training_data = data_R[border1s[0]:border2s[0]]
        self.Relation_Mat = np.matmul(training_data.T, training_data)
        RMS = np.sqrt(np.sum(training_data ** 2, axis=0, keepdims=True))
        RMS_dot = np.matmul(RMS.T, RMS)
        self.Relation_Mat = self.Relation_Mat / RMS_dot
        zero_mat = np.zeros_like(self.Relation_Mat)
        self.Relation_Mat = np.where(self.Relation_Mat < np.cos(self.angle * np.pi / 180), zero_mat, self.Relation_Mat)
        if self.set_type == 0:
            print('Variable Relation Matrix is:')
            print(self.Relation_Mat)

        # time embedding
        if self.set_bed:
            train_data_stamp = dt_embed[border1s[0]:border2s[0]]
            self.scaler3.fit(train_data_stamp)
            data_stamp = self.scaler3.transform(dt_embed)
            data = np.concatenate([data, data_stamp], axis=-1)

        self.feature_num = data.shape[-1]
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        if self.forecasting_form == 'End-to-end':
            r_begin = index
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]

            return seq_x, self.set_bed, self.Relation_Mat
        elif self.forecasting_form == 'Self-supervised':
            r_begin = index * self.label_len // self.block_shift + self.random_shift
            r_end = r_begin + self.label_len
            seq_x = self.data_x[r_begin:r_end]

            seq_x = np.expand_dims(seq_x, axis=0)
            seq_x = seq_x.repeat(self.aug_num, axis=0)
            _, L, D = seq_x.shape
            for i in range(self.aug_num - 1):
                rand_aug = int(np.rint(np.random.rand(1) * 3)[0])
                if 0 <= rand_aug < 1:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(L, D) - 0.5) * self.jitter
                elif 1 <= rand_aug < 2:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(1) - 0.5)[0] * self.jitter
                else:
                    seq_x[i + 1, :, :] += (np.random.rand(L, D) - 0.5) * self.jitter
            return seq_x, self.Relation_Mat
        else:
            print('Invalid forecasting form')
            exit(-1)

    def __len__(self):
        bs = self.block_shift
        ll = self.label_len
        if self.forecasting_form == 'End-to-end':
            return len(self.data_x) - self.label_len - self.pred_len + 1
        elif self.forecasting_form == 'Self-supervised':
            return bs * (len(self.data_x) - (1 + bs) * ll // bs) // ll
        else:
            print('Invalid forecasting form')
            exit(-1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def standard_transformer(self, data):
        return self.scaler2.transform(data)

    def target_feature(self):
        return self.feature_num
