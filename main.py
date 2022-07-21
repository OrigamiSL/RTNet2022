import argparse
import os
import torch
import numpy as np
import time

from exp.exp_model import Exp_Model

parser = argparse.ArgumentParser(description='[RTNet] Respecting Times Series Properties')

parser.add_argument('--model', type=str, required=True, default='RT',
                    help='model of experiment')
parser.add_argument('--forecasting_form', type=str, default='End-to-end',
                    help='forecasting form', choices=['End-to-end', 'Self-supervised'])

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S]; M:multivariate predict multivariate, '
                         'S: univariateb predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or M task')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--label_len', type=int, default=24, help='input length of RTNet auto-regressive feature extractor')
parser.add_argument('--Alter_label_len', type=str, default='168,48',
                    help='alternative sequence length for group instance forecasting')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--pred_list', type=str, default='24,48,168,336,720', help='group of prediction sequence length '
                                                                               'in self-supervised models')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='input variates')
parser.add_argument('--c_out', type=int, default=7, help='output variates')
parser.add_argument('--timebed', type=str, default='None', choices=['None', 'hour', 'year', 'year_min'],
                    help='time embedding type')
parser.add_argument('--d_model', type=int, default=28, help='hidden dimension of model')
parser.add_argument('--pyramid', type=int, default=1, help='num of pyramid networks')
parser.add_argument('--angle', type=float, default=45, help='threshold angle of cosine relationship of Variable Matrix')
parser.add_argument('--feature_extractor', type=str, default='ResNet',
                    choices=['ResNet', 'CSPNet', 'Attention'], help='feature extractor used in backbone')
parser.add_argument('--group', action='store_true', help='group convolution for multivariate forecasting', default=False)
parser.add_argument('--group_pred', type=int, default=1, help='nums of forecasting groups')
parser.add_argument('--group_num', type=int, default=0, help='indexes of forecasting groups')
parser.add_argument('--block_shift', type=int, default=4, help='')
parser.add_argument('--aug_num', type=int, default=4, help='nums of data augmentation, including initial sequences')
parser.add_argument('--jitter', type=float, default=0.2, help='data augmentation amplification')
parser.add_argument('--kernel', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--criterion', type=str, default='Standard', choices=['Standard', 'Maxabs'],
                    help='options:[Standard, Maxabs]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--cost_epochs', type=int, default=10, help='Contrastive epochs')
parser.add_argument('--cost_grow_epochs', type=int, default=5, help='Contrastive growing epochs per experiment')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--cost_batch_size', type=int, default=64, help='batch size of self-supervised train input data')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--block_nums', type=int, default=3, help='num of RTBlocks in the dominant pyramid network')
parser.add_argument('--time_nums', type=int, default=2, help='num of TimeBlocks')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--test_inverse', action='store_true', help='only inverse test data', default=False)
parser.add_argument('--save_loss', action='store_false', help='whether saving results and checkpoints', default=True)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--train', action='store_false',
                    help='whether to train'
                    , default=True)
parser.add_argument('--reproducible', action='store_false',
                    help='whether to make results reproducible'
                    , default=True)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'M': [7, 7], 'S': [1, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'M': [7, 7], 'S': [1, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'M': [7, 7], 'S': [1, 1]},
    'WTH': {'data': 'WTH.csv', 'M': [12, 12], 'S': [1, 1]},
    'ECL': {'data': 'ECL.csv', 'M': [321, 321], 'S': [1, 1]}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.enc_in, args.c_out = data_info[args.features]

assert args.timebed in ['None', 'hour', 'year', 'year_min']
type_bed = {'None': 0, 'hour': 1, 'year': 6, 'year_min': 7}
set_bed = type_bed[args.timebed]
args.enc_in = args.enc_in + int(set_bed)

args.target = args.target.replace('/r', '').replace('/t', '').replace('/n', '')
args.Alter_label_len = [int(leng) for leng in args.Alter_label_len.replace(' ', '').split(',')]
args.pred_list = [int(l) for l in args.pred_list.replace(' ', '').split(',')]

lr = args.learning_rate
print('Args in experiment:')
print(args)

for ii in range(args.itr):
    # setting record of experiments
    if args.group_pred == 1:
        args.cost_epochs = args.cost_epochs + args.cost_grow_epochs if ii != 0 else args.cost_epochs
        setting = '{}_{}_{}_{}_ft{}_ll{}_pl{}_list{}_{}'.format(args.model, args.forecasting_form,
                                                                args.feature_extractor,
                                                                args.data, args.features,
                                                                args.label_len, args.pred_len, args.pred_list, ii)
        Exp = Exp_Model
        exp = Exp(args)  # set experiments
        if args.train:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            try:
                exp.train(setting)
            except KeyboardInterrupt:
                print('-' * 99)
                print('Exiting from training early')

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)

        torch.cuda.empty_cache()
        args.learning_rate = lr
    else:
        args.cost_epochs = args.cost_epochs + args.cost_grow_epochs if ii != 0 else args.cost_epochs
        if args.forecasting_form == 'End-to-end':
            t_mse = 0
            t_mae = 0
        elif args.forecasting_form == 'Self-supervised':
            t_mse = [0] * len(args.pred_list)
            t_mae = [0] * len(args.pred_list)
        assert (args.features == 'M')
        ini_enc = args.enc_in - int(set_bed)  # real instance group num without time embedding
        ini_c_out = args.c_out
        mod_dim = ini_enc % args.group_pred  # extra instance num for the last group
        args.Alter_label_len.append(int(args.label_len))  # total input sequence length choices
        for index_gp in range(args.group_pred):
            args.group_num = index_gp
            args.enc_in = ini_enc // args.group_pred + int(set_bed) + \
                          (mod_dim if index_gp == args.group_pred - 1 else 0)
            args.c_out = ini_c_out // args.group_pred + (mod_dim if index_gp == args.group_pred - 1 else 0)
            args.d_model = args.d_model // (args.enc_in - int(set_bed)) * (args.enc_in - int(set_bed))
            # ensuring dimensions could be divided
            # initializing mse, mae, label_len to be chosen
            if args.forecasting_form == 'End-to-end':
                b_mse = 1000
                b_mae = 1000
                b_label_len = 0
            elif args.forecasting_form == 'Self-supervised':
                b_mse = []
                b_mae = []
                b_label_len = []
            else:
                print('Invalid forecasting form')
                exit(-1)
            for index_label_len, cur_label_len in enumerate(args.Alter_label_len):
                args.label_len = cur_label_len
                setting = '{}_{}_{}_{}_ft{}_ll{}_pl{}_list{}_gp{}_gn{}_{}'.format(args.model, args.forecasting_form,
                                                                                  args.feature_extractor,
                                                                                  args.data,
                                                                                  args.features,
                                                                                  args.label_len, args.pred_len,
                                                                                  args.pred_list,
                                                                                  args.group_pred, args.group_num, ii)
                Exp = Exp_Model
                exp = Exp(args)  # set experiments
                if args.train:
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                    try:
                        exp.train(setting)
                    except KeyboardInterrupt:
                        print('-' * 99)
                        print('Exiting from training early')

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                mse, mae = exp.test(setting, load=True, write_loss=False, save_loss=False, return_loss=True)
                if args.forecasting_form == 'End-to-end':
                    if b_mse > mse:
                        b_mse = mse
                        b_mae = mae
                        b_label_len = args.label_len
                else:
                    if index_label_len == 0:
                        b_mse = mse
                        b_mae = mae
                        b_label_len = [args.label_len] * len(args.pred_list)
                    else:
                        for i_loss in range(len(mse)):
                            b_label_len[i_loss] = args.label_len if b_mse[i_loss] > mse[i_loss] else b_label_len[i_loss]
                            b_mse[i_loss] = mse[i_loss] if b_mse[i_loss] > mse[i_loss] else mse[i_loss]
                            b_mae[i_loss] = mae[i_loss] if b_mae[i_loss] > mae[i_loss] else mae[i_loss]

                torch.cuda.empty_cache()
                args.learning_rate = lr  # re-define learning rate

            if args.forecasting_form == 'End-to-end':
                path = './result.log'
                with open(path, "a") as f:
                    f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                    f.write('|{}|form_{}|label_len{}|pred_len{}|group_{}|mse:{}, mae:{}'.
                            format(args.data, args.forecasting_form, b_label_len, args.pred_len, index_gp,
                                   b_mse, b_mae) + '\n')
                    f.flush()
                    f.close()
                t_mse += b_mse
                t_mae += b_mae
            else:
                path = './result.log'
                for i_result in range(len(t_mse)):
                    with open(path, "a") as f:
                        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                        f.write('|{}|form_{}|label_len{}|pred_len{}|group_{}|mse:{}, mae:{}'.
                                format(args.data, args.forecasting_form,
                                       b_label_len[i_result], args.pred_list[i_result], index_gp, b_mse[i_result],
                                       b_mae[i_result]) + '\n')
                        f.flush()
                        f.close()
                    t_mse[i_result] += b_mse[i_result]
                    t_mae[i_result] += b_mae[i_result]

        if args.forecasting_form == 'End-to-end':
            t_mse /= float(args.group_pred)
            t_mae /= float(args.group_pred)
            print('mse:{}, mae:{}'.format(t_mse, t_mse))
            r_path = './result.log'
            with open(r_path, "a") as f:
                f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                f.write('total_mse:{}, total_mae:{}'.format(t_mse, t_mae) + '\n')
                f.flush()
                f.close()
        else:
            for i_group in range(len(t_mse)):
                t_mse[i_group] /= float(args.group_pred)
                t_mae[i_group] /= float(args.group_pred)
                print('mse:{}, mae:{}'.format(t_mse[i_group], t_mse[i_group]))
                r_path = './result.log'
                with open(r_path, "a") as f:
                    f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                    f.write('pred_len:{}, total_mse:{}, total_mae:{}'.
                            format(args.pred_list[i_group], t_mse[i_group], t_mae[i_group]) + '\n')
                    f.flush()
                    f.close()
