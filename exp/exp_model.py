from data.data_loader import Dataset_ETT_hour, Dataset_ETT_min, Dataset_WTH, Dataset_ECL
from exp.exp_basic import Exp_Basic
from RT.model import RT

from utils.tools import EarlyStopping, adjust_learning_rate, loss_process, Cost_loss
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'RT': RT,
        }
        if self.args.model == 'RT':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.c_out,
                self.args.label_len,
                self.args.pred_len,
                self.args.pred_list,
                self.args.feature_extractor,
                self.args.forecasting_form,
                self.args.kernel,
                self.args.group,
                self.args.block_nums,
                self.args.time_nums,
                self.args.timebed,
                self.args.d_model,
                self.args.pyramid,
                self.args.dropout
            ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, forecasting_form=None, pred_len=0):
        args = self.args
        data_set = None
        if pred_len == 0:
            pred_len = args.pred_len

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_min,
            'WTH': Dataset_WTH,
            'ECL': Dataset_ECL
        }
        Data = data_dict[self.args.data]

        curr_forecasting_form = None
        if forecasting_form is not None:
            curr_forecasting_form = forecasting_form
        else:
            curr_forecasting_form = args.forecasting_form
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.label_len, pred_len],
            features=args.features,
            timebed=args.timebed,
            target=args.target,
            criterion=args.criterion,
            forecasting_form=curr_forecasting_form,
            block_shift=args.block_shift,
            aug_num=args.aug_num,
            jitter=args.jitter,
            angle=args.angle,
            group_pred=args.group_pred,
            group_num=args.group_num
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size if curr_forecasting_form == 'End-to-end' else self.args.cost_batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data=None, vali_loader=None, criterion=None, flag='End-to-end', index=0):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, embed, relat) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    batch_x, embed, relat, flag=flag, index=index)
                loss = loss_process(pred, true, criterion, flag=1)
                total_loss.append(loss)

            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        if self.args.forecasting_form == 'End-to-end':
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

            path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(path):
                os.makedirs(path)

            time_now = time.time()

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            model_optim = self._select_optimizer()
            criterion = self._select_criterion()

            for epoch in range(self.args.train_epochs):
                iter_count = 0

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, embed, relat) in enumerate(train_loader):
                    model_optim.zero_grad()
                    iter_count += 1
                    pred, true = self._process_one_batch(
                        batch_x, embed, relat)
                    # loss = loss_process(pred, true, criterion, flag=0)
                    SE = (pred - true) ** 2
                    loss = torch.mean(SE, dim=0)
                    loss = torch.mean(loss, dim=0)

                    loss.backward(torch.ones_like(loss))
                    model_optim.step()

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                                torch.mean(loss).item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

            return self.model

        elif self.args.forecasting_form == 'Self-supervised':
            #  Contrastive learning
            train_data, train_loader = self._get_data(flag='train')

            path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(path):
                os.makedirs(path)

            time_now = time.time()

            train_steps = len(train_loader)
            model_optim = self._select_optimizer()

            for epoch in range(self.args.cost_epochs):
                if epoch > 0:
                    train_data, train_loader = self._get_data(flag='train')
                iter_count = 0

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, relat) in enumerate(train_loader):
                    B, A, L, D = batch_x.shape
                    batch_x = batch_x.contiguous().view(-1, L, D)
                    model_optim.zero_grad()
                    iter_count += 1
                    hidden_features, group = self._process_one_batch_cons(
                        batch_x, relat)

                    loss = Cost_loss(hidden_features, group, aug_num=self.args.aug_num)

                    loss.backward(torch.ones_like(loss))
                    model_optim.step()

                    if (i + 1) % 1 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                                torch.mean(loss).item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.cost_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
                print("Stage: COST |Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

            # MLP
            lr = self.args.learning_rate
            for lens in self.args.pred_list:
                self.args.learning_rate = lr
                train_data, train_loader = self._get_data(flag='train', forecasting_form='End-to-end', pred_len=lens)
                vali_data, vali_loader = self._get_data(flag='val', forecasting_form='End-to-end', pred_len=lens)
                test_data, test_loader = self._get_data(flag='test', forecasting_form='End-to-end', pred_len=lens)

                time_now = time.time()
                train_steps = len(train_loader)

                early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
                criterion = self._select_criterion()
                for epoch in range(self.args.train_epochs):
                    iter_count = 0

                    self.model.train()
                    epoch_time = time.time()
                    for i, (batch_x, embed, relat) in enumerate(train_loader):
                        model_optim.zero_grad()
                        iter_count += 1
                        pred, true = self._process_one_batch(
                            batch_x, embed, relat, flag='MLP', index=self.args.pred_list.index(lens))
                        SE = (pred - true) ** 2
                        loss = torch.mean(SE, dim=0)
                        loss = torch.mean(loss, dim=0)

                        loss.backward(torch.ones_like(loss))
                        model_optim.step()

                        if (i + 1) % 100 == 0:
                            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                                    torch.mean(loss).item()))
                            speed = (time.time() - time_now) / iter_count
                            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                            iter_count = 0
                            time_now = time.time()

                    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                    vali_loss = self.vali(vali_data, vali_loader, criterion, flag='MLP',
                                          index=self.args.pred_list.index(lens))
                    test_loss = self.vali(test_data, test_loader, criterion, flag='MLP',
                                          index=self.args.pred_list.index(lens))

                    print("Pred_len: {0} Stage: MLP| Epoch: {1}, Steps: {2} | Vali Loss: {3:.7f} Test Loss: {4:.7f}".
                          format(lens, epoch + 1, train_steps, vali_loss, test_loss))
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                    adjust_learning_rate(model_optim, epoch + 1, self.args)

                best_model_path = path + '/' + 'checkpoint.pth'
                self.model.load_state_dict(torch.load(best_model_path))

            return self.model
        else:
            print('Invalid forecasting form')
            exit(-1)

    def test(self, setting, load=False, write_loss=True, save_loss=True, return_loss=False):
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        if self.args.forecasting_form == 'End-to-end':
            test_data, test_loader = self._get_data(flag='test', forecasting_form='End-to-end')

            preds = []
            trues = []

            criterion = self._select_criterion()
            with torch.no_grad():
                for i, (batch_x, embed, relat) in enumerate(test_loader):
                    pred, true = self._process_one_batch(batch_x, embed, relat)
                    if self.args.test_inverse:
                        pred = loss_process(pred, true, criterion, flag=2, dataset=test_data)
                        pred = pred.reshape(self.args.batch_size, self.args.pred_len, self.args.c_out)
                        true = true.reshape(-1, pred.shape[-1])
                        true = test_data.inverse_transform(true.detach().cpu().numpy())
                        true = test_data.standard_transformer(true)
                        true = true.reshape(self.args.batch_size, self.args.pred_len, self.args.c_out)
                    else:
                        pred = loss_process(pred, true, criterion, flag=2)
                        pred = pred.detach().cpu().numpy()
                        true = true.detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(true)

            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            mae, mse = metric(preds, trues)
            print('|{}_{}|form_{}|pred_len{}|mse:{}, mae:{}'.
                  format(self.args.data, self.args.features, self.args.forecasting_form, self.args.pred_len, mse,
                         mae) + '\n')

            if write_loss:
                path = './result.log'
                with open(path, "a") as f:
                    f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                    f.write('|{}_{}|form_{}|pred_len{}|mse:{}, mae:{}'.
                            format(self.args.data, self.args.features, self.args.forecasting_form, self.args.pred_len,
                                   mse, mae) + '\n')
                    f.flush()
                    f.close()
            else:
                pass
            if save_loss:
                # result save
                folder_path = './results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                np.save(folder_path + 'metrics.npy', np.array([mae, mse]))
                np.save(folder_path + 'pred.npy', preds)
                np.save(folder_path + 'true.npy', trues)
            else:
                dir_path = os.path.join(self.args.checkpoints, setting)
                check_path = dir_path + '/' + 'checkpoint.pth'
                if os.path.exists(check_path):
                    os.remove(check_path)
                    os.removedirs(dir_path)
            if return_loss:
                return mse, mae
            else:
                return
        elif self.args.forecasting_form == 'Self-supervised':
            mse_list = []
            mae_list = []
            for lens in self.args.pred_list:
                test_data, test_loader = self._get_data(flag='test', forecasting_form='End-to-end', pred_len=lens)

                preds = []
                trues = []

                criterion = self._select_criterion()
                with torch.no_grad():
                    for i, (batch_x, embed, relat) in enumerate(test_loader):
                        pred, true = self._process_one_batch(
                            batch_x, embed, relat, flag='MLP', index=self.args.pred_list.index(lens))
                        if self.args.test_inverse:
                            pred = loss_process(pred, true, criterion, flag=2, dataset=test_data)
                            pred = pred.reshape(self.args.batch_size, lens, self.args.c_out)
                            true = true.reshape(-1, pred.shape[-1])
                            true = test_data.inverse_transform(true.detach().cpu().numpy())
                            true = test_data.standard_transformer(true)
                            true = true.reshape(self.args.batch_size, lens, self.args.c_out)
                        else:
                            pred = loss_process(pred, true, criterion, flag=2)
                            pred = pred.detach().cpu().numpy()
                            true = true.detach().cpu().numpy()
                        preds.append(pred)
                        trues.append(true)

                preds = np.array(preds)
                trues = np.array(trues)
                print('test shape:', preds.shape, trues.shape)
                preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                print('test shape:', preds.shape, trues.shape)

                mae, mse = metric(preds, trues)
                print('|{}_{}|form_{}|pred_len{}|mse:{}, mae:{}'.
                      format(self.args.data, self.args.features, self.args.forecasting_form, lens, mse, mae) + '\n')

                if write_loss:
                    path = './result.log'
                    with open(path, "a") as f:
                        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                        f.write('|{}_{}|form_{}|pred_len{}|mse:{}, mae:{}'.
                                format(self.args.data, self.args.features, self.args.forecasting_form, lens, mse,
                                       mae) + '\n')
                        f.flush()
                        f.close()
                else:
                    pass
                if save_loss:
                    # result save
                    folder_path = './results/' + setting + '/'
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    np.save(folder_path + f'metrics_{lens}.npy', np.array([mae, mse]))
                    np.save(folder_path + f'pred_{lens}.npy', preds)
                    np.save(folder_path + f'true_{lens}.npy', trues)
                mse_list.append(mse)
                mae_list.append(mae)

            if not save_loss:
                dir_path = os.path.join(self.args.checkpoints, setting)
                check_path = dir_path + '/' + 'checkpoint.pth'
                if os.path.exists(check_path):
                    os.remove(check_path)
                    os.removedirs(dir_path)
            if return_loss:
                return mse_list, mae_list
            else:
                return

    def _process_one_batch(self, batch_x, embed, relat, flag='End-to-end', index=0):
        batch_x = batch_x.float().to(self.device)
        embed = int(embed[0])
        relat = relat[0].float().to(self.device)
        assert (relat.ndim == 2)
        if flag == 'End-to-end':
            dec_inp = torch.zeros([batch_x.shape[0], self.args.pred_len, batch_x.shape[-1] - embed]).float().to(
                self.device)
            if embed:
                dec_inp = torch.cat([dec_inp, batch_x[:, -self.args.pred_len:, -embed:]], dim=-1)
            input_seq = batch_x[:, :self.args.label_len, :]
            dec_inp = torch.cat([input_seq, dec_inp], dim=1).float().to(self.device)
            outputs = self.model(dec_inp, relat, flag)
            if embed:
                batch_y = batch_x[:, -self.args.pred_len:, :-embed].to(self.device)
            else:
                batch_y = batch_x[:, -self.args.pred_len:, :].to(self.device)
        elif flag == 'MLP':
            dec_inp = torch.zeros([batch_x.shape[0], self.args.pred_list[index], batch_x.shape[-1] - embed]).float().to(
                self.device)
            if embed:
                dec_inp = torch.cat([dec_inp, batch_x[:, -self.args.pred_list[index]:, -embed:]], dim=-1)
            input_seq = batch_x[:, :self.args.label_len, :]
            dec_inp = torch.cat([input_seq, dec_inp], dim=1).float().to(self.device)
            outputs = self.model(dec_inp, relat, flag, index=index)
            if embed:
                batch_y = batch_x[:, -self.args.pred_list[index]:, :-embed].to(self.device)
            else:
                batch_y = batch_x[:, -self.args.pred_list[index]:, :].to(self.device)
        else:
            print('error!')
            exit(-1)
        return outputs, batch_y

    def _process_one_batch_cons(self, batch_x, relat):
        batch_x = batch_x.float().to(self.device)
        relat = relat[0].float().to(self.device)
        assert (relat.ndim == 2)
        outputs, group = self.model(batch_x, relat, flag='Cons')
        return outputs, group
