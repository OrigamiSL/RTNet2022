import torch
import torch.nn as nn

from RT.ConvBlock import ConvLayer, ConvBlock
from RT.embed import DataEmbedding


class RT_block(nn.Module):
    def __init__(self, d_model, kernel, dropout, group, block_nums, label_len, pred_len, c_out,
                 flag='End-to-end', FE='ResNet'):
        super(RT_block, self).__init__()
        pro_conv = [ConvBlock(d_model * (2 ** i), d_model * (2 ** (i + 1)),
                              kernel=kernel, dropout=dropout, group=group, FE=FE)
                    for i in range(block_nums)]
        self.pro_conv = nn.ModuleList(pro_conv)
        last_dim = d_model * (2 ** block_nums)
        self.F = nn.Flatten()
        self.flag = flag
        if self.flag == 'End-to-end':
            self.projection = nn.Conv1d(in_channels=last_dim * label_len // (2 ** block_nums),
                                        out_channels=pred_len * c_out, kernel_size=1, groups=group)
        self.c_out = c_out
        self.pred_len = pred_len

    def forward(self, x):
        for conv in self.pro_conv:
            x = conv(x)
        F_out = self.F(x.permute(0, 2, 1)).unsqueeze(-1)
        if self.flag == 'End-to-end':
            x_out = self.projection(F_out).squeeze().contiguous().view(-1, self.c_out, self.pred_len)
            x_out = x_out.transpose(1, 2)
            return x_out
        elif self.flag == 'Self-supervised':
            return F_out.squeeze()


class time_block(nn.Module):
    def __init__(self, enc_in, d_model, dropout, kernel, time_nums, c_out, group):
        super(time_block, self).__init__()
        self.dec_bed = DataEmbedding(enc_in, d_model, dropout)
        time_conv = [ConvBlock(d_model * (2 ** i), d_model * (2 ** (i + 1)), kernel=kernel,
                               dropout=dropout, pool=False, group=group)
                     for i in range(time_nums)]
        self.time_conv = nn.ModuleList(time_conv)
        self.pos_time = ConvLayer(d_model * (2 ** time_nums), c_out, kernel=1, dropout=0, group=group)

    def forward(self, x):
        embed_dec = self.dec_bed(x)
        dec_out = embed_dec
        for t_conv in self.time_conv:
            dec_out = t_conv(dec_out)
        dec_out = self.pos_time(dec_out)
        return dec_out


class FC_block(nn.Module):
    def __init__(self, in_channels, out_channels, group):
        super(FC_block, self).__init__()
        self.FC = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=group)

    def forward(self, x):
        return self.FC(x)


class RT(nn.Module):
    def __init__(self, enc_in, c_out, label_len, pred_len, pred_list, feature_extractor,
                 forecasting_form='End-to-end', kernel=3,
                 group=False, block_nums=3, time_nums=2, timebed='hour',
                 d_model=64, pyramid=1, dropout=0.0):
        super(RT, self).__init__()
        print("Start Embedding")
        # Enbeddinging
        type_bed = {'None': 0, 'hour': 1, 'year': 6, 'year_min': 7}
        timebed = int(type_bed[timebed])
        self.timebed = timebed
        self.group = (enc_in - timebed) if group else 1
        self.pyramid = pyramid

        self.enc_bed = [DataEmbedding(enc_in - timebed, d_model, dropout, group=self.group) for i in range(pyramid)]
        self.enc_bed = nn.ModuleList(self.enc_bed)

        assert (pyramid <= block_nums)
        self.label_len = label_len
        self.pred_len = pred_len
        self.pred_list = pred_list
        self.c_out = c_out
        self.d_model = d_model
        print("Embedding finished")

        RT_blocks = [RT_block(d_model, kernel, dropout, self.group, block_nums - i,
                              label_len // (2 ** i), pred_len, c_out, flag=forecasting_form, FE=feature_extractor)
                     for i in range(pyramid)]
        self.RT_blocks = nn.ModuleList(RT_blocks)

        self.MLP = None
        if forecasting_form == 'Self-supervised':
            # [24 48 168 336 720]
            in_channels = 0
            for i in range(pyramid):
                in_channels += d_model * label_len // (2 ** i)
            MLP_list = [FC_block(in_channels, self.pred_list[i] * c_out, self.group) for i in
                        range(len(self.pred_list))]
            self.MLP = nn.ModuleList(MLP_list)

        if self.timebed:
            if forecasting_form == 'Self-supervised':
                time_list = [time_block(enc_in, d_model, dropout, kernel, time_nums, self.c_out, self.group) for i in
                             range(len(pred_list))]
                self.time_blocks = nn.ModuleList(time_list)
            else:
                self.time_blocks = time_block(enc_in, d_model, dropout, kernel, time_nums, self.c_out, self.group)

    def forward(self, x_enc, relate, flag='End-to-end', index=0):
        if flag == 'End-to-end':
            enc_input = None
            if self.timebed:
                enc_input = x_enc[:, :-self.pred_len, :-self.timebed]
            else:
                enc_input = x_enc[:, :-self.pred_len, :]
            if self.group:
                relate_sum = torch.sum(relate, dim=0, keepdim=True).expand(relate.shape[0], relate.shape[1])
                relate = relate / relate_sum
                enc_input = enc_input.matmul(relate)
            enc_out = 0
            i = 0
            for embed, RT_b in zip(self.enc_bed, self.RT_blocks):
                embed_enc = embed(enc_input[:, -self.label_len // (2 ** i):, :])
                enc_out += RT_b(embed_enc)
                i += 1
            enc_out = enc_out / i

            if self.timebed:
                dec_out = self.time_blocks(x_enc[:, -self.pred_len:, :])
                final_out = enc_out + dec_out
            else:
                final_out = enc_out

            return final_out  # [B, L, D]

        else:
            enc_input = None
            if self.timebed:
                enc_input = x_enc[:, :self.label_len, :-self.timebed]
            else:
                enc_input = x_enc[:, :self.label_len, :]
            if self.group:
                relate_sum = torch.sum(relate, dim=0, keepdim=True).expand(relate.shape[0], relate.shape[1])
                relate = relate / relate_sum
                enc_input = enc_input.matmul(relate)
            i = 0
            enc_out_list = []
            for embed, RT_b in zip(self.enc_bed, self.RT_blocks):
                embed_enc = embed(enc_input[:, -self.label_len // (2 ** i):, :])
                enc_out = RT_b(embed_enc)
                enc_out_list.append(enc_out)
                i += 1

            if flag == 'Cons':
                return enc_out_list, self.group
            elif flag == 'MLP':
                for i in range(len(enc_out_list)):
                    enc_out_list[i] = enc_out_list[i].contiguous().view(enc_out_list[i].shape[0], self.group, -1)
                enc_out_cat = torch.cat(enc_out_list, dim=-1)
                enc_out_cat = enc_out_cat.contiguous().view(enc_out_cat.shape[0], -1)
                temp_out = self.MLP[index](enc_out_cat.detach().unsqueeze(-1))
                enc_out = temp_out.squeeze().contiguous().view(-1, self.c_out, self.pred_list[index])
                enc_out = enc_out.transpose(1, 2)

                if self.timebed:
                    dec_out = self.time_blocks[index](x_enc[:, -self.pred_list[index]:, :])
                    final_out = enc_out + dec_out
                else:
                    final_out = enc_out

                return final_out  # [B, L, D]
