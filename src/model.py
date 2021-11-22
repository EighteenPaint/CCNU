from torch import nn

from src.attention import TransformerLayer, SSAttention
from tcn import TemporalConvNetV4
from utils import *


class ContextEncoderV2(nn.Module):
    def __init__(self, q_num, d_model, out_size, n_heads=2, d_ff=1024, dropout=0.2, model_type='lskt'):
        super(ContextEncoderV2, self).__init__()
        self.embed_size = d_model
        self.n_question = q_num
        self.model_type = model_type
        self.q_embed = nn.Embedding(self.n_question, self.embed_size)
        self.a_embed = nn.Embedding(2, self.embed_size)
        self.transformer_x_enocder = TransformerLayer(d_model=d_model, d_feature=out_size // n_heads, d_ff=d_ff,
                                                      n_heads=n_heads,
                                                      dropout=dropout, kq_same=True)
        self.transformer_y_enocder = TransformerLayer(d_model=d_model, d_feature=out_size // n_heads, d_ff=d_ff,
                                                      n_heads=n_heads,
                                                      dropout=dropout, kq_same=True)
        if model_type == 'lskt':
            self.encoder_fusion = TransformerLayer(d_model=out_size, d_feature=out_size // n_heads, d_ff=d_ff,
                                                   n_heads=n_heads,
                                                   dropout=dropout, kq_same=True)
        self.proj = nn.Linear(2 * d_model, out_size)

    def forward(self, x, y):
        x_embed = self.q_embed(x)
        y_embed = self.a_embed(y)
        # def forward(self, mask, query, key, values, apply_pos=True):
        x_out = self.transformer_x_enocder(mask=1, query=x_embed, key=x_embed, values=x_embed, apply_pos=False)
        y_out = self.transformer_y_enocder(mask=1, query=y_embed, key=y_embed, values=y_embed, apply_pos=False)
        if self.model_type == 'lsktc':
            qa_out = torch.cat([x_out, y_out], dim=-1)
            qa_out = self.proj(qa_out)
        else:
            qa_out = self.encoder_fusion(mask=1, query=x_out, key=x_out, values=y_out, apply_pos=False)
        return qa_out


class LSKT(nn.Module):
    def __init__(self, kernel_size, num_channels, q_num, d_model, encoder_out, ffn_h_num, n=1, d_ff=1024, dropout=0.5,
                 max_len=512, n_heads=8):
        '''
        q_num:number of question
        d_model:embedding size
        out_size:output dim
        d_ff:hidden layer dim
        dropout：dropout
        '''
        super(LSKT, self).__init__()
        self.embed_size = d_model
        self.n_question = q_num
        self.N = n
        self.M = len(num_channels)
        self.out_size = num_channels[-1]
        self.encoder = ContextEncoderV2(q_num=self.n_question, d_model=self.embed_size, out_size=encoder_out, d_ff=d_ff,
                                        dropout=dropout, model_type='lskt', n_heads=n_heads)
        self.sig = nn.Sequential(nn.Sigmoid())
        self.ffc = PositionwiseFeedForward(self.out_size, ffn_h_num, out=q_num, drop_prob=dropout)
        self.p = self.N * self.M
        self.se = SSAttention(channel=self.p, reduction=self.N)
        self.attens = nn.ModuleList(
            [TransformerLayer(d_model=self.out_size, d_feature=self.out_size // n_heads, d_ff=d_ff,
                              n_heads=n_heads,
                              dropout=dropout, kq_same=True) for _ in range(self.p)])
        self.tcns = nn.ModuleList(
            [TemporalConvNetV4(encoder_out, num_channels, kernel_size=kernel_size * 2 ** i, dropout=dropout) for i in
             range(self.N)])

    def forward(self, x, y):
        # 编码
        input_encoder = self.encoder(x=x, y=y)
        out_tcns = []
        for tcn in self.tcns:
            out_tcns.extend(tcn(input_encoder))
        out_attens = []
        for i in range(len(out_tcns)):
            out_attens += [self.attens[i](mask=1, query=out_tcns[i], key=out_tcns[i], values=out_tcns[i])]
        out_stack = torch.stack(out_attens, dim=1).permute(0, 1, 3, 2)  # batch ,channel,height,width
        out = self.se(out_stack).permute(0, 1, 3, 2)
        out = torch.sum(out, dim=1)
        return self.sig(self.ffc(out))


class LSKTNT(nn.Module):
    def __init__(self, kernel_size, num_channels, q_num, d_model, encoder_out, ffn_h_num, n=1, d_ff=1024, dropout=0.5,
                 max_len=512, n_heads=8):
        super(LSKTNT, self).__init__()
        self.embed_size = d_model
        self.n_question = q_num
        self.N = n
        self.M = len(num_channels)
        self.out_size = num_channels[-1]
        self.encoder = ContextEncoderV2(q_num=self.n_question, d_model=self.embed_size, out_size=encoder_out, d_ff=d_ff,
                                        dropout=dropout, model_type='lskt', n_heads=n_heads)
        self.sig = nn.Sequential(nn.Sigmoid())
        self.ffc = PositionwiseFeedForward(self.out_size, ffn_h_num, out=q_num, drop_prob=dropout)
        self.p = self.N * self.M
        self.se = SSAttention(channel=self.p, reduction=self.N)
        self.tcns = nn.ModuleList(
            [TemporalConvNetV4(encoder_out, num_channels, kernel_size=kernel_size * 2 ** i, dropout=dropout) for i in
             range(self.N)])

    def forward(self, x, y):
        input_encoder = self.encoder(x=x, y=y)
        out_tcns = []
        for tcn in self.tcns:
            out_tcns.extend(tcn(input_encoder))
        out_stack = torch.stack(out_tcns, dim=1).permute(0, 1, 3, 2)  # batch ,channel,height,width
        out = self.se(out_stack).permute(0, 1, 3, 2)
        out = torch.sum(out, dim=1)
        return self.sig(self.ffc(out))


class LSKTC(nn.Module):
    def __init__(self, kernel_size, num_channels, q_num, d_model, encoder_out, ffn_h_num, n=1, d_ff=1024, dropout=0.5,
                 max_len=512, n_heads=8):
        super(LSKTC, self).__init__()
        self.embed_size = d_model
        self.n_question = q_num
        self.N = n
        self.M = len(num_channels)
        self.out_size = num_channels[-1]
        self.encoder = ContextEncoderV2(q_num=self.n_question, d_model=self.embed_size, out_size=encoder_out, d_ff=d_ff,
                                        dropout=dropout, model_type='lsktc', n_heads=n_heads)
        self.sig = nn.Sequential(nn.Sigmoid())
        self.ffc = PositionwiseFeedForward(self.out_size, ffn_h_num, out=q_num, drop_prob=dropout)
        self.p = self.N * self.M
        self.se = SSAttention(channel=self.p, reduction=self.N)
        self.attens = nn.ModuleList(
            [TransformerLayer(d_model=self.out_size, d_feature=self.out_size // n_heads, d_ff=d_ff,
                              n_heads=n_heads,
                              dropout=dropout, kq_same=True) for _ in range(self.p)])
        self.tcns = nn.ModuleList(
            [TemporalConvNetV4(encoder_out, num_channels, kernel_size=kernel_size * 2 ** i, dropout=dropout) for i in
             range(self.N)])

    def forward(self, x, y):
        input_encoder = self.encoder(x=x, y=y)
        out_tcns = []
        for tcn in self.tcns:
            out_tcns.extend(tcn(input_encoder))
        out_attens = []
        for i in range(len(out_tcns)):
            out_attens += [self.attens[i](mask=1, query=out_tcns[i], key=out_tcns[i], values=out_tcns[i])]
        out_stack = torch.stack(out_attens, dim=1).permute(0, 1, 3, 2)  # batch ,channel,height,width
        out = self.se(out_stack).permute(0, 1, 3, 2)
        out = torch.sum(out, dim=1)
        return self.sig(self.ffc(out))


class LSKTSK(nn.Module):
    def __init__(self, kernel_size, num_channels, q_num, d_model, encoder_out, ffn_h_num, n=1, d_ff=1024, dropout=0.5,
                 max_len=512, n_heads=8):
        super(LSKTSK, self).__init__()
        self.embed_size = d_model
        self.n_question = q_num
        self.N = n
        self.M = len(num_channels)
        self.out_size = num_channels[-1]
        self.encoder = ContextEncoderV2(q_num=self.n_question, d_model=self.embed_size, out_size=encoder_out, d_ff=d_ff,
                                        dropout=dropout, model_type='lskt', n_heads=n_heads)
        self.sig = nn.Sequential(nn.Sigmoid())
        self.ffc = PositionwiseFeedForward(self.out_size, ffn_h_num, out=q_num, drop_prob=dropout)
        self.p = self.N * self.M
        self.se = SSAttention(channel=self.p, reduction=self.N)
        self.attens = nn.ModuleList(
            [TransformerLayer(d_model=self.out_size, d_feature=self.out_size // n_heads, d_ff=d_ff,
                              n_heads=n_heads,
                              dropout=dropout, kq_same=True) for _ in range(self.p)])
        self.tcns = nn.ModuleList(
            [TemporalConvNetV4(encoder_out, num_channels, kernel_size=kernel_size, dropout=dropout,
                               model_type='lsktsk') for i in
             range(self.N)])

    def forward(self, x, y):
        input_encoder = self.encoder(x=x, y=y)
        out_tcns = []
        for tcn in self.tcns:
            out_tcns.extend(tcn(input_encoder))
        out_attens = []
        for i in range(len(out_tcns)):
            out_attens += [self.attens[i](mask=1, query=out_tcns[i], key=out_tcns[i], values=out_tcns[i])]
        out_stack = torch.stack(out_attens, dim=1).permute(0, 1, 3, 2)  # batch ,channel,height,width
        out = self.se(out_stack).permute(0, 1, 3, 2)
        out = torch.sum(out, dim=1)
        return self.sig(self.ffc(out))


class LSKTNS(nn.Module):
    def __init__(self, kernel_size, num_channels, q_num, d_model, encoder_out, ffn_h_num, n=1, d_ff=1024, dropout=0.5,
                 max_len=512, n_heads=8):
        super(LSKTNS, self).__init__()
        self.embed_size = d_model
        self.n_question = q_num
        self.N = n
        self.M = len(num_channels)
        self.out_size = num_channels[-1]
        self.encoder = ContextEncoderV2(q_num=self.n_question, d_model=self.embed_size, out_size=encoder_out, d_ff=d_ff,
                                        dropout=dropout, model_type='lskt', n_heads=n_heads)
        self.sig = nn.Sequential(nn.Sigmoid())
        self.ffc = PositionwiseFeedForward(self.out_size, ffn_h_num, out=q_num, drop_prob=dropout)
        self.p = self.N * self.M
        self.se = SSAttention(channel=self.p, reduction=self.N)
        self.attens = nn.ModuleList(
            [TransformerLayer(d_model=self.out_size, d_feature=self.out_size // n_heads, d_ff=d_ff,
                              n_heads=n_heads,
                              dropout=dropout, kq_same=True) for _ in range(self.p)])
        self.tcns = nn.ModuleList(
            [TemporalConvNetV4(encoder_out, num_channels, kernel_size=kernel_size * 2 ** i, dropout=dropout) for i in
             range(self.N)])

    def forward(self, x, y):
        input_encoder = self.encoder(x=x, y=y)
        out_tcns = []
        for tcn in self.tcns:
            out_tcns.extend(tcn(input_encoder))
        out_attens = []
        for i in range(len(out_tcns)):
            out_attens += [self.attens[i](mask=1, query=out_tcns[i], key=out_tcns[i], values=out_tcns[i])]
        out_stack = torch.stack(out_attens, dim=1)
        out = torch.mean(out_stack, 1)
        return self.sig(self.ffc(out))


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, out, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
