import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_attn(nn.Module):
    def __init__(self, in_f, out_f, window_size=8):
        super(LSTM_attn, self).__init__()
        self.LSTM = nn.LSTM(
            input_size=in_f,
            hidden_size=64,
            num_layers=3,
            batch_first=True
        )

        self.f1 = nn.Linear(64, 32)
        self.f2 = nn.Linear(32, out_f)
        self.r = nn.ReLU()
        self.d = nn.Dropout(0.3)
        self.window_size = window_size

    def attn(self, lstm_output, h_t):
        # lstm_output [bs, clips, hiden]  h_t[bs, hiden]
        h_t = h_t.unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, h_t) # lstm_output [bs, clips, hidden] ;h_t [bs, hidden, 1] --> attn [bs, clips, 1]
        attn_weights = attn_weights.squeeze()
        attention = F.softmax(attn_weights, dim = 0)
        # bmm : [bs, hidden, clips] [bs, clips, 1]
        attn_out = torch.bmm(lstm_output.transpose(1, 2), attention.unsqueeze(2)) # [bs, hidden, 1]

        return attn_out.squeeze() # [bs, hidden]

    def forward(self, x):
        bs = x.size()[0]
        x = x.view(bs // self.window_size, self.window_size, -1)
        self.LSTM.flatten_parameters()
        x, (hn,hc) = self.LSTM(x) # x.shape -> bs,clip,512
        x_last = x[:, -1, :] # x[:,-1,:].shape [bs, 512]

        # # attention
        # x = self.attn(x, x_last)
        # x = self.d(self.r(self.f1(x)))   
        # x = self.f2(x) # [8, 128] --> [8, 2]

        # direct fc
        x = self.f1(x_last)
        x = self.f2(x)
        return x  # expected output.shape --> [8, 2]