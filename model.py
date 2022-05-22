import torch.nn as nn
# 构建线性回归模型


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # nn.Linear表示y=wx+b,参数分别表示x和y的维度

    def forward(self, x):
        out = self.linear(x)
        return out
