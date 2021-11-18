# 定义cuda()函数：参数为o，如果use_cuda为真返回o.cuda(),为假返回o
import random

import torch

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)
cuda = lambda o: o.cuda() if use_cuda else o
# torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称。
tensor = lambda o: cuda(torch.tensor(o))
# 生成对角线全1，其余部分全0的二维数组,函数原型：torch.eye(n, m=None, out=None)，m (int) ：列数.如果为None,则默认为n。
eye = lambda d: cuda(torch.eye(d))
# 返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor。
zeros = lambda *args: cuda(torch.zeros(*args))

crandn = lambda *args: cuda(torch.randn(*args))
# 截断反向传播的梯度流,返回一个新的Variable即tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
# 不同之处只是它的requires_grad是false，也就是说这个Variable永远不需要计算其梯度，不具有grad。
detach = lambda o: o.cpu().detach().numpy().tolist()


def set_seed(seed=0):
    # seed()方法改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数。random:随机数生成器，seed:种子
    random.seed(seed)
    # 为CPU设置种子用于生成随机数
    torch.manual_seed(seed)
    # 为当前GPU设置随机种子,如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    torch.cuda.manual_seed(seed)
    '''
    置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，
    应该可以保证每次运行网络的时候相同输入的输出是固定的。（说人话就是让每次跑出来的效果是一致的）
    '''
    torch.backends.cudnn.deterministic = True
    '''
     置为True的话会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，
    其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
    '''
    torch.backends.cudnn.benchmark = False


# def collate(batch, q_num, length=200):
#     lens = [len(row) for row in batch]
#     max_len = length  # max(lens)
#     batch = tensor([[[*e, 1] for e in row] + [[0, 0, 0]] * (max_len - len(row)) for row in batch])
#     Q, Y, S = batch.T  # Q:问题，Y:预测，S:padding,样本数据缺失或者说不够时填充[[0,0,0]]张量
#     Q, Y, S = Q.T, Y.T, S.T  # torch.size([32,200])
#     X = Q + q_num * (1 - Y)
#     return X, Y, S, Q

