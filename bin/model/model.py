# -*- coding: utf-8 -*- 
# @File : model.py
# @Author : Zhengming Li
# @Date : Nov 2021 

import torch
import torch.nn.functional as F
from torch import nn


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活
        # 定义可训练参数，即论文中的W和a, W means compact feature; a means how to calculate weight based on two feature
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
            inp: input_fea [N, in_features] N 结点个数  in_features表示节点的输入特征向量元素个数
            adj: 图的邻接矩阵  [N, N] 非零即一
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 图的节点数
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，只计算邻居节点
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = nn.Softmax(dim=1)(attention)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = nn.Dropout(p=self.dropout)(attention)
        # attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return nn.ELU()(h_prime)
        else:
            return h_prime


# multi-head
class GATmulti_head(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAT层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GATmulti_head, self).__init__()
        self.dropout = dropout
        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))  # 输出并激活
        return x
        # return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero(as_tuple=False).t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        # edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime


class multiGAT_2layer(nn.Module):
    def __init__(self, input_feature_len, out_put, drop, alpha, head):
        super(multiGAT_2layer, self).__init__()
        self.gat1 = GATmulti_head(int(input_feature_len), int(input_feature_len / head), int(input_feature_len / head),
                                  drop, alpha, head)
        self.relu = nn.ELU()
        self.gat2 = GATmulti_head(int(input_feature_len / head), out_put, out_put, drop, alpha, 1)

    def forward(self, input_feature, adj):
        gat1 = self.gat1(input_feature, adj)
        x = self.relu(gat1)
        gat2 = self.gat2(x, adj)

        return gat2


class GAT_FeatureToDistance(nn.Module):
    def __init__(self, feature_length, dropout=0.2, alpha=0.2):
        """
        初始化
        :param feature_length: 节点特征向量的维度，这里是节点数
        :param dropout: 丢弃率，默认 0，2
        :param alpha: LearyReLU的激活函数，即 x < 0 时的斜率。默认 0.2
        """
        super().__init__()
        self.GAT = GraphAttentionLayer(feature_length, feature_length, dropout, alpha)
        self.decoder = nn.Sequential(
            nn.Linear(feature_length, feature_length),
            nn.ReLU(),
            nn.Linear(feature_length, feature_length),
            nn.ReLU()
        )

    def forward(self, input_feature, adj):
        """
        前向计算
        :param input_feature: 输入特征
        :param adj: 邻接矩阵
        :return decode: 返回通过GAT后得到的distance
        """
        encode = self.GAT(input_feature, adj)
        decode = self.decoder(encode)
        return decode


class GAT_DistanceToPosition(nn.Module):
    def __init__(self, feature_length, output_length, dropout=0.2, alpha=0.2):
        """
        初始化
        :param feature_length: 节点特征向量的维度
        :param output_length: 输出向量维度
        :param dropout: 丢弃率，默认 0，2
        :param alpha: LearyReLU的激活函数，即 x < 0 时的斜率。默认 0.2
        """

        super().__init__()
        self.GAT = GraphAttentionLayer(feature_length, output_length, dropout, alpha)
        '''
        self.decoder = nn.Sequential(
            nn.Linear(feature_length, int(feature_length / 4), ),
            nn.ReLU(),
            nn.Linear(int(feature_length / 4), output_length),
            nn.ReLU()
        )
        '''

    def forward(self, input_feature, adj):
        """
        前向计算
        :param input_feature: tensor 输入特征
        :param adj: tensor 邻接矩阵
        :return decode: tensor 返回通过GAT后得到的 position
        """

        encode = self.GAT(input_feature, adj)
        # decode = self.decoder(encode)
        # return decode
        return encode


class model1(nn.Module):
    def __init__(self, feature_len, compact, midoutput):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_len, int(feature_len / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_len / 2), compact),
            nn.ReLU()
        )
        self.GAT = GraphAttentionLayer(compact, midoutput, 0.2, 0.2, concat=True)
        self.decoder = nn.Sequential(
            nn.Linear(midoutput, int(feature_len / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_len / 2), feature_len),
            nn.ReLU()
        )

    def forward(self, x, adj):
        encoded = self.encoder(x)
        gat = self.GAT(encoded, adj)
        decoded = self.decoder(gat)
        return decoded


"""
模型分为两个task，共用一个Encoder：

Encoder 包含两个 MLP 和一个 GAT（输入维度为 feature_length, 输出维度为 feature_length）：
    1. MLP1 包含两个全连接层（Fully Connection Layer，FCL）,（输入维度为 feature_length, 输出维度为 encoder_mid_output_length）：
        - FCL1 : 输入维度为 feature_length, 输出维度为 feature_length / 2，激活函数为 ReLU
        - FCL2 : 输入维度为 feature_length / 2， 输出维度为 encoder_mid_output_length，激活函数为 ReLU

    2. MLP2 包含两个全连接层（输入维度为 encoder_mid_output_length, 输出维度为 encoder_mid_output_length）：
        - FCL3 : 输入维度为 encoder_mid_output_length， 输出维度为 feature_length / 2，激活函数为 ReLU
        - FCL4 : 输入维度为 feature_length / 2， 输出维度为 feature_length，激活函数为 ReLU

    3. GAT (输入维度为 encoder_mid_output_length，输出维度为encoder_mid_output_length)

数据经过 MLP1 后输入到 GAT，然后再输入到 MLP2 得到输出
"""


class ResGAT_Encoder(nn.Module):
    def __init__(self, feature_length, encoder_mid_output_length, dropout=0.2, alpha=0.2):
        super(ResGAT_Encoder, self).__init__()
        self.MLP1 = nn.Sequential(
            nn.Linear(feature_length, int(feature_length / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_length / 2), encoder_mid_output_length),
            nn.ReLU()
        )
        self.GAT = GraphAttentionLayer(encoder_mid_output_length, encoder_mid_output_length, dropout, alpha)
        self.MLP2 = nn.Sequential(
            nn.Linear(encoder_mid_output_length, int(feature_length / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_length / 2), feature_length),
            nn.ReLU()
        )

    def forward(self, feature, adj_matrix):
        mlp1_out = self.MLP1(feature)
        gat_out = self.GAT(mlp1_out, adj_matrix)
        output = self.MLP2(gat_out)
        return output


"""
模型由一个GAT，一个 MLP, 和一个Encoder 组成
Encoder: ResGAT_Encoder
GAT: 输入维度为 feature_length, 输出维度为 mid_out_put_length。
MLP：输入维度为 mid_out_put_length， 输出维度为 feature_length

数据经过 Encoder 后输入GAT得到 distance_gat_mid_output, 再输入MLP得到output。
模型返回 distance_gat_mid_output 和 output。distance_gat_mid_output用于 ResGAT_Position 输入
"""


class ResGAT_Distance(nn.Module):
    def __init__(self, feature_length, encoder_mid_output_length, mid_out_put_length, dropout=0.2, alpha=0.2):
        super(ResGAT_Distance, self).__init__()
        self.encoder = ResGAT_Encoder(feature_length, encoder_mid_output_length)
        self.GAT = GraphAttentionLayer(feature_length, mid_out_put_length, dropout, alpha)
        self.MLP = nn.Sequential(
            nn.Linear(mid_out_put_length, int(feature_length / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_length / 2), feature_length),
            nn.ReLU()
        )

    def forward(self, feature, adj_matrix):
        encoder_output = self.encoder(feature, adj_matrix)
        distance_gat_mid_output = self.GAT(encoder_output, adj_matrix)
        output = self.MLP(distance_gat_mid_output)
        return distance_gat_mid_output, output


"""
模型由一个GAT，一个 MLP, 和一个Encoder 组成
Encoder: ResGAT_Encoder
GAT: 输入维度为 feature_length, 输出维度为 mid_out_put_length。
MLP：输入维度为 mid_out_put_length * 2， 输出维度为 feature_length

数据经过 Encoder 后输入GAT得到 position_gat_mid_output, 再与 ResGAT_Distance 中的 distance_gat_mid_output 按列拼接后输入MLP得到output。
模型返回 output。
"""


class ResGAT_Position(nn.Module):
    def __init__(self, feature_length, encoder_mid_output_length, mid_output_length, output_length, dropout=0.2,
                 alpha=0.2):
        super(ResGAT_Position, self).__init__()
        self.encoder = ResGAT_Encoder(feature_length, encoder_mid_output_length)
        self.GAT = GraphAttentionLayer(feature_length, mid_output_length, dropout, alpha)
        self.MLP = nn.Sequential(
            nn.Linear(mid_output_length * 2, int(feature_length / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_length / 2), output_length),
            nn.ReLU()
        )

    def forward(self, feature, distance_gat_mid_output, adj_matrix):
        encoder_output = self.encoder(feature, adj_matrix)
        position_gat_mid_output = self.GAT(encoder_output, adj_matrix)
        MLP_input = torch.cat((position_gat_mid_output, distance_gat_mid_output), dim=1);
        # print(MLP_input._version)
        output = self.MLP(MLP_input);
        return output


'''
第一个encoder直接GAT不经过MLP的模型
'''


class ResGAT_Encoder_noMLP(nn.Module):
    def __init__(self, feature_length, encoder_mid_output_length, dropout=0.2, alpha=0.2):
        super(ResGAT_Encoder_noMLP, self).__init__()
        self.GAT = GraphAttentionLayer(feature_length, encoder_mid_output_length, dropout, alpha)
        self.MLP2 = nn.Sequential(
            nn.Linear(encoder_mid_output_length, int(feature_length / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_length / 2), feature_length),
            nn.ReLU()
        )
        # self.GAT = GraphAttentionLayer(feature_length, feature_length, dropout, alpha)

    def forward(self, feature, adj_matrix):
        gat_out = self.GAT(feature, adj_matrix)
        output = self.MLP2(gat_out)
        return output
        # return  gat_out


class ResGAT_Distance_noMLP(nn.Module):
    def __init__(self, feature_length, encoder_mid_output_length, mid_out_put_length, dropout=0.2, alpha=0.2):
        super(ResGAT_Distance_noMLP, self).__init__()
        self.encoder = ResGAT_Encoder_noMLP(feature_length, encoder_mid_output_length)
        self.GAT = GraphAttentionLayer(feature_length, mid_out_put_length, dropout, alpha)
        self.MLP = nn.Sequential(
            nn.Linear(mid_out_put_length, int(feature_length / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_length / 2), feature_length),
            nn.ReLU()
        )

    def forward(self, feature, adj_matrix):
        encoder_output = self.encoder(feature, adj_matrix)
        distance_gat_mid_output = self.GAT(encoder_output, adj_matrix)
        output = self.MLP(distance_gat_mid_output)
        return distance_gat_mid_output, output


class ResGAT_Position_noMLP(nn.Module):
    def __init__(self, feature_length, encoder_mid_output_length, mid_output_length, output_length, dropout=0.2,
                 alpha=0.2):
        super(ResGAT_Position_noMLP, self).__init__()
        self.encoder = ResGAT_Encoder_noMLP(feature_length, encoder_mid_output_length)
        self.GAT = GraphAttentionLayer(feature_length, mid_output_length, dropout, alpha)
        self.MLP = nn.Sequential(
            nn.Linear(mid_output_length * 2, int(feature_length / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_length / 2), output_length),
            nn.ReLU()
        )

    def forward(self, feature, distance_gat_mid_output, adj_matrix):
        encoder_output = self.encoder(feature, adj_matrix)
        position_gat_mid_output = self.GAT(encoder_output, adj_matrix)
        MLP_input = torch.cat((position_gat_mid_output, distance_gat_mid_output), dim=1);
        # print(MLP_input._version)
        output = self.MLP(MLP_input);
        return output