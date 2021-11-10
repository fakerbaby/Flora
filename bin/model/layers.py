import torch
import torch.nn.functional as F
from torch import nn


class GraphAttentionLayer(nn.Module):
    """
       Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  #  represent the number of input features of the vector
        self.out_features = out_features  #  represent the number of output features of the vector
        self.dropout = dropout  # dropout
        self.alpha = alpha  # the parameter to acitivate leakyrelu
        self.concat = concat  # the parameter to acitivate elu 
        #  'W' means compact feature; 'a' means how to calculate weight based on two feature
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # initialization
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        Args:
            inp: input_fea [N, in_features] 'N' represents the number of nodes, 'in_features' represent the number of input features of the vector
            adj: Adjacent matrix  [N, N] either Non-zero or one
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] The coefficient of GraphAttentionLayer(Unnormalized) 
        zero_vec = -1e12 * torch.ones_like(e)  # set non-connectivity node Non-infinity
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # masked 
        attention = nn.Softmax(dim=1)(attention)  # Normalized
        attention = nn.Dropout(p=self.dropout)(attention)
        # attention = F.dropout(attention, self.dropout, training=self.training)  # use dropout to prohibit overfitting
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # Get the representation updated by the surrounding nodes through the attention weight
        if self.concat:
            return nn.ELU()(h_prime)
        else:
            return h_prime


# multi-head
class GATmulti_head(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """
        Dense version of GAT
        Args:
        n_heads: Indicates that there are several GAT layers, which are finally spliced ​​together, 
        similar to multi-head attention. For extracting features from different dimensions
        """
        super(GATmulti_head, self).__init__()
        self.dropout = dropout
        # define multi-head GraphAttentionLayer
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  
        x = F.elu(self.out_att(x, adj))  
        return x
        # return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer.
    To solve the issue that tough backpropataion problem of Sparse Matrix
    # There are numerically instability because of softmax function. Therefore, you need to initialize carefully.
    # To use sparse version GAT, add flag --sparse. The performance of sparse version is similar with tensorflow. On a Titan Xp takes 0.08~0.14 sec. 
    # come from the https://github.com/Diego999/pyGAT
    """

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
        self.gat1 = GATmulti_head(int(input_feature_len), int(input_feature_len/head), int(input_feature_len/head), drop, alpha, head)
        self.relu = nn.ELU()
        self.gat2 = GATmulti_head(int(input_feature_len/head), out_put, out_put, drop, alpha, 1)

    def forward(self, input_feature, adj):
        gat1 = self.gat1(input_feature, adj)
        x = self.relu(gat1)
        gat2 = self.gat2(x, adj)
        return gat2


class GAT_2line(nn.Module):
    def __init__(self, input_feature_len, out_put, drop, alpha):
        super(GAT_2line, self).__init__()
        self.gat = GraphAttentionLayer(input_feature_len, input_feature_len,drop, alpha)
        self.decoder = nn.Sequential(
            nn.Linear(input_feature_len, int(input_feature_len / 4)),
            nn.ReLU(),
            nn.Linear(int(input_feature_len / 4), out_put),
            nn.ReLU()
        )

    def forward(self, input_feature, adj):
        encode = self.gat(input_feature, adj)
        decode = self.decoder(encode)
        return decode




