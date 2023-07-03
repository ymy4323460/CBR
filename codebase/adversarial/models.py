import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class MLP(nn.Module):

    def __init__(self, fc_in_dim, fc_dims, dropout=None, batch_norm=None, activation=nn.ReLU()):
        """
        The MLP(Multi-Layer Perceptrons) module
        :param fc_in_dim: The dimension of input tensor
        :param fc_dims: The num_neurons of each layer, should be array-like
        :param dropout: The dropout rate of the MLP module, can be number or array-like ranges (0,1), by default None
        :param batch_norm: Whether to use batch normalization after each layer, by default None
        :param activation: The activation function used in each layer, by default nn.ReLU()
        """
        super(MLP, self).__init__()
        self.fc_dims = fc_dims
        layer_dims = [fc_in_dim]
        layer_dims.extend(fc_dims)
        layers = []

        if not dropout:
            dropout = np.repeat(0, len(fc_dims))
        if isinstance(dropout, float):
            dropout = np.repeat(dropout, len(fc_dims))

        for i in range(len(layer_dims) - 1):
            fc_layer = nn.Linear(in_features=layer_dims[i], out_features=layer_dims[i + 1])
            nn.init.xavier_uniform_(fc_layer.weight)
            layers.append(fc_layer)
            if batch_norm:
                batch_norm_layer = nn.BatchNorm1d(num_features=layer_dims[i + 1])
                layers.append(batch_norm_layer)
            layers.append(activation)
            if dropout[i]:
                dropout_layer = nn.Dropout(dropout[i])
                layers.append(dropout_layer)
        self.mlp = nn.Sequential(*layers)

    def forward(self, feature):
        y = self.mlp(feature)
        return y



class Discriminator(nn.Module):
    def __init__(self, fc_in_dim, fc_dims, dropout=None, batch_norm=None, activation=nn.ELU(), embedding_classweight=None):
        """
        The MLP(Multi-Layer Perceptrons) module
        :param fc_in_dim: The dimension of input tensor
        :param fc_dims: The num_neurons of each layer, should be array-like
        :param dropout: The dropout rate of the MLP module, can be number or array-like ranges (0,1), by default None
        :param batch_norm: Whether to use batch normalization after each layer, by default None
        :param activation: The activation function used in each layer, by default nn.ReLU()
        """
        super(Discriminator, self).__init__()
        # self.fc_dims = fc_dims
        # layer_dims = [fc_in_dim]
        # layer_dims.extend(fc_dims)
        # layers = []
        #
        # if not dropout:
        #     dropout = np.repeat(0, len(fc_dims))
        # if isinstance(dropout, float):
        #     dropout = np.repeat(dropout, len(fc_dims))
        #
        # for i in range(len(layer_dims) - 1):
        #     fc_layer = nn.Linear(in_features=layer_dims[i], out_features=layer_dims[i + 1])
        #     nn.init.xavier_uniform_(fc_layer.weight)
        #     layers.append(fc_layer)
        #     if batch_norm:
        #         batch_norm_layer = nn.BatchNorm1d(num_features=layer_dims[i + 1])
        #         layers.append(batch_norm_layer)
        #     layers.append(activation)
        #     if dropout[i]:
        #         dropout_layer = nn.Dropout(dropout[i])
        #         layers.append(dropout_layer)
        self.mlp = nn.Sequential(
            nn.Linear(fc_in_dim, fc_dims[0]), # 6_11 confounder_test
            nn.ELU(),
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ELU(),
            nn.Linear(fc_dims[1], 2),
        )
        # self.mlp = nn.Sequential(*layers)
        self.sftcross = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(embedding_classweight)).float())

    def forward(self, feature, item):
        y = self.mlp(torch.cat((feature, item), 1))
        return y

    def loss(self, feature, item, r_true):
        r_hat = self.forward(feature, item)
        # loss = self.sftcross(r_hat, torch.ones_like(r_hat)[:, 0].long().squeeze_())
        # print(r_hat1, r_true)
        loss = self.sftcross(r_hat, r_true.squeeze_())
        return loss


class ConfounderInference(nn.Module):
    def __init__(self, args):
        super(ConfounderInference, self).__init__()

        self.args = args
        self.x_dim = args.user_dim
        self.a_dim = args.item_dim
        self.x_emb_dim = args.user_emb_dim
        self.a_emb_dim = args.item_emb_dim
        self.confounder_dims = args.confounder_dims
        if self.x_dim == 1:
            self.user_embedding_lookup = nn.Embedding(self.args.user_item_size[0], self.x_emb_dim)
        else:
            self.user_embedding_lookup = nn.Sequential(
                nn.Linear(self.x_dim, self.x_emb_dim)
            )
        if self.a_dim == 1:
            self.item_embedding_lookup = nn.Embedding(self.args.user_item_size[1], self.a_emb_dim)
        else:
            self.item_embedding_lookup = nn.Sequential(
                nn.Linear(self.a_dim, self.a_emb_dim)
            )
        if args.dataset == 'huawei':
            enc_in_dim = self.x_emb_dim
            dec_in_dim = self.args.confounder_dims + self.x_emb_dim
        else:
            enc_in_dim = self.x_emb_dim + self.a_emb_dim
            dec_in_dim = self.args.confounder_dims + self.x_emb_dim + self.a_emb_dim

        self.confounder_encode = nn.Sequential(
                nn.Linear(enc_in_dim, self.args.confounder_encode_layer_dims[0]),
                nn.ELU(),
                nn.Linear(self.args.confounder_encode_layer_dims[0], self.args.confounder_dims),
                nn.ELU()
                # nn.BatchNorm1d(self.args.confounder_dims)
            )
        self.confounder_decode = nn.Sequential(
                nn.Linear(dec_in_dim, self.args.confounder_decode_layer_dims[0]),
                nn.ELU(),
                nn.Linear(self.args.confounder_decode_layer_dims[0], self.args.confounder_decode_layer_dims[1]),
                nn.ELU(),
                nn.Linear(self.args.confounder_decode_layer_dims[1], 2)
            )
        self.sftcross = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(self.args.ctr_classweight)).float())

    def get_confounder(self, x, a):
        # inference
        x_emb = self.user_embedding_lookup(x).reshape(-1, self.args.user_emb_dim)
        # if self.x_dim == 1:
        #     x_emb = self.user_embedding_lookup(x)
        # else:
        #     x_emb = x
        if self.args.dataset != 'huawei':
            a_emb = self.item_embedding_lookup(a).reshape(-1, self.args.item_emb_dim)
            z = self.confounder_encode(torch.cat((x_emb, a_emb), 1))
            # generate
            r_hat = self.confounder_decode(torch.cat((z, x_emb, a_emb), 1))
        else:
            a_emb = self.item_embedding_lookup(a)

            z = self.confounder_encode(x_emb)
            # generate
            r_hat = self.confounder_decode(torch.cat((z, x_emb), 1))
        return z, x_emb, a_emb, r_hat

    def loss(self, x, a, r):
        z, x_emb, a_emb, r_hat = self.get_confounder(x, a)
        loss = self.sftcross(r_hat, r.squeeze_())
        return z, x_emb, a_emb, r_hat, loss

class DCN(nn.Module):
    def __init__(self, num_cont_fields, cross_depth, fc_dims=None,
                 dropout=None, batch_norm=None, out_type='binary', emb_dim=None, num_feats=None, num_cate_fields=None):
        super(DCN, self).__init__()
        # do not consider categorical in this version
        # self.emb_dim = emb_dim
        # self.num_feats = num_feats
        # self.num_cate_fields = num_cate_fields
        self.num_cont_fields = num_cont_fields

        self.cross_depth = cross_depth
        # embedding for category features
        # self.emb_layer = nn.Embedding(num_embeddings=num_feats - num_cont_fields, embedding_dim=emb_dim)
        # nn.init.xavier_uniform_(self.emb_layer.weight)

        # deep network
        if not fc_dims:
            fc_dims = [64, 32]
        self.fc_dims = fc_dims
        x0_dim = num_cont_fields  # + num_cate_fields * emb_dim
        self.deep = MLP(x0_dim, fc_dims, dropout, batch_norm)

        # cross network
        cross_layers = []
        for _ in range(cross_depth):
            cross_layers.append(CrossLayer(x0_dim))
        self.cross = nn.ModuleList(cross_layers)

        # self.outlayer = OutputLayer(in_dim=fc_dims[-1] + x0_dim, out_type=out_type)

    def embeddings(self, continuous_value, categorical_index=None):
        # cate_emb_value = self.emb_layer(categorical_index)  # N * num_cate_fields * emb_dim
        # # N * (num_cate_fields * emb_dim)
        # cate_emb_value = cate_emb_value.reshape((-1, self.num_cate_fields * self.emb_dim))
        x0 = continuous_value
        y_dnn = self.deep(x0)

        xi = x0
        for cross_layer in self.cross:
            xi = cross_layer(x0, xi)

        output = torch.cat([y_dnn, xi], dim=1)
        # output = self.out_layer(output)
        return output


class CrossLayer(nn.Module):
    def __init__(self, x_dim):
        super(CrossLayer, self).__init__()
        self.x_dim = x_dim
        self.weights = nn.Parameter(torch.zeros(x_dim, 1))  # x_dim * 1
        nn.init.xavier_uniform_(self.weights.data)
        self.bias = nn.Parameter(torch.randn(x_dim))  # x_dim

    def forward(self, x0, xi):
        # x0,x1: N * x_dim
        # print(x0.size(), xi.size())
        x = torch.mm(xi, self.weights)  # N * x_dim
        x = torch.sum(x, dim=1)  # N
        # x = x.unsqueeze(dim=1)  # N * 1
        # print(x.size())
        x = torch.matmul(x, x0)  # N * x_dim
        x = x + self.bias + xi
        return x


class IPMEmbedding(nn.Module):
    def __init__(self, x_dim, a_dim, emb_dim1, emb_dim2, layer_dim, item_space, x_a_size=None, discriminitor=None, confounder_dims=None, embedding_classweight=None):
        super().__init__()
        self.name = 'IPM'
        self.item_space = item_space
        self.x_dim = x_dim
        self.a_dim = a_dim
        # if self.x_dim == self.a_dim == 1:
        self.x_emb_dim = emb_dim1
        self.a_emb_dim = emb_dim2
        self.x_a_size = x_a_size
        self.confounder_dims = confounder_dims
        if discriminitor is not None:
            self.discriminitor = discriminitor
        else:
            self.discriminitor = Discriminator(fc_in_dim=self.x_emb_dim+self.a_emb_dim, fc_dims=layer_dim, activation=nn.ELU(), embedding_classweight=embedding_classweight)

        self.user_embedding_lookup =  nn.Sequential(
            nn.Embedding(self.x_a_size[0], self.x_emb_dim),
            nn.ReLU(),
            nn.Linear(self.x_emb_dim, self.x_emb_dim)
        )
        self.user_embedding = nn.Sequential(
            nn.Linear(self.x_dim, self.x_emb_dim)
        )
        if self.a_dim == 1:
            self.item_embedding_lookup = nn.Embedding(self.x_a_size[1], self.a_emb_dim)
        else:
            self.item_embedding = nn.Sequential(
                nn.Linear(self.a_dim, self.a_emb_dim)
            )

        self.emb_with_confounder = nn.Sequential(
            nn.Linear(self.x_emb_dim + self.confounder_dims, layer_dim[0]),
            nn.ELU(),
            nn.Linear(layer_dim[0], self.x_emb_dim)
        )

        # self.prediction_net = nn.Sequential(
        # 	nn.Linear(emb_dim+a_dim, layer_dim[1]),
        # 	nn.ELU(),
        # 	nn.Linear(layer_dim[1], layer_dim[2]),
        # 	nn.ELU(),
        # 	nn.Linear(layer_dim[2], self.item_space)
        # 	)
        self.prediction_net = nn.Sequential(
            nn.Linear(self.x_emb_dim, layer_dim[0]),
            nn.ELU(),
            nn.Linear(layer_dim[0], layer_dim[1]),
            nn.ELU(),
            nn.Linear(layer_dim[1], item_space),
        )
        self.bce = torch.nn.BCELoss()
        self.sigmd = torch.nn.Sigmoid()
        self.sftcross = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(embedding_classweight)).float(), reduce=False)
        self.sftcross1 = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(embedding_classweight)).float(), reduce=False)

    # self.sftcross = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([12.5,1.0])).float())

    def embedding(self, x, a=None):
        if self.x_dim > 1:
            x_embedding = self.user_embedding(x).reshape([x.size()[0], self.x_emb_dim])
        else:
            x_embedding = self.user_embedding_lookup(x).reshape([x.size()[0], self.x_emb_dim])
        if a is not None:
            if self.a_dim > 1:
                a_embedding = self.item_embedding(a).reshape([a.size()[0], self.a_emb_dim])
            else:
                a_embedding = self.item_embedding_lookup(a).reshape([a.size()[0], self.a_emb_dim])
            return x_embedding, a_embedding
        # print("embedding x_emb_lookup.size():%s" % list(x_emb_lookup.size())) #[8192, 64]
        return x_embedding

    # return self.user_embedding(x_emb_lookup).reshape([x.size()[0], self.x_emb_dim])
    def embedding_with_confounder(self, x, z):
        # print(x.size(), z.size())
        return self.emb_with_confounder(torch.cat((x, z), dim=1))
        # return torch.cat((x, z), dim=1)

    def discepency(self, p_x, a, impressioon_or_click='impression', z=None, r_true=None):
        '''
        p_x: (batch_size, feature size)
        a: (batch_size, index)
        a_hat: (batch_size, item_space_size)
        '''


        # r_hat = self.prediction_net(torch.cat((x_embedding, a_embedding), 1)) # 12_29

        # if impressioon_or_click == 'impression':
        #     x_embedding = self.embedding(p_x)
        #     if self.a_dim > 1:
        #         a_embedding = self.item_embedding(a).reshape([a.size()[0], self.a_emb_dim])
        #     else:
        #         a_embedding = self.item_embedding_lookup(a).reshape([a.size()[0], self.a_emb_dim])
        #     r_hat = self.prediction_net(x_embedding)  # 12_29
        #     discepency = torch.abs(self.sftcross(r_hat, r.squeeze_()) - 0.5)
        if impressioon_or_click == 'Adversarial':
            if z is None:
                x_embedding = self.embedding(p_x)
                if self.a_dim > 1:
                    a_embedding = self.item_embedding(a).reshape([a.size()[0], self.a_emb_dim])
                else:
                    a_embedding = self.item_embedding_lookup(a).reshape([a.size()[0], self.a_emb_dim])
            else:

                x_embedding = self.embedding_with_confounder(p_x, z)
                a_embedding = a
            discrepency = -self.discriminitor.loss(x_embedding, a_embedding, r_true)
        return x_embedding, discrepency.mean(), discrepency


class DoubleModelCTR(nn.Module):
    def __init__(self, mode, x_dim, a_dim, emb_dim1, emb_dim2, layer_dim, y_space, x_a_size=None, ctr_classweight=None, is_debias=None):
        super().__init__()
        self.name = 'CTR'
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.mode = mode
        self.x_emb_dim = emb_dim1#6_11 confounder test
        self.a_emb_dim = emb_dim2
        self.x_a_size = x_a_size
        self.is_debias = is_debias
        y_space = 1
        # print(self.x_dim)
        self.user_embedding_lookup = nn.Embedding(self.x_a_size[0], self.x_emb_dim)
        self.user_embedding = nn.Sequential(
            nn.Linear(self.x_dim, self.x_emb_dim),
            nn.ELU()
            # nn.Linear(layer_dim[0], self.x_emb_dim),
        )
        if self.a_dim == 1:
            # print("DoubleModelCTR", self.x_a_size[1], self.a_emb_dim)
            self.item_embedding_lookup = nn.Embedding(self.x_a_size[1], self.a_emb_dim)
        else:
            self.item_embedding = nn.Sequential(
                nn.Linear(self.a_dim, self.a_emb_dim),
                nn.ELU()
            )
        if self.is_debias:
            predict_in_dim = self.x_emb_dim+self.a_emb_dim
        else:
            if self.x_dim == 1:
                predict_in_dim = self.x_emb_dim+self.a_emb_dim
            else:
                predict_in_dim = self.x_dim + self.a_emb_dim

        if self.is_debias:
            self.predict_net = nn.Sequential(
                nn.Linear(predict_in_dim, layer_dim[1]),
                nn.ELU(),
                nn.Linear(layer_dim[1], layer_dim[2]),
                nn.ELU(),
                nn.Linear(layer_dim[2], y_space)
            )
        else:
            self.predict_net = nn.Sequential(
                nn.Linear(predict_in_dim, layer_dim[1]),
                nn.ELU(),
                nn.Linear(layer_dim[1], layer_dim[2]),
                nn.ELU(),
                nn.Linear(layer_dim[2], y_space)
            )
        # # huawei
        # if self.is_debias:
        #     self.predict_net = nn.Sequential(
        #         nn.Linear(self.x_emb_dim + self.a_emb_dim, layer_dim[2]),
        #         nn.ELU(),
        #         nn.Linear(layer_dim[2], y_space)
        #     )
        # else:
        #     self.predict_net = nn.Sequential(
        #         nn.Linear(self.x_dim + self.a_emb_dim, layer_dim[2]),
        #         nn.ELU(),
        #         nn.Linear(layer_dim[2], y_space)
        #     )
        self.sigmd = torch.nn.Sigmoid()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ctr_classweight[1]))
        # self.sftcross = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([25.0,1.0])).float())
        self.sftcross = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(ctr_classweight)).float())

    def predict(self, x, a):
        if self.mode == 'Noweight':
            if self.x_dim > 1:
                # print(x.size(), self.x_emb_dim)
                user_emb_lookup = self.user_embedding(x).reshape([x.size()[0], self.x_emb_dim])
            else:
                user_emb_lookup = self.user_embedding_lookup(x).reshape([x.size()[0], self.x_emb_dim])
        elif self.mode in ['IPM_Embedding', 'Catboost', 'Adversarial']:
            if self.is_debias:
                user_emb_lookup = x.reshape([x.size()[0], self.x_emb_dim])
            else:
                if self.x_dim == 1:
                    user_emb_lookup = self.user_embedding_lookup(x).reshape([x.size()[0], self.x_emb_dim])
                else:
                    user_emb_lookup = x.reshape([x.size()[0], self.x_dim])

        if self.a_dim > 1:
            item_emb_lookup = self.item_embedding(a).reshape([a.size()[0], self.a_emb_dim])
        else:
            item_emb_lookup = self.item_embedding_lookup(a).reshape([a.size()[0], self.a_emb_dim])
        # user_emb = self.user_net(user_emb_lookup).reshape([user_emb_lookup.size()[0], self.x_emb_dim])
        # item_emb = self.item_net(item_emb_lookup).reshape([user_emb_lookup.size()[0], self.a_emb_dim])
        # print(user_emb.size(), item_emb.size())
        # return self.predict_net(torch.cat((user_emb, item_emb),	1))
        # return self.predict_net(torchuser_emb_lookup, item_emb_lookup)
        # print(user_emb_lookup)
        return self.predict_net(torch.cat((user_emb_lookup, item_emb_lookup), 1))
        # return self.predict_net(user_emb_lookup)

    def loss(self, x, a, y):
        y = torch.tensor(y, dtype=torch.float32).to(device)
        return self.bce(self.predict(x, a).reshape(-1), y)

    def weighted_loss(self, x, a, y, w):
        return w * self.bce(self.predict(x, a), y)


class NeuBPR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.layers = [int(l) for l in args.layers.split('|')]
        # self.layers = args.layers
        if args.user_dim == 1:
            self.W_mlp = torch.nn.Embedding(num_embeddings=args.user_item_size[0], embedding_dim=args.user_emb_dim)
            self.W_mf = torch.nn.Embedding(num_embeddings=args.user_item_size[0], embedding_dim=args.user_emb_dim)
        else:
            self.W_mlp = torch.nn.Linear(self.args.user_dim, self.args.user_emb_dim)
            self.W_mf = torch.nn.Linear(self.args.user_dim, self.args.user_emb_dim)
        if args.item_dim == 1:
            self.H_mlp = torch.nn.Embedding(num_embeddings=args.user_item_size[1], embedding_dim=args.item_emb_dim)
            self.H_mf = torch.nn.Embedding(num_embeddings=args.user_item_size[1], embedding_dim=args.item_emb_dim)
        else:
            self.H_mlp = torch.nn.Linear(self.args.item_dim, self.args.item_emb_dim)
            self.H_mf = torch.nn.Linear(self.args.item_dim, self.args.item_emb_dim)

        nn.init.xavier_normal_(self.W_mlp.weight.data)
        nn.init.xavier_normal_(self.H_mlp.weight.data)
        nn.init.xavier_normal_(self.W_mf.weight.data)
        nn.init.xavier_normal_(self.H_mf.weight.data)

        if self.args.downstream == 'NeuBPR':
            self.fc_layers = torch.nn.ModuleList()
            for idx, (in_size, out_size) in enumerate(zip(self.args.ctr_layer_dims[:-1], self.args.ctr_layer_dims[1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))

            self.affine_output = torch.nn.Linear(in_features=self.args.ctr_layer_dims[-1] + args.user_emb_dim, out_features=1)

        elif self.args.downstream == 'gmfBPR':
            self.affine_output = torch.nn.Linear(in_features=args.user_emb_dim, out_features=1)

        elif self.args.downstream == 'mlpBPR':
            self.fc_layers = torch.nn.ModuleList()
            for idx, (in_size, out_size) in enumerate(zip(self.args.ctr_layer_dims[:-1], self.args.ctr_layer_dims[1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))

            self.affine_output = torch.nn.Linear(in_features=self.args.ctr_layer_dims[-1], out_features=1)

        self.logistic = torch.nn.Sigmoid()
        self.weight_decay = args.weight_decay
        self.dropout = torch.nn.Dropout(p=args.dropout)

        self.sftcross = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.args.ctr_classweight[1]))


    def loss(self, u, i, y):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:s
            torch.FloatTensor
        """
        # print(u.size(), i.size())
        if self.args.is_debias:
            u = u.reshape(u.size()[0], self.args.user_emb_dim)
        else:
            if self.args.user_dim == 1:
                u = torch.tensor(u, dtype=torch.int64).to(device).reshape(u.size()[0], 1)
            else:
                u = torch.tensor(u, dtype=torch.float32).to(device).reshape([u.size()[0], self.args.user_dim])
        if self.args.item_dim == 1:
            i = torch.tensor(i, dtype=torch.int64).to(device).reshape([i.size()[0], self.args.item_dim])
        else:
            i = torch.tensor(i, dtype=torch.float32).to(device).reshape([i.size()[0], self.args.item_dim])
        y = torch.tensor(y, dtype=torch.float32).to(device)
        x_ui = self.predict(u, i, mode='dev')
        # x_uj = self.predict(u, j, mode='dev')
        # x_uij = x_ui - x_uj
        # -------------------------------Mengyue Yang---------------------------------
        # # log_prob = F.logsigmoid(x_uij).mean()
        # log_prob = F.logsigmoid(x_uij)
        if not self.args.is_debias:
            Wu_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
            Wu_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)

        Hi_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
        Hi_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)


        # print('***'*10)
        # print('x_uij', x_uij.size(), Wu_mlp.size(), Hi_mlp.size())
        # print('***'*10)

        # log_prob = F.logsigmoid(x_uij).mean()

        # if self.args.model_name == 'NeuBPR':
        # 	regularization = self.weight_decay * (Wu_mlp.norm(dim=1).pow(2).mean() + \
        # 		Wu_mf.norm(dim=1).pow(2).mean() + Hi_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hi_mf.norm(dim=1).pow(2).mean() + Hj_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hj_mf.norm(dim=1).pow(2).mean())
        # elif self.args.model_name in ['gmfBPR', 'bprBPR']:
        # 	regularization = self.weight_decay * (Wu_mf.norm(dim=1).pow(2).mean() + \
        # 		Hi_mf.norm(dim=1).pow(2).mean() + Hj_mf.norm(dim=1).pow(2).mean())
        # elif self.args.model_name == 'mlpBPR':
        # 	regularization = self.weight_decay * (Wu_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hi_mlp.norm(dim=1).pow(2).mean() + Hj_mlp.norm(dim=1).pow(2).mean())

        log_prob = self.sftcross(x_ui, y)

        # -----------------------------------------------------------------------
        if self.args.downstream == 'NeuBPR':
            if self.args.is_debias:
                regularization = self.weight_decay * (Hi_mlp.norm(dim=1) + Hi_mf.norm(dim=1))
            else:
                regularization = self.weight_decay * (Wu_mlp.norm(dim=1) + Wu_mf.norm(dim=1) + Hi_mlp.norm(dim=1) + Hi_mf.norm(dim=1))
        elif self.args.downstream in ['gmfBPR', 'bprBPR']:
            if self.args.is_debias:
                regularization = self.weight_decay * (Hi_mf.norm(dim=1))
            else:
                regularization = self.weight_decay * (Wu_mf.norm(dim=1) + Hi_mf.norm(dim=1))
        elif self.args.downstream == 'mlpBPR':
            if self.args.is_debias:
                regularization = self.weight_decay * (Hi_mlp.norm(dim=1))
            else:
                regularization = self.weight_decay * (Wu_mlp.norm(dim=1) + Hi_mlp.norm(dim=1))
        # ------------------------------------------------------------------------
        return (log_prob + regularization).mean()

    # ----------------------------Quanyu Dai----------------------------------
    def predict(self, u, i, mode='test'):
        #
        # if mode == 'test':
        #     u = torch.tensor(u, dtype=torch.int64).to(device).reshape(u.shape[0], 1)
        #     i = torch.tensor(i, dtype=torch.int64).to(device).reshape(i.shape[0], 1)

        if self.args.downstream == 'NeuBPR':
            if self.args.is_debias:
                user_embedding_mlp = u.reshape(u.size()[0], self.args.user_emb_dim)
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
            mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

            for idx, _ in enumerate(range(len(self.fc_layers))):
                mlp_vector = self.fc_layers[idx](mlp_vector)
                mlp_vector = torch.nn.ReLU()(mlp_vector)

            vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        elif self.args.downstream == 'gmfBPR':
            if self.args.is_debias:
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.mul(user_embedding_mf, item_embedding_mf)

        elif self.args.downstream == 'bprBPR':
            if self.args.is_debias:
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.mul(user_embedding_mf, item_embedding_mf)

        elif self.args.downstream == 'mlpBPR':
            if self.args.is_debias:
                user_embedding_mlp = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector

            for idx, _ in enumerate(range(len(self.fc_layers))):
                vector = self.fc_layers[idx](vector)
                vector = torch.nn.ReLU()(vector)
                vector = self.dropout(vector)



        # print('###'*10)
        # print('user_emb, item_emb, vector', user_embedding_mf.size(), item_embedding_mf.size(), vector.size())
        # print('###'*10)

        if self.args.downstream in ['NeuBPR', 'gmfBPR', 'mlpBPR']:
            logits = self.affine_output(vector)
            rating = logits.reshape(logits.size()[0])
        elif self.args.downstream == 'bprBPR':
            rating = vector.sum(dim=1)
            rating = rating.reshape(rating.size()[0])

        if mode == 'test':
            # rating = self.logistic(rating)
            rating = rating#.detach().cpu().numpy()

        # print('rating', rating.shape, rating)

        return rating

    # ------------------------------------------------------------------------

    def load_pretrain_weights(self, gmf_model, mlp_model):
        """Loading weights from trained MLP model & GMF model for NeuBPR"""

        self.W_mlp.weight.data = mlp_model.W_mlp.weight.data
        self.H_mlp.weight.data = mlp_model.H_mlp.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        self.W_mf.weight.data = gmf_model.W_mf.weight.data
        self.H_mf.weight.data = gmf_model.H_mf.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat(
            [mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)

# class DoubleModelCTR(nn.Module):
# 	def __init__(self, mode, x_dim, a_dim, emb_dim1, emb_dim2, layer_dim, y_space, x_a_size=None):
# 		super().__init__()
# 		self.name = 'CTR'
# 		self.x_dim = x_dim
# 		self.a_dim = a_dim
# 		self.mode = mode
# 		self.x_emb_dim = emb_dim1
# 		self.a_emb_dim = emb_dim2
# 		self.x_a_size = x_a_size
# 		# print(self.x_dim)
# 		self.user_embedding_lookup = nn.Embedding(self.x_a_size[0], self.x_emb_dim)
# 		self.user_embedding = nn.Sequential(
# 			nn.Linear(self.x_dim, self.x_emb_dim)
# 			# nn.ReLU(),
# 			# nn.Linear(layer_dim[0], self.x_emb_dim),
# 		)
# 		if self.a_dim == 1:
# 			# print("DoubleModelCTR", self.x_a_size[1], self.a_emb_dim)
# 			self.item_embedding_lookup = nn.Embedding(self.x_a_size[1], self.a_emb_dim)
# 		else:
# 			self.item_embedding = nn.Sequential(
# 				nn.Linear(self.a_dim, self.a_emb_dim)
# 			)
# 		self.predict_net = nn.Sequential(
# 			nn.Linear(self.x_emb_dim + self.a_emb_dim, layer_dim[2]),
# 			nn.ReLU(),
# 			nn.Linear(layer_dim[2], y_space)
# 		)
# 		self.sigmd = torch.nn.Sigmoid()
# 		self.bce = torch.nn.BCELoss()
# 		# self.sftcross = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([25.0,1.0])).float())
# 		self.sftcross = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1,10.0])).float())

# 	def predict(self, x, a):
# 		# if self.mode != 'IPM_Embedding':
# 		# 	if self.x_dim > 1:
# 		# 		user_emb_lookup = self.user_embedding(x).reshape([x.size()[0], self.x_emb_dim])
# 		# 	else:
# 		# 		user_emb_lookup = self.user_embedding_lookup(x).reshape([x.size()[0], self.x_emb_dim])
# 		# else:
# 		user_emb_lookup = x.reshape([x.size()[0], self.x_emb_dim])
# 		if self.a_dim > 1:
# 			item_emb_lookup = self.item_embedding(a).reshape([a.size()[0], self.a_emb_dim])
# 		else:
# 			item_emb_lookup = self.item_embedding_lookup(a).reshape([a.size()[0], self.a_emb_dim])
# 		# user_emb = self.user_net(user_emb_lookup).reshape([user_emb_lookup.size()[0], self.x_emb_dim])
# 		# item_emb = self.item_net(item_emb_lookup).reshape([user_emb_lookup.size()[0], self.a_emb_dim])
# 		# print(user_emb.size(), item_emb.size())
# 		# return self.predict_net(torch.cat((user_emb, item_emb),	1))

# 		return self.predict_net(torch.cat((user_emb_lookup, item_emb_lookup),1))

# 	def loss(self, x, a, y):
# 		return self.sftcross(self.predict(x,a), y.squeeze_())

# 	def weighted_loss(self, x, a, y):
# 		return w*self.sftcross(self.predict(x,a), y)
