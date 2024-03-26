import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_list():
    return loss_list.keys()


def get_loss_fn(args):
    return loss_list[args.loss]


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class HProxy_Anchor(torch.nn.Module):
    def __init__(self, num_cls, num_obj, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(num_cls*num_obj, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = num_cls*num_obj
        self.num_obj = num_obj
        self.num_cls = num_cls
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, embeddings, style_Ind, object_Ind, style_id, object_id, s_o_id, w_1, w_2):
        P = self.proxies
        P0 = torch.cat([torch.mean(P[self.num_obj*i:self.num_obj*(i+1)], dim=0, keepdim=True) for i in range(self.num_cls)],dim=0)
        soft_loss_s = F.cross_entropy(style_Ind, style_id)
        soft_loss_o = F.cross_entropy(object_Ind, object_id)
        soft_loss = soft_loss_s + soft_loss_o

        cos0 = F.linear(l2_norm(embeddings), l2_norm(P0))  # Calcluate cosine similarity
        P_one_hot0 = binarize(T=style_id, nb_classes=self.num_cls)
        N_one_hot0 = 1 - P_one_hot0
        pos_exp0 = torch.exp(-self.alpha * (cos0 - self.mrg))
        neg_exp0 = torch.exp(self.alpha * (cos0 + self.mrg))

        with_pos_proxies0 = torch.nonzero(P_one_hot0.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies0 = len(with_pos_proxies0)  # The number of positive proxies

        P_sim_sum0 = torch.where(P_one_hot0 == 1, pos_exp0, torch.zeros_like(pos_exp0)).sum(dim=0)
        N_sim_sum0 = torch.where(N_one_hot0 == 1, neg_exp0, torch.zeros_like(neg_exp0)).sum(dim=0)

        pos_term0 = torch.log(1 + P_sim_sum0).sum() / num_valid_proxies0
        neg_term0 = torch.log(1 + N_sim_sum0).sum() / self.num_cls

        cos = F.linear(l2_norm(embeddings), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=s_o_id, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        p_loss = pos_term + neg_term
        hp_loss = pos_term0 + neg_term0

        loss = p_loss + w_1*hp_loss + w_2*soft_loss

        return loss, p_loss, hp_loss, soft_loss_o, soft_loss_s


loss_list = {
    'hproxy_anchor': HProxy_Anchor,
}
