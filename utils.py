import os
import torch
import numpy as np

def build_path(path):
    path_levels = path.split('/')
    cur_path = ""
    for path_seg in path_levels:
        if len(cur_path):
            cur_path = cur_path + "/" + path_seg
        else:
            cur_path = path_seg
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)

def get_label(data, order, offset, label_dim):
    output = []
    for i in order:
        output.append(data[i][offset:offset+label_dim])
    output = np.array(output, dtype="int") 
    return output

def get_feat(data, order, meta_offset, label_dim, feature_dim):
    output = []
    meta_output = []
    offset = meta_offset + label_dim
    for i in order:
        meta_output.append(data[i][:meta_offset])
        output.append(data[i][offset:offset + feature_dim])
    output = np.array(output, dtype="float32") 
    meta_output = np.array(meta_output, dtype="float32")
    return np.concatenate([output, meta_output], axis=1)

def log_normal(x, m, v):
    log_prob = (-0.5 * (torch.log(v) + (x-m).pow(2) / v)).sum(-1)
    return log_prob

def log_normal_mixture(z, m, v, mask=None):
    m = m.unsqueeze(0).expand(z.shape[0], -1, -1)
    v = v.unsqueeze(0).expand(z.shape[0], -1, -1)
    batch, mix, dim = m.size()
    z = z.view(batch, 1, dim).expand(batch, mix, dim)
    indiv_log_prob = log_normal(z, m, v) + torch.ones_like(mask)*(-1e6)*(1.-mask)
    log_prob = log_mean_exp(indiv_log_prob, mask)
    return log_prob

def log_mean_exp(x, mask):
    return log_sum_exp(x, mask) - torch.log(mask.sum(1))

def log_sum_exp(x, mask):
    max_x = torch.max(x, 1)[0]
    new_x = x - max_x.unsqueeze(1).expand_as(x)
    return max_x + (new_x.exp().sum(1)).log()

THRESHOLDS = []
for i in range(1, 10):
    THRESHOLDS.append(i / 10.)


def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats
