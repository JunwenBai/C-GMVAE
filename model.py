import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from losses.losses import SupConLoss
from btcvae import KL
from utils import log_normal, log_normal_mixture, imq_kernel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        # feature layers
        input_dim = args.feature_dim+args.meta_offset
        self.fx1 = nn.Linear(input_dim, 256)
        self.fx2 = nn.Linear(256, 512)
        self.fx3 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, args.latent_dim)
        self.fx_logvar = nn.Linear(256, args.latent_dim)

        self.emb_size = args.emb_size

        self.fd_x1 = nn.Linear(input_dim+args.latent_dim, 512)
        self.fd_x2 = torch.nn.Sequential(
            nn.Linear(512, self.emb_size)
        )
        self.feat_mp_mu = nn.Linear(self.emb_size, args.label_dim)

        self.recon = torch.nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        self.label_recon = torch.nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.emb_size),
            nn.LeakyReLU()
        )

        # label layers
        self.fe0 = nn.Linear(args.label_dim, self.emb_size)
        self.fe1 = nn.Linear(self.emb_size, 512)
        self.fe2 = nn.Linear(512, 256)
        self.fe_mu = nn.Linear(256, args.latent_dim)
        self.fe_logvar = nn.Linear(256, args.latent_dim)

        self.fd1 = self.fd_x1
        self.fd2 = self.fd_x2
        self.label_mp_mu = self.feat_mp_mu

        self.bias = nn.Parameter(torch.zeros(args.label_dim))

        assert id(self.fd_x1) == id(self.fd1)
        assert id(self.fd_x2) == id(self.fd2)

        # things they share
        self.dropout = nn.Dropout(p=args.keep_prob)
        self.scale_coeff = args.scale_coeff

    def label_encode(self, x):
        h0 = self.dropout(F.relu(self.fe0(x)))
        h1 = self.dropout(F.relu(self.fe1(h0)))
        h2 = self.dropout(F.relu(self.fe2(h1)))
        mu = self.fe_mu(h2) * self.scale_coeff
        logvar = self.fe_logvar(h2) * self.scale_coeff
        fe_output = {
            'fe_mu': mu,
            'fe_logvar': logvar
        }
        return fe_output

    def feat_encode(self, x):
        h1 = self.dropout(F.relu(self.fx1(x)))
        h2 = self.dropout(F.relu(self.fx2(h1)))
        h3 = self.dropout(F.relu(self.fx3(h2)))
        mu = self.fx_mu(h3) * self.scale_coeff
        logvar = self.fx_logvar(h3) * self.scale_coeff
        fx_output = {
            'fx_mu': mu,
            'fx_logvar': logvar
        }
        
        return fx_output

    def label_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def feat_reparameterize(self, mu, logvar, coeff=1.0):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def label_decode(self, z):
        d1 = F.relu(self.fd1(z))
        d2 = F.leaky_relu(self.fd2(d1))
        d3 = F.normalize(d2, dim=1)
        return d3

    def feat_decode(self, z):
        d1 = F.relu(self.fd_x1(z))
        d2 = F.leaky_relu(self.fd_x2(d1))
        d3 = F.normalize(d2, dim=1)
        return d3

    def label_forward(self, x, feat):
        if self.args.reg == "gmvae":
            n_label = x.shape[1]
            all_labels = torch.eye(n_label).to(x.device)
            fe_output = self.label_encode(all_labels)
        else:
            fe_output = self.label_encode(x)
        mu = fe_output['fe_mu']
        logvar = fe_output['fe_logvar']
        
        if self.args.reg == "wae" or not self.training:
            if self.args.reg == "gmvae":
                z = torch.matmul(x, mu) / x.sum(1, keepdim=True)
            else:
                z = mu
        else:
            if self.args.reg == "gmvae":
                z = torch.matmul(x, mu) / x.sum(1, keepdim=True)
            else:
                z = self.label_reparameterize(mu, logvar)
        label_emb = self.label_decode(torch.cat((feat, z), 1))
        single_label_emb = F.normalize(self.label_recon(mu), dim=1)

        fe_output['label_emb'] = label_emb
        fe_output['single_label_emb'] = single_label_emb
        return fe_output

    def feat_forward(self, x):
        fx_output = self.feat_encode(x)
        mu = fx_output['fx_mu']
        logvar = fx_output['fx_logvar']

        if self.args.reg == "wae" or not self.training:
            if self.args.test_sample:
                z = self.feat_reparameterize(mu, logvar)
                z2 = self.feat_reparameterize(mu, logvar)
            else:
                z = mu
                z2 = mu
        else:
            z = self.feat_reparameterize(mu, logvar)
            z2 = self.feat_reparameterize(mu, logvar)
        feat_emb = self.feat_decode(torch.cat((x, z), 1))
        feat_emb2 = self.feat_decode(torch.cat((x, z2), 1))
        fx_output['feat_emb'] = feat_emb
        fx_output['feat_emb2'] = feat_emb2

        feat_recon = self.recon(z)
        fx_output['feat_recon'] = feat_recon
        return fx_output

    def forward(self, label, feature):
        fe_output = self.label_forward(label, feature)
        label_emb, single_label_emb = fe_output['label_emb'], fe_output['single_label_emb']
        fx_output = self.feat_forward(feature)
        feat_emb, feat_emb2 = fx_output['feat_emb'], fx_output['feat_emb2']
        embs = self.fe0.weight
        
        label_out = torch.matmul(label_emb, embs)
        single_label_out = torch.matmul(single_label_emb, embs)
        feat_out = torch.matmul(feat_emb, embs)
        feat_out2 = torch.matmul(feat_emb2, embs)
        
        fe_output.update(fx_output)
        output = fe_output
        output['embs'] = embs
        output['label_out'] = label_out
        output['single_label_out'] = single_label_out
        output['feat_out'] = feat_out
        output['feat_out2'] = feat_out2
        output['feat'] = feature

        return output

def build_multi_classification_loss(predictions, labels):
    shape = tuple(labels.shape)
    labels = labels.float()
    y_i = torch.eq(labels, torch.ones(shape).to(device))
    y_not_i = torch.eq(labels, torch.zeros(shape).to(device))

    truth_matrix = pairwise_and(y_i, y_not_i).float()
    sub_matrix = pairwise_sub(predictions, predictions)
    exp_matrix = torch.exp(-5*sub_matrix)
    sparse_matrix = exp_matrix * truth_matrix
    sums = torch.sum(sparse_matrix, dim=[2,3])
    y_i_sizes = torch.sum(y_i.float(), dim=1)
    y_i_bar_sizes = torch.sum(y_not_i.float(), dim=1)
    normalizers = y_i_sizes * y_i_bar_sizes
    loss = torch.div(sums, 5*normalizers) # 100*128  divide  128
    zero = torch.zeros_like(loss) # 100*128 zeros
    loss = torch.where(torch.logical_or(torch.isinf(loss), torch.isnan(loss)), zero, loss)
    loss = torch.mean(loss)
    return loss

def pairwise_and(a, b):
    column = torch.unsqueeze(a, 2)
    row = torch.unsqueeze(b, 1)
    return torch.logical_and(column, row)

def pairwise_sub(a, b):
    column = torch.unsqueeze(a, 3)
    row = torch.unsqueeze(b, 2)
    return column - row

def cross_entropy_loss(logits, labels, n_sample):
    labels = torch.tile(torch.unsqueeze(labels, 0), [n_sample, 1, 1])
    ce_loss = nn.BCEWithLogitsLoss(labels=labels, logits=logits)
    ce_loss = torch.mean(torch.sum(ce_loss, dim=1))
    return ce_loss

def compute_loss(input_label, output, args=None, epoch=0, class_weights=None):
    if args.reg == "gumbel":
        fe_out, fe_mu, fe_logvar, label_emb = output['label_out'], output['fe_mu'], output['fe_logvar'], output['label_emb']
        fx_out, fx_mu, fx_logvar, feat_emb = output['feat_out'], output['fx_mu'], output['fx_logvar'], output['feat_emb']
        fx_out2, single_label_out = output['feat_out2'], output['single_label_out']
        embs = output['embs']
        feat_recon, feat = output['feat_recon'], output['feat']
        losses = LossFunctions()
        fx_loss_cat = -losses.entropy(output['fx_gumbel_logits'], output['fx_gumbel_prob']) - np.log(0.1)
        fe_loss_cat = -losses.entropy(output['fe_gumbel_logits'], output['fe_gumbel_prob']) - np.log(0.1)
    else:
        fe_out, fe_mu, fe_logvar, label_emb = output['label_out'], output['fe_mu'], output['fe_logvar'], output['label_emb']
        fx_out, fx_mu, fx_logvar, feat_emb = output['feat_out'], output['fx_mu'], output['fx_logvar'], output['feat_emb']
        fx_out2, single_label_out = output['feat_out2'], output['single_label_out']
        embs = output['embs']

    feat_recon_loss = 0.

    latent_cpc_loss = 0.
    if args.reg == "gmvae":
        fe_sample = torch.matmul(input_label, fe_mu) / input_label.sum(1, keepdim=True)
        latent_cpc_loss = SupConLoss(temperature=0.1)(torch.stack([fe_sample, fx_mu], dim=1), input_label.float())
    else:
        latent_cpc_loss = SupConLoss(temperature=0.1)(torch.stack([fe_mu, fx_mu], dim=1), input_label.float())


    if args.reg == "vae":
        kl_loss = torch.mean(0.5*torch.sum((fx_logvar-fe_logvar)-1+torch.exp(fe_logvar-fx_logvar)+torch.square(fx_mu-fe_mu)/(torch.exp(fx_logvar)+1e-6), dim=1))
    elif args.reg == "btcae":
        std = torch.exp(0.5*fx_logvar)
        eps = torch.randn_like(std)
        fx_sample = fx_mu + eps*std
        kl_loss = KL((fe_mu, fe_logvar), fx_sample)
    elif args.reg == "wae":
        kl_loss = 0.
        for i in range(10):
            std = torch.exp(0.5*fx_logvar)
            eps = torch.randn_like(std)
            fx_sample = fx_mu + eps*std
            kl_loss += imq_kernel(fe_mu, fx_sample, h_dim=fx_mu.shape[1])
        kl_loss /= 10.
        kl_loss *= 5
    elif args.reg == "gmvae":
        std = torch.exp(0.5*fx_logvar)
        eps = torch.randn_like(std)
        fx_sample = fx_mu + eps*std
        fx_var = torch.exp(fx_logvar)
        fe_var = torch.exp(fe_logvar)
        kl_loss = (log_normal(fx_sample, fx_mu, fx_var) - log_normal_mixture(fx_sample, fe_mu, fe_var, input_label)).mean()

    def compute_BCE_and_RL_loss(E):
        #compute negative log likelihood (BCE loss) for each sample point
        sample_nll = -(torch.log(E)*input_label+torch.log(1-E)*(1-input_label))
        
        logprob=-torch.sum(sample_nll, dim=2)

        #the following computation is designed to avoid the float overflow (log_sum_exp trick)
        maxlogprob=torch.max(logprob, dim=0)[0]
        Eprob=torch.mean(torch.exp(logprob-maxlogprob), axis=0)
        nll_loss=torch.mean(-torch.log(Eprob)-maxlogprob)

        #c_loss = build_multi_classification_loss(E, input_label)
        return nll_loss

    def supconloss(label_emb, feat_emb, embs, temp=1.0, sample_wise=False):
        if sample_wise:
            loss_func = SupConLoss(temperature=0.1)
            return loss_func(torch.stack([label_emb, feat_emb], dim=1), input_label.float())

        features = torch.cat((label_emb, feat_emb))
        labels = torch.cat((input_label, input_label)).float()
        n_label = labels.shape[1]
        emb_labels = torch.eye(n_label).to(feat_emb.device)
        mask = torch.matmul(labels, emb_labels)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, embs),
            temp)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_neg = ((1.0-mask) * log_prob).sum(1) / (1.0-mask).sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss

    if not args.finetune:
        pred_e = torch.sigmoid(fe_out)
        pred_x = torch.sigmoid(fx_out)
        pred_x2 = torch.sigmoid(fx_out2)
        pred_single_label = torch.sigmoid(single_label_out)
        single_label_recon_loss = nn.BCELoss()(pred_single_label, torch.eye(pred_single_label.shape[1]).to(pred_single_label.device))
        
        nll_loss = compute_BCE_and_RL_loss(pred_e.unsqueeze(0))
        nll_loss_x = compute_BCE_and_RL_loss(pred_x.unsqueeze(0))
        nll_loss_x2 = compute_BCE_and_RL_loss(pred_x2.unsqueeze(0))
        cpc_loss = supconloss(label_emb, feat_emb, embs, sample_wise=args.cpc_sample_wise)
        total_loss = (nll_loss + nll_loss_x + nll_loss_x2) * args.nll_coeff + kl_loss*6. + (cpc_loss) #+ latent_cpc_loss
    else:
        pred_x = torch.sigmoid(fx_out)
        pred_e = torch.ones_like(pred_x)
        nll_loss_x, c_loss_x = compute_BCE_and_RL_loss(pred_x.unsqueeze(0))
        nll_loss, c_loss = 0., 0.
        total_loss = nll_loss_x * args.nll_coeff + c_loss_x * args.c_coeff + kl_loss

    return total_loss, nll_loss, nll_loss_x, 0., 0., kl_loss, cpc_loss, latent_cpc_loss, single_label_recon_loss, pred_e, pred_x

