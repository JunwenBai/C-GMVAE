import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import log_normal, log_normal_mixture

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(p=args.keep_prob)
        
        """Feature encoder"""
        self.fx = nn.Sequential(
            nn.Linear(args.feature_dim, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout
        )
        self.fx_mu = nn.Linear(256, args.latent_dim)
        self.fx_logvar = nn.Linear(256, args.latent_dim)

        """Label encoder"""
        self.label_lookup = nn.Linear(args.label_dim, args.emb_size)
        self.fe = nn.Sequential(
            nn.Linear(args.emb_size, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout
        )
        self.fe_mu = nn.Linear(256, args.latent_dim)
        self.fe_logvar = nn.Linear(256, args.latent_dim)

        """Decoder"""
        self.fd = nn.Sequential(
            nn.Linear(args.feature_dim + args.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, args.emb_size),
            nn.LeakyReLU()
        )

    def label_encode(self, x):
        h0 = self.dropout(F.relu(self.label_lookup(x)))
        h = self.fe(h0)
        mu = self.fe_mu(h)
        logvar = self.fe_logvar(h)
        fe_output = {
            'fe_mu': mu,
            'fe_logvar': logvar
        }
        return fe_output

    def feat_encode(self, x):
        h = self.fx(x)
        mu = self.fx_mu(h)
        logvar = self.fx_logvar(h)
        fx_output = {
            'fx_mu': mu,
            'fx_logvar': logvar
        }
        return fx_output

    def decode(self, z):
        d = self.fd(z)
        d = F.normalize(d, dim=1)
        return d

    def label_forward(self, x, feat):
        n_label = x.shape[1]
        all_labels = torch.eye(n_label).to(device)
        fe_output = self.label_encode(all_labels)
        mu = fe_output['fe_mu']
        
        z = torch.matmul(x, mu) / x.sum(1, keepdim=True)
        label_emb = self.decode(torch.cat((feat, z), 1))

        fe_output['label_emb'] = label_emb
        return fe_output

    def feat_forward(self, x):
        fx_output = self.feat_encode(x)
        mu = fx_output['fx_mu']
        logvar = fx_output['fx_logvar']

        if not self.training:
            z = mu
            z2 = mu
        else:
            z = reparameterize(mu, logvar)
            z2 = reparameterize(mu, logvar)
        feat_emb = self.decode(torch.cat((x, z), 1))
        feat_emb2 = self.decode(torch.cat((x, z2), 1))
        fx_output['feat_emb'] = feat_emb
        fx_output['feat_emb2'] = feat_emb2
        return fx_output

    def forward(self, label, feature):
        fe_output = self.label_forward(label, feature)
        label_emb = fe_output['label_emb']
        fx_output = self.feat_forward(feature)
        feat_emb, feat_emb2 = fx_output['feat_emb'], fx_output['feat_emb2']

        embs = self.label_lookup.weight
        label_out = torch.matmul(label_emb, embs)
        feat_out = torch.matmul(feat_emb, embs)
        feat_out2 = torch.matmul(feat_emb2, embs)
        
        fe_output.update(fx_output)
        output = fe_output
        output['embs'] = embs
        output['label_out'] = label_out
        output['feat_out'] = feat_out
        output['feat_out2'] = feat_out2
        output['feat'] = feature
        return output


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


def compute_loss(input_label, output, args=None):
    fe_out, fe_mu, fe_logvar, label_emb = \
        output['label_out'], output['fe_mu'], output['fe_logvar'], output['label_emb']
    fx_out, fx_mu, fx_logvar, feat_emb = \
        output['feat_out'], output['fx_mu'], output['fx_logvar'], output['feat_emb']
    fx_out2 = output['feat_out2']
    embs = output['embs']

    fx_sample = reparameterize(fx_mu, fx_logvar)
    fx_var = torch.exp(fx_logvar)
    fe_var = torch.exp(fe_logvar)
    kl_loss = (log_normal(fx_sample, fx_mu, fx_var) - \
        log_normal_mixture(fx_sample, fe_mu, fe_var, input_label)).mean()

    pred_e = torch.sigmoid(fe_out)
    pred_x = torch.sigmoid(fx_out)
    pred_x2 = torch.sigmoid(fx_out2)

    def compute_BCE_and_RL_loss(E):
        #compute negative log likelihood (BCE loss) for each sample point
        sample_nll = -(
            torch.log(E) * input_label + torch.log(1 - E) * (1 - input_label)
        )
        logprob = -torch.sum(sample_nll, dim=2)

        #the following computation is designed to avoid the float overflow (log_sum_exp trick)
        maxlogprob = torch.max(logprob, dim=0)[0]
        Eprob = torch.mean(torch.exp(logprob - maxlogprob), axis=0)
        nll_loss = torch.mean(-torch.log(Eprob) - maxlogprob)
        return nll_loss

    def supconloss(label_emb, feat_emb, embs, temp=1.0):
        features = torch.cat((label_emb, feat_emb))
        labels = torch.cat((input_label, input_label)).float()
        n_label = labels.shape[1]
        emb_labels = torch.eye(n_label).to(device)
        mask = torch.matmul(labels, emb_labels)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, embs),
            temp)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

    nll_loss = compute_BCE_and_RL_loss(pred_e.unsqueeze(0))
    nll_loss_x = compute_BCE_and_RL_loss(pred_x.unsqueeze(0))
    nll_loss_x2 = compute_BCE_and_RL_loss(pred_x2.unsqueeze(0))
    sum_nll_loss = nll_loss + nll_loss_x + nll_loss_x2
    cpc_loss = supconloss(label_emb, feat_emb, embs)
    total_loss = sum_nll_loss * args.nll_coeff + kl_loss * 6. + cpc_loss
    return total_loss, nll_loss, nll_loss_x, 0., 0., kl_loss, cpc_loss, pred_e, pred_x
