import numpy as np
import torch
import sys
import datetime
from copy import deepcopy
import evals
from utils import build_path, get_label, get_feat, THRESHOLDS
from model import VAE, compute_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')


def test(args):
    METRICS = ['HA', 'ebF1', 'miF1', 'maF1', 'p_at_1']
    print('reading npy...')
    data = np.load(args.data_dir)
    test_idx = np.load(args.test_idx)
    print('reading completed')
    labels = get_label(data, test_idx, args.meta_offset, args.label_dim)
    #print(labels.sum(0))

    print('building network...')
    vae = VAE(args)
    vae.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    vae.eval()

    print("loaded model: %s" % (args.checkpoint_path))

    def test_step(test_idx):
        all_nll_loss = 0
        all_l2_loss = 0
        all_c_loss = 0
        all_total_loss = 0

        all_pred_x = []
        all_label = []
        all_indiv_max = []
        all_feat_mu = []
        all_label_mu = []

        sigma=[]
        real_batch_size=min(args.batch_size, len(test_idx))
        
        N_test_batch = int( (len(test_idx)-1)/real_batch_size ) + 1

        for i in range(N_test_batch):
            if i % 20 == 0:
                print("%.1f%% completed" % (i*100.0/N_test_batch))

            start = real_batch_size*i
            end = min(real_batch_size*(i+1), len(test_idx))

            input_feat = get_feat(data,test_idx[start:end], args.meta_offset, args.label_dim, args.feature_dim)
            input_label = get_label(data,test_idx[start:end], args.meta_offset, args.label_dim)

            input_feat, input_label = torch.from_numpy(input_feat).to(device), torch.from_numpy(input_label)
            input_label = deepcopy(input_label).float().to(device)

            with torch.no_grad():
                output = vae(input_label, input_feat) 
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, cpc_loss, _, pred_x = compute_loss(input_label, output, args)

            all_nll_loss += nll_loss*(end-start)
            all_c_loss += c_loss*(end-start)
            all_total_loss += total_loss*(end-start)

            if len(all_pred_x) == 0: 
                all_pred_x = pred_x.cpu().data.numpy()
                all_label = input_label.cpu().data.numpy()
                all_feat_mu = output['fx_mu'].cpu().data.numpy()
                all_label_mu = output['fe_mu'].cpu().data.numpy()
            else:
                all_pred_x = np.concatenate((all_pred_x, pred_x.cpu().data.numpy()))
                all_label = np.concatenate((all_label, input_label.cpu().data.numpy()))
                all_feat_mu= np.concatenate((all_feat_mu, output['fx_mu'].cpu().data.numpy()))
                all_label_mu= np.concatenate((all_label_mu, output['fe_mu'].cpu().data.numpy()))

        nll_loss = all_nll_loss / len(test_idx)
        c_loss = all_c_loss / len(test_idx)
        total_loss = all_total_loss / len(test_idx)
        return all_pred_x, all_label, all_feat_mu, all_label_mu

    pred_x, input_label, all_feat_mu, all_label_mu = test_step(test_idx)

    
    best_test_metrics = None
    for threshold in THRESHOLDS:
        test_metrics = evals.compute_metrics(pred_x, input_label, threshold, all_metrics=True)
        if best_test_metrics == None:
            best_test_metrics = {}
            for metric in METRICS:
                best_test_metrics[metric] = test_metrics[metric]
        else:
            for metric in METRICS:
                if 'FDR' in metric:
                    best_test_metrics[metric] = min(best_test_metrics[metric], test_metrics[metric])
                else:
                    best_test_metrics[metric] = max(best_test_metrics[metric], test_metrics[metric])


    print("****************")
    for metric in METRICS:
        print(metric, ":", best_test_metrics[metric])
    print("****************")
