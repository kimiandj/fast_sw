# Some files in this project were adapted from the following open source implementations:
# 1) https://github.com/ishansd/swg
# 2) https://github.com/maremun/swg
# 3) https://github.com/gmum/cwae-pytorch

import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from args_parser import parse_args
from utils import wasserstein1d, clt_sw, cov

from noise_creator import NoiseCreator
from factories.fid_evaluator_factory import create_fid_evaluator
from factories.dataset_factory import get_dataset
from factories.model_factory import get_model


def run(args, lmbda_1, lmbda_2, id_train):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    if args.dataset == 'mnist':
        imsize = 28
        c = 1
        d = 32
    elif args.dataset == 'celeba':
        imsize = 64
        c = 3
        d = 100

    x_dim = imsize * imsize * c
    dataroot = os.path.join(args.homedir, 'data')
    dataset = get_dataset(args.dataset, dataroot, train=True)
    # print(len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
    if args.loss == 'montecarlo':
        nprojections = 10000
    
    # Set model
    if args.use_disc:
        G, D = get_model(args.dataset, device, args.use_disc)
        d_opt = optim.Adam(D.parameters(), lr=args.lr)
        d_criterion = nn.BCEWithLogitsLoss()
        n_dopt = 1
        if args.dataset == 'mnist':
            f_dim = 256
    else:
        G = get_model(args.dataset, device, args.use_disc)
        n_dopt = 0
        f_dim = x_dim
    g_opt = optim.Adam(G.parameters(), lr=args.lr)

    z_fixed_ = torch.randn(81, d, device=device)
    
    # Train
    tot_time_epoch = 0
    tot_time_loss = 0
    for epoch in range(args.max_epochs):
        torch.cuda.synchronize()
        tstart_epoch = time()
        for i, (batch_x, _) in enumerate(train_loader):
            # Generator step
            G.train()
            x = batch_x.to(device) 
            z = torch.randn(batch_x.size(0), d, requires_grad=False, device=device)
            xpred = G(z)
            if args.dataset == 'mnist':
                x = x.view(-1, x_dim)        
                xpred = xpred.view(-1, x_dim)

            if args.loss == 'montecarlo':
                if args.use_disc:
                    _, fake_features = D(xpred)
                    _, true_features = D(x)

                    f_dim = fake_features.size(1)
                    
                    torch.cuda.synchronize()
                    tstart_loss = time()

                    theta = torch.randn(
                            (f_dim, nprojections),
                            requires_grad=False,
                            device=device
                            )
                    theta = theta / torch.norm(theta, dim=0)[None, :]

                    fake_1d = fake_features @ theta
                    true_1d = true_features @ theta
                    g_loss = wasserstein1d(fake_1d, true_1d)

                    torch.cuda.synchronize()
                    tot_time_loss += time() - tstart_loss
                else:
                    torch.cuda.synchronize()
                    tstart_loss = time()

                    theta = torch.randn(
                            (x_dim, nprojections),
                            requires_grad=False,
                            device=device
                            )
                    theta = theta / torch.norm(theta, dim=0)[None, :]

                    xpred_1d = xpred @ theta
                    x_1d = x @ theta
                    g_loss = wasserstein1d(xpred_1d, x_1d)

                    torch.cuda.synchronize()
                    tot_time_loss += time() - tstart_loss
            elif args.loss == 'clt':
                if args.use_disc: 
                    _, fake_features = D(xpred)
                    _, true_features = D(x)

                    f_dim = fake_features.size(1)

                    torch.cuda.synchronize()
                    tstart_loss = time()
                    g_loss = clt_sw(fake_features, true_features)
                    torch.cuda.synchronize()
                    tot_time_loss += time() - tstart_loss
                else:
                    torch.cuda.synchronize()
                    tstart_loss = time()
                    g_loss = clt_sw(xpred, x)
                    torch.cuda.synchronize()
                    tot_time_loss += time() - tstart_loss

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            # Discriminator step
            if args.use_disc:
                for _ in range(n_dopt):
                    z = torch.randn(batch_x.size(0), d, requires_grad=False, device=device)
                    xpred = G(z).detach()
                    fake_score, fake_features = D(xpred)
                    true_score, true_features = D(x)

                    d_loss_fake = d_criterion(fake_score, torch.zeros_like(fake_score))
                    d_loss_true = d_criterion(true_score, torch.ones_like(true_score))
                    d_loss = d_loss_fake.mean() + d_loss_true.mean()

                    if lmbda_1 > 0.0:
                        cov_fake = cov(fake_features)
                        cov_true = cov(true_features)
                        diag_fake = torch.diag(cov_fake)
                        diag_true = torch.diag(cov_true)
                        d_loss += lmbda_1 * (
                            torch.linalg.norm(cov_fake - diag_fake) ** 2 
                            + torch.linalg.norm(cov_true - diag_true) ** 2
                        )
                        if lmbda_2 > 0.0:
                            d_loss += lmbda_2 * (
                                1 / torch.linalg.norm(fake_features) ** 2 
                                + 1 / torch.linalg.norm(true_features) ** 2
                            )
                    d_opt.zero_grad()
                    d_loss.backward()
                    d_opt.step()
        torch.cuda.synchronize()
        tepoch = time() - tstart_epoch
        tot_time_epoch += tepoch
        status = 'Epoch %d iter %d: G loss %.4f time %.2f' % (epoch+1, i+1, g_loss.item(), tepoch)
        print(status)

    output_dir = os.path.join(args.homedir, 'results_cpu', args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    print('Created output dir: ', output_dir)

    # Save generated images
    samples = G(z_fixed_).detach()
    if args.dataset == 'mnist':
       samples = samples.reshape(81, imsize, imsize)
    else:
       samples = samples.permute(0, 2, 3, 1)
    fig = plt.figure(figsize=(16,16))
    for i in range(9):
       for j in range(9):
           plt.subplot(9, 9, i*9+j+1)
           if args.dataset == 'mnist':
               plt.imshow(samples[i*9+j].cpu().numpy(), cmap='gray')
           else:
               plt.imshow(samples[i*9+j].cpu().numpy())
           plt.axis('off')

    loss = args.loss
    if loss == 'montecarlo':
        loss += str(nprojections)
    filename = 'loss={}_bs={}_nepochs={}_lr={}'.format(loss, args.batch_size, args.max_epochs, args.lr)
    if args.use_disc:
        filename += '_ndopt=' + str(n_dopt) + '_fdim=' + str(f_dim)
    if lmbda_1 > 0.0:
        filename += '_lambda1=' + str(lmbda_1)
    if lmbda_2 > 0.0:
        filename += '_lambda2=' + str(lmbda_2)
    filename += '_idtrain=' + str(id_train)
    plt.savefig(os.path.join(output_dir, filename + '.pdf'))

    # Compute FID score
    noise_creator = NoiseCreator(d)
    fid_evaluator = create_fid_evaluator(os.path.join(args.homedir, 'fid', args.dataset + '_fid_stats.npz'), noise_creator)
    fid_score = fid_evaluator.evaluate(G, device)
    fid_fpath = os.path.join(output_dir, 'fid.csv')
    with open(fid_fpath, "a") as f:
        if os.stat(fid_fpath).st_size == 0:
            fieldnames = ['loss', 'bs', 'nepochs', 'lr', 'ndopt', 'fdim', 'lambda1', 'lambda2', 'fid', 'time_epoch', 'time_loss_per_epoch']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        writer = csv.writer(f)
        new_row = [loss, args.batch_size, args.max_epochs, args.lr, n_dopt, 
                   f_dim, lmbda_1, lmbda_2, fid_score, tot_time_epoch / args.max_epochs, tot_time_loss / args.max_epochs]
        writer.writerow(new_row)


if __name__ == '__main__':
    args = parse_args()
    n_trains = 5
    if args.cross_valid and args.use_disc:
        l1 = np.array([0.001, 0.01, 0.1, 1])
        l2 = np.array([0.0, 0.001, 0.01, 0.1, 1])

        for i in range(l1.shape[0]):
            l1i = l1[i]
            for j in range(l2.shape[0]):
                l2j = l2[j]
                for nt in range(n_trains):
                    run(args, l1i, l2j, id_train=int(nt))
    else:
        for nt in range(n_trains):
            run(args, args.lambda1_val, args.lambda2_val, id_train=int(nt))
