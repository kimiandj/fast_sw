import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='mnist | celeba')
    parser.add_argument('--homedir', default='/swg/', help='path to home')
    
    parser.add_argument('--loss', required=True, help='montecarlo | clt')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum number of epochs during training')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for the optimizer')

    parser.add_argument('--disc', dest='use_disc', action='store_true')
    parser.set_defaults(use_disc=False)
    
    parser.add_argument('--crossval', dest='cross_valid', action='store_true')
    parser.set_defaults(cross_valid=False)
    
    parser.add_argument('--lambda1_val', type=float, default=0.0, help='value of the regularization parameter for the off-diagonal coefficients of the covariance matrices')
    parser.add_argument('--lambda2_val', type=float, default=0.0, help='value of the regularization parameter for the norm of the samples')
    
    return parser.parse_args()
