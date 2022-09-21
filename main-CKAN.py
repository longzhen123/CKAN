from src.CKAN import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=100, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=16, help='embedding size')
    # parser.add_argument('--L', type=int, default=2, help='H')
    # parser.add_argument('--K_l', type=int, default=8, help='K_l')
    # parser.add_argument('--agg', type=str, default='concat', help='K')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=50, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=16, help='embedding size')
    # parser.add_argument('--L', type=int, default=2, help='H')
    # parser.add_argument('--K_l', type=int, default=8, help='K_l')
    # parser.add_argument('--agg', type=str, default='concat', help='K')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=16, help='embedding size')
    parser.add_argument('--L', type=int, default=2, help='H')
    parser.add_argument('--K_l', type=int, default=8, help='K_l')
    parser.add_argument('--agg', type=str, default='concat', help='K')
    parser.add_argument('--ratio', type=float, default=0.8, help='The proportion of training set used')
    #
    # parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=50, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=16, help='embedding size')
    # parser.add_argument('--L', type=int, default=2, help='H')
    # parser.add_argument('--K_l', type=int, default=8, help='K_l')
    # parser.add_argument('--agg', type=str, default='concat', help='K')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()

    train(args, False)
