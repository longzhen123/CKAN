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
    # parser.add_argument('--dim', type=int, default=30, help='embedding size')
    # parser.add_argument('--L', type=int, default=3, help='H')
    # parser.add_argument('--K_l', type=int, default=50, help='K_l')
    # parser.add_argument('--agg', type=str, default='concat', help='K')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=50, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=30, help='embedding size')
    # parser.add_argument('--L', type=int, default=1, help='H')
    # parser.add_argument('--K_l', type=int, default=5, help='K_l')
    # parser.add_argument('--agg', type=str, default='concat', help='K')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')
    #
    # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=100, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=30, help='embedding size')
    # parser.add_argument('--L', type=int, default=3, help='H')
    # parser.add_argument('--K_l', type=int, default=5, help='K_l')
    # parser.add_argument('--agg', type=str, default='concat', help='K')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')
    #
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=20, help='embedding size')
    parser.add_argument('--L', type=int, default=2, help='H')
    parser.add_argument('--K_l', type=int, default=30, help='K_l')
    parser.add_argument('--agg', type=str, default='concat', help='K')
    parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()

    train(args, True)

'''
music	train_auc: 0.910 	 train_acc: 0.500 	 eval_auc: 0.765 	 eval_acc: 0.500 	 test_auc: 0.765 	 test_acc: 0.500 		[0.11, 0.19, 0.32, 0.39, 0.39, 0.43, 0.46, 0.47]
book	train_auc: 0.731 	 train_acc: 0.500 	 eval_auc: 0.660 	 eval_acc: 0.500 	 test_auc: 0.656 	 test_acc: 0.500 		[0.07, 0.08, 0.22, 0.25, 0.25, 0.33, 0.35, 0.37]
ml	train_auc: 0.876 	 train_acc: 0.500 	 eval_auc: 0.844 	 eval_acc: 0.500 	 test_auc: 0.846 	 test_acc: 0.500 		[0.17, 0.28, 0.44, 0.46, 0.46, 0.55, 0.56, 0.57]
yelp	train_auc: 0.815 	 train_acc: 0.500 	 eval_auc: 0.759 	 eval_acc: 0.500 	 test_auc: 0.759 	 test_acc: 0.500 		[0.07, 0.16, 0.35, 0.35, 0.35, 0.43, 0.44, 0.45]

'''