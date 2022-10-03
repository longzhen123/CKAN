from src.CKAN import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='数据集')
    # parser.add_argument('--lr', type=float, default=1e-2, help='学习率')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2正则化系数')
    # parser.add_argument('--batch_size', type=int, default=1024, help='批量大小')
    # parser.add_argument('--epochs', type=int, default=30, help='迭代次数')
    # parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    # parser.add_argument('--dim', type=int, default=16, help='嵌入维度')
    # parser.add_argument('--L', type=int, default=2, help='L')
    # parser.add_argument('--K_l', type=int, default=8, help='集合大小')
    # parser.add_argument('--agg', type=str, default='concat', help='聚合器')
    # parser.add_argument('--ratio', type=float, default=1, help='训练集使用百分比')
    # parser.add_argument('--topk', type=int, default=10, help='top K')

    # parser.add_argument('--dataset', type=str, default='book', help='数据集')
    # parser.add_argument('--lr', type=float, default=1e-2, help='学习率')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2正则化系数')
    # parser.add_argument('--batch_size', type=int, default=1024, help='批量大小')
    # parser.add_argument('--epochs', type=int, default=50, help='迭代次数')
    # parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    # parser.add_argument('--dim', type=int, default=16, help='嵌入维度')
    # parser.add_argument('--L', type=int, default=2, help='L')
    # parser.add_argument('--K_l', type=int, default=8, help='集合大小')
    # parser.add_argument('--agg', type=str, default='concat', help='聚合器')
    # parser.add_argument('--ratio', type=float, default=1, help='训练集使用百分比')
    # parser.add_argument('--topk', type=int, default=10, help='top K')

    # parser.add_argument('--dataset', type=str, default='ml', help='数据集')
    # parser.add_argument('--lr', type=float, default=1e-2, help='学习率')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2正则化系数')
    # parser.add_argument('--batch_size', type=int, default=1024, help='批量大小')
    # parser.add_argument('--epochs', type=int, default=100, help='迭代次数')
    # parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    # parser.add_argument('--dim', type=int, default=16, help='嵌入维度')
    # parser.add_argument('--L', type=int, default=2, help='L')
    # parser.add_argument('--K_l', type=int, default=8, help='集合大小')
    # parser.add_argument('--agg', type=str, default='concat', help='聚合器')
    # parser.add_argument('--ratio', type=float, default=1, help='训练集使用百分比')
    # parser.add_argument('--topk', type=int, default=10, help='top K')

    parser.add_argument('--dataset', type=str, default='yelp', help='数据集')
    parser.add_argument('--lr', type=float, default=1e-2, help='学习率')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2正则化系数')
    parser.add_argument('--batch_size', type=int, default=1024, help='批量大小')
    parser.add_argument('--epochs', type=int, default=100, help='迭代次数')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--dim', type=int, default=16, help='嵌入维度')
    parser.add_argument('--L', type=int, default=2, help='L')
    parser.add_argument('--K_l', type=int, default=8, help='集合大小')
    parser.add_argument('--agg', type=str, default='concat', help='聚合器')
    parser.add_argument('--ratio', type=float, default=1, help='训练集使用百分比')
    parser.add_argument('--topk', type=int, default=10, help='top K')

    args = parser.parse_args()

    train(args, True)
