import argparse
import time

import pandas as pd
from scCDAN import scCDAN
import os


def preprocess(args):
    dataset_path = args.dataset_path  #"../processed_data/"
    print("dataset_path: ", dataset_path)
    normcounts = pd.read_csv(dataset_path + 'combine_expression.csv',index_col=0)
    # print(normcounts.T)
    labels = pd.read_csv(dataset_path + 'combine_labels.csv',index_col=0)
    # print(labels)
    domain_labels = pd.read_csv(dataset_path + 'domain_labels.csv',index_col=0)
    # print(domain_labels)
    data_set = {'features': normcounts.values, 'labels': labels.iloc[:, 0].values,
               'accessions': domain_labels.iloc[:, 0].values}
    return data_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scAdapt: virtual adversarial domain adaptation network')
    parser.add_argument('--method', type=str, default='DANN', choices=['DANN', 'mmd'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding_size')
    parser.add_argument('--margin', type=float, default=0.5, help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--source_name', type=str, default='Baron_human')
    parser.add_argument('--target_name', type=str, default='Baron_mouse')
    parser.add_argument('--dataset_path', type=str, default='E:/work-2/1-代码-/scCDAN - 12.12/scCDAN-master/scCDAN/processed_data/Baron_human_Baron_mouse/')
    parser.add_argument('--result_path', type=str,default='D:/experiments/new-work2/scAdapt-master-triplet_loss+center loss(2)/scAdapt-master/scAdapt/results/')
    parser.add_argument('--num_iterations', type=int, default=50010, help="num_iterations")
    parser.add_argument('--BNM_coeff', type=float, default=0.2, help="regularization coefficient for BNM loss")
    parser.add_argument('--centerloss_coeff', type=float, default=1.0,  help='regularization coefficient for center loss')
    parser.add_argument('--DA_coeff', type=float, default=1.0, help="regularization coefficient for domain alignment loss")
    parser.add_argument('--pseudo_th', type=float, default=0.0, help='pseudo_th')
    parser.add_argument('--cell_th', type=int, default=20, help='cell_th')
    # 默认
    # parser.add_argument('--epoch_th', type=int, default=10000, help='epoch_th')
    parser.add_argument('--epoch_th', type=int, default=15000, help='epoch_th')
    # parser.add_argument('--epoch_xx', type=int, default=5000, help='epoch_th')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient for VAT loss')
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0,1', help="device id to run")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  #'0,1,2,3'
    print(args)
    data_set = preprocess(args)
    s = time.time()
    scCDAN(args, data_set=data_set)
    t = time.time()
    print("时间：",t-s)
