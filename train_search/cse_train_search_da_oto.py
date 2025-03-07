Project_Path = 'XXXXX'
import sys
sys.path.append(Project_Path)

import os
import random
import logging
import time
import argparse
import utils
import torch

import numpy as np
import config.searchspace as ss
import torch.nn.functional as F

from dataset import load_dataset, load_dataset_config
from train.train_da_fun import train_da


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--dataset', type=str, default='SEED-IV', help='dataset name')
# search space
parser.add_argument('--N1_LIST', type=list, default=ss.N1_LIST)
parser.add_argument('--N2_LIST', type=list, default=ss.N2_LIST)
parser.add_argument('--N3_LIST', type=list, default=ss.N3_LIST)
parser.add_argument('--AF_LIST', type=list, default=ss.AF_LIST)
# cross session task
parser.add_argument('--ss', type=int, default=1, help='source session')
parser.add_argument('--ts', type=int, default=2, help='target session')
# GPU
parser.add_argument('--gpu', type=int, default=-1, help='using gpu' )
args = parser.parse_args()

args.save = '{}/SAVED_LOGS/SEARCH-S{}-S{}-DAConvBLS-{}-{}'.format(Project_Path, args.ss, args.ts, args.dataset, time.strftime("%Y%m%d-%H%M%S"))
time.strftime("%Y%m%d-%H%M%S")
utils.create_exp_dir(path=args.save)
utils.create_exp_dir(path='{}/saved_models'.format(args.save))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('args = %s', args)

"""random seed"""
random.seed(args.seed)
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

if torch.cuda.is_available() and args.gpu != -1:
    device = torch.device('cuda:{}'.format(str(args.gpu)))
    torch.cuda.manual_seed(args.seed)
    logging.info('using gpu : {}'.format(args.gpu))
else:
    device = torch.device('cpu')
    logging.info('using cpu')

input_shape = load_dataset_config[args.dataset]['input_shape']
args.in_channels = input_shape[-1]
args.num_class = load_dataset_config[args.dataset]['num_class']

session_1_loader_list, session_2_loader_list, session_3_loader_list = load_dataset[args.dataset](
    batch_size=None, shuffle=False, normalize='zscore'
)

for idx in range(1, len(session_1_loader_list) + 1):
    utils.create_exp_dir(path='{}/saved_models/sub{}'.format(args.save, idx))

data_list = [session_1_loader_list, session_2_loader_list, session_3_loader_list]
train_loader_list = data_list[args.ss-1]
test_loader_list = data_list[args.ts-1]

best_acc_per_sub = []  
best_acc_per_sub_base = []

total = len(args.N1_LIST) * len(args.N2_LIST) * len(args.N3_LIST) * len(args.AF_LIST)

time_start = time.time()

for sub, train_loader in enumerate(train_loader_list):

    test_loader = test_loader_list[sub]

    best_acc_among_archs = 0   
    best_acc_among_archs_base = 0
    logging.info('--------------------------------SUBJECT={}--------------------------------'.format(sub+1))
    for N1 in args.N1_LIST:
        for N2 in args.N2_LIST:
            for N3 in args.N3_LIST:
                for AF in args.AF_LIST:
                    best_results, best_results_base = train_da(
                        train_loader=train_loader,
                        test_loader=test_loader,
                        in_channels=args.in_channels,
                        N1=N1,
                        N2=N2,
                        N3=N3,
                        KS=3,
                        activation=AF,
                        num_class=args.num_class,
                        device=device,
                        save=args.save,
                        seed=args.seed,
                        verbose=True,
                        last_best_acc = best_acc_among_archs,   
                        last_best_acc_base = best_acc_among_archs_base,  
                        sub=sub                                 
                        )
                    if best_results['acc'] > best_acc_among_archs:
                        best_acc_among_archs = best_results['acc']
                        logging.info('| DAConvBLS | SUBJECT:{}; N1={}; N2={}; N3={}; AF={}; BEST_C={}; BEST_LAM={}; BEST_ACC={}'
                                    .format(sub+1, N1, N2, N3, AF, best_results['c'], best_results['lambda'], best_acc_among_archs))

                    if best_results_base['acc'] > best_acc_among_archs_base:
                        best_acc_among_archs_base = best_results_base['acc']
                        logging.info('| ConvBLS | SUBJECT:{}; N1={}; N2={}; N3={}; AF={}; BEST_C: {}; BEST_ACC: {}'
                                    .format(sub+1, N1, N2, N3, AF, best_results_base['c'], best_acc_among_archs_base))

                    time_end = time.time()

                    print('time:{}'.format(time_end-time_start))
    
    fe_gconvbls = torch.load('{}/saved_models/sub{}/fe_daconvbls.pth'.format(args.save, sub+1))
    cls_gconvbls = torch.load('{}/saved_models/sub{}/cls_daconvbls.pth'.format(args.save, sub+1)) 
    gconvbls = torch.nn.Sequential(fe_gconvbls, cls_gconvbls)   

    test_acc = utils.Accuracy() 
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device).double(), F.one_hot(test_label.to(device), args.num_class).double()
        out = gconvbls(test_data)
        test_acc.update(out, test_label)
    logging.info('-------------------------------------------')
    logging.info('DAConvBLS:{}%'.format(test_acc.acc * 100))
    logging.info('-------------------------------------------')

    fe_convbls = torch.load('{}/saved_models/sub{}/fe_convbls.pth'.format(args.save, sub+1))
    cls_convbls = torch.load('{}/saved_models/sub{}/cls_convbls.pth'.format(args.save, sub+1)) 
    convbls = torch.nn.Sequential(fe_convbls, cls_convbls)

    test_acc = utils.Accuracy() 
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device).double(), F.one_hot(test_label.to(device), args.num_class).double()
        out = convbls(test_data)
        test_acc.update(out, test_label)
    logging.info('-------------------------------------------')
    logging.info('ConvBLS:{}%'.format(test_acc.acc * 100))
    logging.info('-------------------------------------------')
    
    best_acc_per_sub.append(best_acc_among_archs)
    logging.info('------DAConvBLS------')
    logging.info('SUBJECT1-SUBJECT16 best accuracy: {}'.format(str(best_acc_per_sub)))
    logging.info('SESSION{}->SESSION{} average accuracy: {}'.format(args.ss, args.ts, np.mean(best_acc_per_sub)))

    best_acc_per_sub_base.append(best_acc_among_archs_base)
    logging.info('---------ConvBLS---------')
    logging.info('SUBJECT1-SUBJECT16 best accuracy: {}'.format(str(best_acc_per_sub_base)))
    logging.info('SESSION{}->SESSION{} average accuracy: {}'.format(args.ss, args.ts, np.mean(best_acc_per_sub_base)))
