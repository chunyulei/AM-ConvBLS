Project_Path = 'XXX'
import sys
sys.path.append(Project_Path)

import random
import torch
import utils
import logging

import numpy as np
import torch.nn.functional as F
import config.searchspace as ss

from progress.bar import IncrementalBar
from models import GRDALinearClassifier


def train_grda(
        train_loader, 
        test_loader,  
        num_class, 
        device, 
        save, 
        seed,
        model_path  
        ):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    feature_extractor = torch.load('{}/fe_daconvbls.pth'.format(model_path))
    logging.info(feature_extractor)

    feature_extractor.eval()

    for step, (train_data, train_label) in enumerate(train_loader):
        train_data, train_label = train_data.to(device).double(), F.one_hot(train_label.to(device), num_class).double()
        train_fea = feature_extractor(train_data)
    for step, (test_data, test_label) in enumerate(test_loader):
        test_data, test_label = test_data.to(device).double(), F.one_hot(test_label.to(device), num_class).double()
        test_fea = feature_extractor(test_data)

    cls_daconvbls = torch.load('{}/cls_daconvbls.pth'.format(model_path))

    test_acc = utils.Accuracy() 
    out = cls_daconvbls(test_fea)
    test_acc.update(out, test_label)
    logging.info('A-ConvBLS: {}%'.format(test_acc.acc*100))

    bar = IncrementalBar('search for hyperparameters', max=len(ss.C_LIST)*len(ss.LAMBDA1_LIST)*len(ss.LAMBDA2_LIST))
    
    best_result = {}
    best_result['c'] = None
    best_result['lambda1'] = None
    best_result['lambda2'] = None
    best_result['acc'] = 0
    best_result['f1'] = 0
    best_result['mcc'] = 0
    for c in ss.C_LIST:
        for lam1 in ss.LAMBDA1_LIST:
            for lam2 in ss.LAMBDA2_LIST:
                train_acc = utils.Accuracy()
                grdalinear_classifier = GRDALinearClassifier(
                    feature_dim=100, 
                    num_class=num_class, 
                    c=c, 
                    lam1=lam1,
                    lam2=lam2
                    )

                grdalinear_classifier.train()
                out = grdalinear_classifier(train_fea, test_fea, train_label)
                train_acc.update(out, train_label)

                test_metric = utils.metric() 
                grdalinear_classifier.eval()
                out = grdalinear_classifier(test_fea)

                pred = torch.argmax(out, dim=1)
                test_metric.update(pred, torch.argmax(test_label, dim=1))

                test_acc = test_metric.get_accuracy() * 100
                test_f1 = test_metric.get_F1_socre(average='macro') * 100
                test_mcc = test_metric.get_MCC() * 100

                if test_acc > best_result['acc']:
                    best_result['c'] = c
                    best_result['lambda1'] = lam1
                    best_result['lambda2'] = lam2
                    best_result['acc'] = test_acc
                    best_result['f1'] = test_f1
                    best_result['mcc'] = test_mcc
                    torch.save(grdalinear_classifier, '{}/cls_grdaconvbls.pth'.format(save))

                train_acc.reset(), test_metric.reset()
                bar.next() 
    bar.finish()

    cls_grdaconvbls = torch.load('{}/cls_grdaconvbls.pth'.format(save))
    convbls_dag = torch.nn.Sequential(feature_extractor, cls_grdaconvbls)

    test_acc = utils.Accuracy()
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device).double(), F.one_hot(test_label.to(device), args.num_class).double()
        out = convbls_dag(test_data)
        test_acc.update(out, test_label)
    logging.info('AM-ConvBLS ACC: {}%'.format(test_acc.acc*100))

    return best_result


if __name__ == '__main__':

    import os
    import sys
    import time
    import logging
    import argparse

    from dataset import load_dataset, load_dataset_config


    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', type=str, default='SEED-V', help='dataset name')
    parser.add_argument('--sess', type=int, default=1, help='session id(1, 2, 3)')
    parser.add_argument('--ts', type=int, default=1, help='target domain subject(1-15 for seed-v,1-14 for seed-iv)')
    parser.add_argument('--gpu', type=int, default=-1, help='using gpu' )
    args = parser.parse_args()

    args.save = '{}/SAVED_LOGS/TRAIN-OTO-SESS-{}-TS-{}-ConvBLS-DAG-{}-{}'.format(
        Project_Path, args.sess, args.ts, args.dataset, time.strftime("%Y%m%d-%H%M%S"))
    time.strftime("%Y%m%d-%H%M%S")
    utils.create_exp_dir(path=args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('args = %s', args)

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


    session_1_loader_list, session_2_loader_list, session_3_loader_list = load_dataset[args.dataset](
        batch_size=None, shuffle=False, normalize='zscore'
    )

    input_shape = load_dataset_config[args.dataset]['input_shape']
    args.in_channels = input_shape[-1]
    args.num_class = load_dataset_config[args.dataset]['num_class']

    data_loader_list = [session_1_loader_list, session_2_loader_list, session_3_loader_list]
    test_loader = data_loader_list[args.sess-1][args.ts]
    train_loader = data_loader_list[args.sess-1][0]

    if args.dataset == 'SEED-V' and args.sess == 1:
        Searched_Path = '{}/SAVED_LOGS/XXX'.format(Project_Path)
    elif args.dataset == 'SEED-V' and args.sess == 2:
        Searched_Path = '{}/SAVED_LOGS/XXX'.format(Project_Path)
    elif args.dataset == 'SEED-V' and args.sess == 3:
        Searched_Path = '{}/SAVED_LOGS/XXX'.format(Project_Path)
    else:
        raise ValueError('ERROR')

    best_result = train_grda(
            train_loader=train_loader,
            test_loader=test_loader,
            num_class=args.num_class,
            device=device,
            save=args.save,
            seed=args.seed,
            model_path='{}/saved_models/sub{}'.format(Searched_Path, args.ts)
    )
    logging.info('-----------------------------------------------------AM-ConvBLS--------------------------------------------------------------')
    logging.info('| TARGET SUBJECT={} | BEST_C={} | BEST_LAM1={} | BEST_LAM2={} | TEST_ACC={:>7.3f}% | TEST_F1={:>7.3f}% | TEST_MCC={:>7.3f}% |'.format(args.ts+1, best_result['c'], best_result['lambda1'], best_result['lambda2'], best_result['acc'], best_result['f1'], best_result['mcc']))
    logging.info('------------------------------------------------------------------------------------------------------------------------------')
