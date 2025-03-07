import os
import random
import torch
import utils
import logging

import numpy as np
import config.searchspace as ss
import torch.nn.functional as F

from models import LinearClassifier, DALinearClassifier, FE_SPP_SFL_Parallel


def train_da(
        train_loader, 
        test_loader, 
        in_channels, 
        N1, 
        N2, 
        N3, 
        KS, 
        activation, 
        num_class, 
        device, 
        save, 
        seed,
        verbose,
        last_best_acc,   
        last_best_acc_base,  
        sub 
        ):

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    feature_extractor = FE_SPP_SFL_Parallel(
        in_channels=in_channels,
        N1=N1,
        N2=N2,
        N3=N3,
        KS=KS,
        activation=activation,
        verbose=verbose
    )

    feature_extractor.train()
    for step, (train_data, train_label) in enumerate(train_loader):
        train_data, train_label = train_data.to(device).double(), F.one_hot(train_label.to(device), num_class).double()
        train_fea = feature_extractor(train_data)

    feature_extractor.eval()
    for step, (test_data, test_label) in enumerate(test_loader):
        test_data, test_label = test_data.to(device).double(), F.one_hot(test_label.to(device), num_class).double()
        test_fea = feature_extractor(test_data)


    input_dim = (N1 * N2 + N3) * (9 + 4 + 1)
    
    best_results = {}
    best_results['c'] = None
    best_results['lambda'] = None
    best_results['acc'] = 0
    for c in ss.C_LIST:
        for lam in ss.LAMBDA2_LIST:
            train_acc, test_acc = utils.Accuracy(), utils.Accuracy()
            dalinear_classifier = DALinearClassifier(
                feature_dim=input_dim, 
                num_class=num_class, 
                c=c, 
                lam=lam
                )

            dalinear_classifier.train()
            out = dalinear_classifier(train_fea,  test_fea, train_label)

            dalinear_classifier.eval()
            out = dalinear_classifier(test_fea)
            test_acc.update(out, test_label)

            if test_acc.acc > best_results['acc']:
                best_results['c'] = c
                best_results['lambda'] = lam
                best_results['acc'] = test_acc.acc
                torch.save(dalinear_classifier, '{}/saved_models/sub{}/temp_cls_daconvbls.pth'.format(save, sub+1))

            train_acc.reset(), test_acc.reset()
    
    if best_results['acc'] > last_best_acc:
        torch.save(feature_extractor, '{}/saved_models/sub{}/fe_daconvbls.pth'.format(save, sub+1))
        os.rename('{}/saved_models/sub{}/temp_cls_daconvbls.pth'.format(save, sub+1), '{}/saved_models/sub{}/cls_daconvbls.pth'.format(save, sub+1))
    else:
        os.remove('{}/saved_models/sub{}/temp_cls_daconvbls.pth'.format(save, sub+1))
    
    """
    ######################################## baseline ############################################
    """

    input_dim = (N1 * N2 + N3) * (9 + 4 + 1)

    best_results_base = {}
    best_results_base['c'] = None
    best_results_base['acc'] = 0
    for c in ss.C_LIST:
        linear_classifier = LinearClassifier(feature_dim=input_dim, num_class=num_class, c=c)
        
        train_acc, test_acc = utils.Accuracy(), utils.Accuracy()

        linear_classifier.train()
        out = linear_classifier(train_fea, train_label)

        linear_classifier.eval()
        out = linear_classifier(test_fea)
        test_acc.update(out, test_label)

        if test_acc.acc > best_results_base['acc']:
            best_results_base['c'] = c
            best_results_base['acc'] = test_acc.acc
            torch.save(linear_classifier, '{}/saved_models/sub{}/temp_cls_convbls.pth'.format(save, sub+1))
        train_acc.reset(), test_acc.reset()
    
    if best_results_base['acc'] > last_best_acc_base:
        torch.save(feature_extractor, '{}/saved_models/sub{}/fe_convbls.pth'.format(save, sub+1))
        os.rename('{}/saved_models/sub{}/temp_cls_convbls.pth'.format(save, sub+1), '{}/saved_models/sub{}/cls_convbls.pth'.format(save, sub+1))
    else:
        os.remove('{}/saved_models/sub{}/temp_cls_convbls.pth'.format(save, sub+1))

    return best_results, best_results_base
