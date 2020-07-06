import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, 
                       plot_confusion_matrix, print_accuracy, 
                       write_leaderboard_submission, write_evaluation_submission)
from models_pytorch import move_data_to_gpu, DecisionLevelMaxPooling, FGSMAttack, ResNet, Vggish
import config
from torch.autograd import Variable

Model = Vggish # ResNet # DecisionLevelMaxPooling
batch_size = 16
#epsilon_value = 0.1
#alpha_value = 0.05

def evaluate(model, model_adv, generator, data_type, devices, max_iteration, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
      max_iteration: int, maximum iteration for validation
      cuda: bool.
      
    Returns:
      accuracy: float
    """
    
    # Generate function
    generate_func = generator.generate_validate(data_type=data_type, 
                                                devices=devices, 
                                                shuffle=True, 
                                                max_iteration=max_iteration)
            
    # Forward
    dict = forward(model=model, 
		   model_adv=model_adv,
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=True)

    outputs = dict['output']    # (audios_num, classes_num)
    outputs_adv = dict['output_adv']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)
    
    predictions = np.argmax(outputs, axis=-1)   # (audios_num,)
    predictions_adv = np.argmax(outputs_adv, axis=-1)   # (audios_num,)

    # Evaluate
    classes_num = outputs.shape[-1]

    loss = F.nll_loss(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy()

    loss = float(loss)

    loss_adv = F.nll_loss(Variable(torch.Tensor(outputs_adv)), Variable(torch.LongTensor(targets))).data.numpy()

    loss_adv = float(loss_adv)

    confusion_matrix = calculate_confusion_matrix(
        targets, predictions, classes_num)
    
    accuracy = calculate_accuracy(targets, predictions, classes_num, 
                                  average='macro')

    accuracy_adv = calculate_accuracy(targets, predictions_adv, classes_num, 
                                  average='macro')

    return accuracy, loss, accuracy_adv, loss_adv

# forward: model_pytorch---        Return_heatmap = False
# forward_heatmap: model_pytorch---        Return_heatmap = True
def forward(model, model_adv, generate_func, cuda, return_target):
    """Forward data to a model.
    
    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs = []
    audio_names = []
    outputs_adv = []
    
    if return_target:
        targets = []
    
    # Evaluate on mini-batch
    for data in generate_func:
            
        if return_target:
            (batch_x, batch_y, batch_audio_names) = data
            
        else:
            (batch_x, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()
        batch_output, _ = model(batch_x)
	
	# advesarial predict
        batch_y_pred = np.argmax(batch_output.data.cpu().numpy(), axis=-1)
        batch_y_pred = move_data_to_gpu(batch_y_pred, cuda)

        model_cp = copy.deepcopy(model)
        for p in model_cp.parameters():
            p.requires_grad = False
        model_cp.eval()

        model_adv.model = model_cp
        del model_cp

        batch_x_adv = model_adv.perturb(batch_x.data.cpu().numpy(), batch_y_pred.data.cpu().numpy(), cuda=cuda)
        batch_x_adv = move_data_to_gpu(batch_x_adv, cuda)
        batch_output_adv, _ = model(batch_x_adv)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)
        outputs_adv.append(batch_output_adv.data.cpu().numpy())        

        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    outputs_adv = np.concatenate(outputs_adv, axis=0)
    dict['output_adv'] = outputs_adv
    
    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets
        
    return dict



def train(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    feature_type = args.feature_type
    filename = args.filename
    validation = args.validation
    holdout_fold = args.holdout_fold
    epsilon_value = args.epsilon_value
    alpha_value = args.alpha_value
    mini_data = args.mini_data
    cuda = args.cuda

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    classes_num = len(labels)

    # Paths
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', feature_type, 'mini_development.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', feature_type, 'development.h5')

    if validation:
        
        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                     'fold{}_train.txt'.format(holdout_fold))
                                    
        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold{}_devel.txt'.format(holdout_fold))
                              
        models_dir = os.path.join(workspace, 'models', subdir, filename,
                                  'holdout_fold={}'.format(holdout_fold), 'epsilon={}-alpha={}'.format(epsilon_value, alpha_value))
                                        
    else:
        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                     'fold{}_traindevel.txt'.format(holdout_fold))

        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                        'fold{}_test.txt'.format(holdout_fold))
        
        models_dir = os.path.join(workspace, 'models', subdir, filename,
                                  'full_train', 'epsilon={}-alpha={}'.format(epsilon_value, alpha_value))

    create_folder(models_dir)

    # Model
    model = Model(classes_num)
    adversary = FGSMAttack(epsilon=epsilon_value, alpha=alpha_value)

    if cuda:
        model.cuda()

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)

    # Optimizer
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)

    train_bgn_time = time.time()

    # Train on mini batches
    for (iteration, (batch_x, batch_y)) in enumerate(generator.generate_train()):

        # Evaluate
        if iteration % 100 == 0:

            train_fin_time = time.time()

            (tr_acc, tr_loss, tr_acc_adv, tr_loss_adv) = evaluate(model=model,
					 model_adv=adversary,
                                         generator=generator,
                                         data_type='train',
                                         devices=devices,
                                         max_iteration=None,
                                         cuda=cuda)

            logging.info('tr_acc: {:.3f}, tr_loss: {:.3f}, tr_acc_adv: {:.3f}, tr_loss_adv: {:.3f}'.format(
                tr_acc, tr_loss, tr_acc_adv, tr_loss_adv))

            (va_acc, va_loss, va_acc_adv, va_loss_adv) = evaluate(model=model,
					model_adv=adversary,
                                        generator=generator,
                                        data_type='validate',
                                        devices=devices,
                                        max_iteration=None,
                                        cuda=cuda)

            logging.info('va_acc: {:.3f}, va_loss: {:.3f}, va_acc_adv: {:.3f}, va_loss_adv: {:.3f}'.format(
                    va_acc, va_loss, va_acc_adv, va_loss_adv))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:

            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))
            
        # Reduce learning rate
        if iteration % 100 == 0 > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        # Train
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        model.train()
        batch_output, batch_outputvector = model(batch_x)

        loss = F.nll_loss(batch_output, batch_y)

        if iteration >= 1000:
            batch_y_pred = np.argmax(batch_output.data.cpu().numpy(), axis=-1)
            batch_y_pred = move_data_to_gpu(batch_y_pred, cuda)

            model_cp = copy.deepcopy(model)
            for p in model_cp.parameters():
                p.requires_grad = False
            model_cp.eval()

            adversary.model = model_cp
            del model_cp

            batch_x_adv = adversary.perturb(batch_x.data.cpu().numpy(), batch_y_pred.data.cpu().numpy(), cuda=cuda)
            batch_x_adv = move_data_to_gpu(batch_x_adv, cuda)
            batch_output_adv, batch_outputvector_adv = model(batch_x_adv)
            loss_adv = F.nll_loss(batch_output_adv, batch_y)
	    
	    loss_pair = F.mse_loss(batch_outputvector_adv, batch_outputvector)
		
	    #print('loss:'+str(loss)+'\t'+'loss_adv'+str(loss_adv)+'\t'+'loss_pair'+str(loss_pair))
            loss = 0.4 * loss + 0.4 * loss_adv + 0.2 * loss_pair

            #if iteration % 10 == 0:
            #    logging.info('batch loss: {}, batch loss_adv: {}'.format(loss, loss_adv))


        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 10000:
            break


def inference_validation_data(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    feature_type = args.feature_type
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    epsilon_value = args.epsilon_value
    alpha_value = args.alpha_value
    filename = args.filename
    cuda = args.cuda
    validation = args.validation

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    classes_num = len(labels)

    # Paths
    #hdf5_path = os.path.join(workspace, 'features', subdir, '{}.h5'.format(feature_type))
    hdf5_path = os.path.join(workspace, 'features', feature_type, 'development.h5')

    if validation:

        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup', 'fold{}_train.txt'.format(holdout_fold))
                                 
        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup', 'fold{}_devel.txt'.format(holdout_fold))

        model_path = os.path.join(workspace, 'models', subdir, filename,
                                'holdout_fold={}'.format(holdout_fold), 'epsilon={}-alpha={}'.format(epsilon_value, alpha_value),
                                'md_{}_iters.tar'.format(iteration))
    else:

        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup', 'fold{}_traindevel.txt'.format(holdout_fold))

        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup', 'fold{}_test.txt'.format(holdout_fold))

        model_path = os.path.join(workspace, 'models', subdir, filename,
                                  'full_train', 'epsilon={}-alpha={}'.format(epsilon_value, alpha_value),
                                  'md_{}_iters.tar'.format(iteration))

    # Load model
    model = Model(classes_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    adversary = FGSMAttack(epsilon=epsilon_value, alpha=alpha_value)

    if cuda:
        model.cuda()

    # Predict & evaluate
    for device in devices:

        print('Device: {}'.format(device))

        # Data generator
        generator = DataGenerator(hdf5_path=hdf5_path,
                                  batch_size=batch_size,
                                  dev_train_csv=dev_train_csv,
                                  dev_validate_csv=dev_validate_csv)

        generate_func = generator.generate_validate(data_type='validate', 
                                                     devices=device, 
                                                     shuffle=False)

        # Inference
        dict = forward(model=model,
		       model_adv=adversary,
                       generate_func=generate_func, 
                       cuda=cuda, 
                       return_target=True)

        outputs = dict['output']    # (audios_num, classes_num)
        targets = dict['target']    # (audios_num, classes_num)
  	outputs_adv = dict['output_adv']    # (audios_num, classes_num)

        predictions = np.argmax(outputs, axis=-1)
        predictions_adv = np.argmax(outputs_adv, axis=-1)

        classes_num = outputs.shape[-1]      

        # Evaluate
        confusion_matrix = calculate_confusion_matrix(targets, predictions, classes_num)
            
        class_wise_accuracy = calculate_accuracy(targets, predictions, classes_num)


        confusion_matrix_adv = calculate_confusion_matrix(targets, predictions_adv, classes_num)
            
        class_wise_accuracy_adv = calculate_accuracy(targets, predictions_adv, classes_num)



        # Print
        print_accuracy(class_wise_accuracy, labels)
        print('confusion_matrix: \n', confusion_matrix)
        logging.info('confusion_matrix: \n', confusion_matrix)

        print_accuracy(class_wise_accuracy_adv, labels)
        print('confusion_matrix: \n', confusion_matrix_adv)
        logging.info('confusion_matrix: \n', confusion_matrix_adv)

        # Plot confusion matrix
#        plot_confusion_matrix(
#            confusion_matrix,
#            title='Device {}'.format(device.upper()), 
#            labels=labels,
#            values=class_wise_accuracy,
#            path=os.path.join(workspace, 'logs', 'main_pytorch', 'fig-confmat-device-'+device+'.pdf'))

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--subdir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--feature_type', type=str, default='logmel')
    parser_train.add_argument('--validation', action='store_true', default=False)
    parser_train.add_argument('--holdout_fold', type=int)
    parser_train.add_argument('--epsilon_value', type=float)
    parser_train.add_argument('--alpha_value', type=float)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)

    
    parser_inference_validation_data = subparsers.add_parser('inference_validation_data')
    parser_inference_validation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation_data.add_argument('--subdir', type=str, required=True)
    parser_inference_validation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_data.add_argument('--feature_type', type=str, default='logmel')
    parser_inference_validation_data.add_argument('--validation', action='store_true', default=False)
    parser_inference_validation_data.add_argument('--holdout_fold', type=int, required=True)
    parser_inference_validation_data.add_argument('--epsilon_value', type=float)
    parser_inference_validation_data.add_argument('--alpha_value', type=float)
    parser_inference_validation_data.add_argument('--iteration', type=int, required=True)
    parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)

    else:
        raise Exception('Error argument!')

