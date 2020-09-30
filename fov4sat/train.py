"""Train and evaluate model"""
from __future__ import print_function, division
import os
import shutil
import argparse
import torch
import logging
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms
import numpy as np
from .gaussian_foveanet_dataset import HeatmapFoveanetDataset
from .model import Foveanet
from .eval_measures import EvalMeasures
from .early_stopping import EarlyStopping
import foveanet.config as cconf
from datetime import datetime
import cv2 as cv
import foveanet.center_loss as center_loss
from .torchsummary import summary


def init_logging(train_aoi_dirs, eval_aoi_dirs, num_channels):
    filename_ = os.path.join(cconf.cfg.save_folder, 'app.log')
    logging.basicConfig(filename=filename_, level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.info('train_aoi_dirs: %s', train_aoi_dirs)
    logging.info('eval_aoi_dirs: %s', eval_aoi_dirs)
    logging.info('num_channels: %d', num_channels)
    logging.info('num_channels: %d', num_channels)
    config_info = cconf.cfg.print2()
    logging.info(
        '-------------------------Start Config info----------------------------')
    for ii in range(len(config_info)):
        logging.info(config_info[ii])
    logging.info(
        '-------------------------End Config info----------------------------')


def init_save_folder():
    """Prepare folder to save outputs"""
    dirpath = cconf.cfg.save_folder
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)


def save_weights(_net, _epoch, _iteration):
    """Save current weights of network"""
    file_name = cconf.cfg.name + '_' + \
        str(_epoch) + '_' + str(_iteration) + '.pt'
    save_path = os.path.join(cconf.cfg.save_folder, file_name)
    _net.save_weights(save_path)


def load_weights(_net, _num_channels):
    _stat = 'FAILED'
    status = False
    if _num_channels == 5:
        status = _net.load_checkpoint(cconf.cfg.path_checkpoint_c5)
    elif _num_channels == 1:
        status = _net.load_checkpoint(cconf.cfg.path_checkpoint_c1)
    else:
        logging.info("WARNING: No weights available for selected channels")
        print("WARNING: No weights available for selected channels")
    if status == True:
        _stat = 'OK'
    logging.info("load_checkpoint: %s", _stat)
    print("load_checkpoint: ", _stat)


def freeze_layers(_net):
    _net.conv1.weight.requires_grad = False
    _net.conv1.bias.requires_grad = False
    _net.conv2.weight.requires_grad = False
    _net.conv2.bias.requires_grad = False
    _net.conv3.weight.requires_grad = False
    _net.conv3.bias.requires_grad = False
    _net.conv4.weight.requires_grad = False
    _net.conv4.bias.requires_grad = False
    _net.conv5.weight.requires_grad = False
    _net.conv5.bias.requires_grad = False


def get_args():
    """Reads arguments from stdin"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument("-train", "--training-data", nargs='+',
                        help="Path to folder containing training ROOBIS", action="store", required=True)
    parser.add_argument("-eval", "--evaluation-data", nargs='+',
                        help="Path to folder containing evaluation ROOBIS", action="store", required=True)
    return parser.parse_args()


def get_device():
    """Provides device used for training"""
    if not torch.cuda.is_available():
        print('CUDA not available')
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataloader(aoi_dirs, num_channels, typeDataset, mean, std, 
                      data_set_class):
    """Returns dataloader containing data in given paths and number of frames in dataset"""
    if typeDataset == 'training':
        _seq_size = [cconf.cfg.min_frame_training,
                     cconf.cfg.max_frame_training]
        nth_frame = cconf.cfg.nth_frame_training
    else:
        _seq_size = [cconf.cfg.min_frame_eval, cconf.cfg.max_frame_eval]
        nth_frame = cconf.cfg.nth_frame_eval

    channel_means = []
    channel_stds = []
    for __ in range(num_channels):
        channel_means.append(mean)
        channel_stds.append(std)
    length_data = 0
    datasets = []
    for aoi_dir in aoi_dirs:
        for dir_roobi, __, __ in os.walk(aoi_dir, followlinks=True):
            if "ROOBI" in dir_roobi and "ANNOTATED" not in dir_roobi and "FULL" not in dir_roobi:
                new_dataset = data_set_class(
                    csv_file=os.path.join(dir_roobi, 'gt.csv'),
                    root_dir=dir_roobi,
                    size=cconf.cfg.roobiSize,
                    channels=num_channels,
                    seq_size=_seq_size,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(channel_means, channel_stds)]),
                    nth_frame=nth_frame
                )
                if new_dataset.frame_numbers.size > 0:
                    datasets.append(new_dataset)
                length_data = length_data + new_dataset.frame_numbers.size
    dataset = ConcatDataset(datasets)

    dataloader = DataLoader(dataset, batch_size=cconf.cfg.batch_size,
                            shuffle=(typeDataset == 'training'), num_workers=cconf.cfg.num_workers,
                            collate_fn=data_set_class.collate_fn)
    return dataloader, length_data


def train(net, device, training_dataloader, evaluation_dataloader):
    """Train network"""
    criterion = nn.MSELoss(reduction='sum').cuda()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=cconf.cfg.lr)
    early_stopping = EarlyStopping(patience=cconf.cfg.earlyStopping_patience,
                                   save_folder=cconf.cfg.save_folder, verbose=True)
    eval_measures = EvalMeasures()
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    f1_scores = []
    recalls = []
    precisions = []
    n_epochs = cconf.cfg.num_epochs
    print_min_batch = 100
    iteration = 0
    try:
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            net.train()
            running_loss = 0.0
            for i, data in enumerate(training_dataloader, 0):
                iteration = i
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = net(inputs)
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                # loss = loss.mean()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                # print statistics
                running_loss += loss.item()
                if iteration % print_min_batch == print_min_batch-1:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, iteration + 1, running_loss / print_min_batch))
                
            eval_measures.reset()
            with torch.no_grad():
                net.eval()
                for idx_batch, data_eval in enumerate(evaluation_dataloader):
                    inputs, labels, positions = data_eval[0].to(
                        device), data_eval[1].to(device), data_eval[2]
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    # record validation loss
                    valid_losses.append(loss.item())
                    eval_measures.update(outputs.cpu(), positions)
                    
            prec, rec, f1 = eval_measures.get_measures()
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)
            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f} ' +
                         f'precission: {prec:.5f} ' +
                         f'recall: {rec:.5f} ' +
                         f'f1_score: {f1:.5f}')
            print(print_msg)
            logging.info(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # calculate_accuracy(test_dataloader, accuracy, device, net)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, f1_scores, net)

            if early_stopping.early_stop:
                print("Early stopping")
                logging.info("Early stopping")
                break

    except KeyboardInterrupt:
        print('Stopping early. Saving network...')
    save_weights(net, epoch, iteration)


    logging.info('Finished Training')
    print('Finished Training')
    return avg_train_losses, avg_valid_losses, f1_scores, precisions, recalls, early_stopping.early_stop_F1_score


def main():
    """Main entry point"""
    args = get_args()
    train_aoi_dirs = args.training_data
    eval_aoi_dirs = args.evaluation_data

    if args.config is not None:
        cconf.set_cfg(args.config)
    print(args.config)
    num_channels = cconf.cfg.num_channels

    init_save_folder()
    init_logging(train_aoi_dirs, eval_aoi_dirs, num_channels)

    assert num_channels % 2 != 0

    device = get_device()

    logging.info('Prepare training dataset - Begin')
    print('Prepare training dataset - Begin')
    training_dataloader, num_training_data = create_dataloader(train_aoi_dirs, num_channels, 'training',
                                                               cconf.cfg.inputData_mean, cconf.cfg.inputData_std,
                                                               HeatmapFoveanetDataset)
    print('Prepare training dataset - Finished')
    print('Amount frames for training: ', num_training_data)
    logging.info('Prepare training dataset - Finished')
    logging.info('Amount frames for training: %d', num_training_data)

    logging.info('Prepare evaluation dataset - Begin')
    print('Prepare evaluation dataset - Begin')
    evaluation_dataloader, num_eval_data = create_dataloader(eval_aoi_dirs, num_channels, 'eval',
                                                             cconf.cfg.inputData_mean, cconf.cfg.inputData_std,
                                                             HeatmapFoveanetDataset)
    print('Prepare evaluation dataset - Finished')
    print('Amount frames for evaluation: ', num_eval_data)
    logging.info('Prepare evaluation dataset - Finished')
    logging.info('Amount frames for evaluation: %d', num_eval_data)

    logging.info("Start training")
    print("Start training")

    net = Foveanet(channels=num_channels)

    if cconf.cfg.fineTuning is True:
        freeze_layers(net)

    if (cconf.cfg.loadTrainedModel):
        load_weights(net, cconf.cfg._num_channels)

    net.cuda()
    summary(
        net, (num_channels, cconf.cfg.roobiSize[0], cconf.cfg.roobiSize[1]))
    print(device.type)

    avg_train_losses, avg_valid_losses, f1_scores, precisions, recalls, 
    early_stop_F1_score = \
        train(net, device,
              training_dataloader,
              evaluation_dataloader)
    create_plots.create_loss_plot(avg_train_losses, avg_valid_losses,
                                  os.path.join(cconf.cfg.save_folder,
                                               'loss_plot.png'))
    create_plots.create_f1_plot(f1_scores,
                                os.path.join(cconf.cfg.save_folder,
                                             'f1_plot.png'))
    create_plots.create_precision_recall_plot(precisions,
                                              recalls,
                                              os.path.join(
                                                  cconf.cfg.save_folder,
                                                  'precission_recall_plot.png'))
    # save data to disk
    out_arr = np.asarray(avg_train_losses)
    filename = os.path.join(cconf.cfg.save_folder, 'avg_train_losses')
    np.save(filename, out_arr)
    out_arr = np.asarray(avg_valid_losses)
    filename = os.path.join(cconf.cfg.save_folder, 'avg_valid_losses')
    np.save(filename, out_arr)
    out_arr = np.asarray(f1_scores)
    filename = os.path.join(cconf.cfg.save_folder, 'f1')
    np.save(filename, out_arr)
    out_arr = np.asarray(precisions)
    filename = os.path.join(cconf.cfg.save_folder, 'precissions')
    np.save(filename, out_arr)
    out_arr = np.asarray(recalls)
    filename = os.path.join(cconf.cfg.save_folder, 'recalls')
    np.save(filename, out_arr)

    if not early_stop_F1_score:
        timestr = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]

        dst = cconf.cfg.save_folder + '_' + timestr
        try:
            shutil.copytree(cconf.cfg.save_folder, dst)
        # Directories are the same
        except shutil.Error as e:
            print('Directory not copied. Error: %s' % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print('Directory not copied. Error: %s' % e)


if __name__ == "__main__":
    main()
