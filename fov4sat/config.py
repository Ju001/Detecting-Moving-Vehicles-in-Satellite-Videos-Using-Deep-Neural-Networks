"""This module serves as configuration file
for training and applying FoveaNet4Sat."""


class Config(object):
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)


foveanet_base_config = Config({
    'name': 'foveanet',  # Name of this configuration

    'roobi_size': [128, 128],  # Size of individual ROOBIs
    'batch_size': 64,

    'num_workers': 8,  # Threads used by PyTorch for loading input data
    'learning_rate': 1e-5,

    'num_epochs': 30,
    'early_stopping_patience': 10,

    'min_frame_training': 100,
    'max_frame_training': 1124,

    'min_frame_eval': 100,
    'max_frame_eval': 1124,

    'num_channels': 5,  # How many temproal consistent frames are used as input

    'save_folder': 'results',  # Folder where logs and checkpoint are saved

    'inputData_mean': 0.26710122557785,
    'inputData_std': 0.28491544784918,

    'gaussian_std': 2.0,  # Determines size of Gaussians in GT-heatmaps

    'stationary_car_threshold': 15,  # How far a car has to move over the
                                     # course of 5 frames in order to be
                                     # recognized as moving vehicle
    'evaluation_tolerance': 20,  # Tolerance of detection position regarding
                                 # to the ground truth

    'evaluation_min_blob_size': 15,  # Min size of predicted blob

    'nth_frame_training': 1,  # Used to skip frames for training
    'nth_frame_eval': 1,  # Used to skip frames for evaluation

    'include_stationary': False,  # Include or dismiss stationary vehicles

    'kernels': [15, 13, 11, 9, 7, 5, 3, 1],  # Kernel sizes for each layer
    'paddings': [7, 6, 5, 4, 3, 2, 1, 0],  # Padding for each layer

    'fineTuning': False,  # Freeze layers for fine tuning or not
    'loadTrainedModel': False,
    'path_checkpoint_c5': ('/home/julian/Master/Results_downsampled/Results/'
                           'results_2_2019-11-11_00-38-30-690/checkpoint.pt'),

    'eval_nms': False,  # Use Non-maximum suppression scheme for
                        # finding blobs in preditions.

    'heatmap_scaling_factor': 2,  # Scale down GT-heatmap
    'poolingMax': True,  # Use or omit max-pooling layer

    'gaussian_reduction': 'sum'  # Use either 'sum' or 'max'
})


"""Example configuration for FoveaNet4Sat.
This is also how you derivate custom configurations"""


fov4sat = foveanet_base_config.copy({
    'name': 'FoveaNet4Sat',
    'roobi_size': [200, 200],
    'min_frame_eval': 0,
    'max_frame_eval': 699,
    'min_frame_training': 0,
    'max_frame_training': 699,
    'include_stationary': True,
    'num_epochs': 200,
    'early_stopping_patience': 10,
    'gaussian_std': 1.0,
    'stationary_car_threshold': 3,
    'evaluation_tolerance': 4,
    'evaluation_min_blob_size': 3.5,
    'num_channels': 3,
    'nth_frame_eval': 10,
    'nth_frame_training': 10,
    'save_folder': 'master/conf_9_100_100_test',
    'eval_nms': True,
    'gaussian_reduction': 'max',
    'nms_threshold': 0.35,
    'kernels': [3, 3, 3, 3, 3, 3, 3, 1],
    'paddings': [1, 1, 1, 1, 1, 1, 1, 0],
    'poolingMax': False,
    'heatmap_scaling_factor': 1,
    'batch_size': 16,
})
