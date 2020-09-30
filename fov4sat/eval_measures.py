import copy
import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
import foveanet.config as cconf


# checking your OpenCV version using Python
def is_cv2():
    # if we are using OpenCV 2, then our cv2.__version__ will start
    # with '2.'
    return check_opencv_version("2.")


def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")


def is_cv4():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '4.'
    return check_opencv_version("4.")


def check_opencv_version(major):
    # return whether or not the current OpenCV version matches the
    # major version number
    return cv.__version__.startswith(major)


class EvalMeasures():
    '''
    Keeps track of current precision, recall and f1 score.
    '''

    def __init__(self):
        '''
        Ctor.
        '''
        self.reset()
        self.threshold = threshold

    def reset(self):
        '''
        Resets the internal state.
        '''
        self.t_p = 0
        self.f_p = 0
        self.f_n = 0

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def update(self, predictions, targets, offset=0):
        """Updates current amount of detections"""
        _targets = copy.deepcopy(targets)

        outputs = predictions.squeeze(1).numpy()

        if cconf.cfg.eval_nms:
            predictions = self._nms(predictions)

        predictions = predictions.squeeze(1).numpy()

        debug_positions = []
        for idx, prediction in enumerate(predictions):
            true_positive = []
            true_true_positive = []
            false_positive = []
            false_negative = []
            distances_list = []
            if(prediction.max() > 0):
                prediction = np.clip(prediction, 0, None)
                if cconf.cfg.eval_nms:
                    centers_predictions_bool = prediction >=
                    cconf.cfg.nms_threshold
                    centers_predictions_non_zero = np.nonzero(
                        centers_predictions_bool)
                    centers_predictions = np.transpose(
                        centers_predictions_non_zero)
                    if len(centers_predictions) > 0:
                        centers_predictions[:, [0, 1]] =
                        centers_predictions[:, [1, 0]]
                else:
                    prediction_img = ((prediction - prediction.min()) *
                                      (1/(prediction.max() -
                                          prediction.min()) * 255))
                    .astype('uint8')
                    __, predictions_bin = cv.threshold(prediction_img, 0, 255,
                                                       cv.THRESH_BINARY +
                                                       cv.THRESH_OTSU)

                    centers_predictions = self.__find_centroids(
                        predictions_bin)

                for prediction_pos in centers_predictions:
                    if len(_targets[idx]) > 0:
                        distances = _targets[idx] - prediction_pos
                        distances = np.sqrt(np.sum(np.square(distances),
                                                   axis=-1))
                        nearest_idx = np.argmax(distances == distances.min())
                        if distances[nearest_idx] <=
                        cconf.cfg.evaluation_tolerance/np.float32(
                                cconf.cfg.heatmap_scaling_factor):
                            self.t_p += 1
                            true_positive.append(prediction_pos)
                            true_true_positive.append(
                                _targets[idx][nearest_idx])
                            distances_list.append(distances[nearest_idx])
                            if len(_targets[idx]) > 1:
                                mask = np.ones(len(_targets[idx]), dtype=bool)
                                mask[nearest_idx] = False
                                _targets[idx] = _targets[idx][mask, ...]
                            else:
                                _targets[idx] = []
                        else:
                            self.f_p += 1
                            false_positive.append(prediction_pos)
                    else:
                        self.f_p += 1
                        false_positive.append(prediction_pos)

            self.f_n += len(_targets[idx])
            false_negative.extend(_targets[idx])
            pos_dict = {"tp": true_positive, "fp": false_positive,
                        "fn": false_negative, "distances": distances_list,
                        "ttp": true_true_positive}
            debug_positions.append(pos_dict)
        return debug_positions

    def __find_centroids(self, binary_image):
        """Finds centroids in bianry image"""
        centers = np.array([], dtype=np.int32).reshape(0, 2)
        output = cv.connectedComponentsWithStats(binary_image, 4, cv.CV_32S)
        stats = output[2]
        for idx, stat in enumerate(stats[1:, :]):
            if stat[cv.CC_STAT_AREA] >= cconf.cfg.evaluation_min_blob_size:
                x = output[3][idx + 1, 0]
                y = output[3][idx + 1, 1]
                centers = np.vstack([centers, [int(x), int(y)]])
        return centers

    def get_measures(self):
        """Provides measures for evaluation"""
        precision = 0
        recall = 0
        f1_score = 0
        if self.t_p > 0:
            precision = self.t_p / (self.t_p + self.f_p)
            recall = self.t_p / (self.t_p + self.f_n)
        if (precision + recall) > 0:
            f1_score = 2*(precision * recall / (precision + recall))
        return precision, recall, f1_score
