"""Module for removing stationary cars"""
import numpy as np
import sys
import helpers


def filter_stationary(ground_truth, nframes, threshold, nth_frame=1):
    if nframes < 3:
        print('Error: filter_stationary. num channels is', nframes)
        sys.exit('stop here')
    frame_nrs = np.unique(ground_truth[:, 3])
    frames_before_and_after = (nframes // 2) * nth_frame
    # Fix boundaries
    frame_nrs = frame_nrs[frame_nrs >= np.min(frame_nrs) +
                          frames_before_and_after]
    if(frame_nrs.shape[0] == 0):
        return ground_truth
    frame_nrs = frame_nrs[frame_nrs <= np.max(frame_nrs) -
                          frames_before_and_after]

    filtered_ground_truth = ground_truth[:]
    for middle_frame_nr in frame_nrs:
        frame_sequence = ground_truth[ground_truth[:, 3] >=
                                      (middle_frame_nr -
                                       frames_before_and_after)]

        frame_sequence = frame_sequence[frame_sequence[:, 3] <=
                                        (middle_frame_nr +
                                         frames_before_and_after)]
        frame_sequence_nrs = np.unique(frame_sequence[:, 3])

        cars_middle_frame = frame_sequence[frame_sequence[:, 3] ==
                                           middle_frame_nr]
        # Sort by id
        cars_middle_frame = cars_middle_frame[cars_middle_frame[:, 0]
                                              .argsort()]

        distances_cars = np.zeros((cars_middle_frame.shape[0], 2))
        distances_cars[:, 0] = cars_middle_frame[:, 0]
        for idx, frame_sequence_frame_nr in enumerate(frame_sequence_nrs):
            cars_current_frame = frame_sequence[frame_sequence_frame_nr ==
                                                frame_sequence[:, 3]]
            cars_current_frame = cars_current_frame[np.isin(
                                                    cars_current_frame[:, 0],
                                                    cars_middle_frame[:, 0])]
            cars_current_frame = cars_current_frame[cars_current_frame[:, 0]
                                                    .argsort()]
            # Not last frame of frame sequence
            if frame_sequence_frame_nr < frame_sequence_nrs[frame_sequence_nrs
                                                            .shape[0] - 1]:
                # No cars, nothing to calculate
                if(cars_current_frame.shape[0] == 0):
                    continue
                # Cars present in middle frame, current_frame AND next_frame
                next_cars = frame_sequence[frame_sequence_nrs[idx + 1] ==
                                           frame_sequence[:, 3]]
                next_cars = next_cars[np.isin(next_cars[:, 0],
                                              cars_current_frame[:, 0])]
                next_cars = next_cars[next_cars[:, 0].argsort()]
                # Cars moved out of frame, handle seperately
                if(next_cars.shape[0] == 0):
                    continue
                cars_current_frame_intersect = cars_current_frame[np.isin(
                    cars_current_frame[:, 0],
                    next_cars[:, 0])]
                pos_0 = cars_current_frame_intersect[:, [1, 2]]
                pos_1 = next_cars[:, [1, 2]]
                difference = pos_0 - pos_1
                power = np.power(difference, 2)
                sum_power = np.sum(power, axis=1)
                distances = np.sqrt(sum_power)
                # Get sum of distances
                for dist_idx, car_id in enumerate(
                        cars_current_frame_intersect[:, 0]):
                    distances_cars[distances_cars[:, 0] == car_id, 1] +=
                    distances[dist_idx]
            else:
                # Vehicle moved out of frame, inf because not stationary
                distances_cars[np.logical_not(
                    np.isin(distances_cars[:, 0],
                            cars_current_frame[:, 0])), 1] = np.inf
                # Cars left, no annotations
                if frame_sequence_nrs[frame_sequence_nrs.shape[0] - 1] <
                middle_frame_nr + frames_before_and_after:
                    # Next frame has no cars, so all current are not stationary
                    for dist_idx, car_id in
                    enumerate(cars_current_frame[:, 0]):
                        distances_cars[distances_cars[:, 0]
                                       == car_id, 1] = np.inf
        below_threshold = distances_cars[distances_cars[:, 1] < threshold, 0]
        filtered_ground_truth = filtered_ground_truth[
            np.logical_not(np.logical_and(np.isin(
                filtered_ground_truth[:, 0], below_threshold),
                filtered_ground_truth[:, 3] == middle_frame_nr))]
    return filtered_ground_truth
