"""Base class for datasets used with FoveaNet4Sat"""
from abc import ABCMeta, abstractmethod

import helpers
from torch.utils.data import Dataset
import pandas as pd


class FoveanetDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, csv_file, data_dir, roobi_size, amount_channels,
                 frame_limits, transforms=None, nth_frame=1,
                 include_empty=False):
        self.amount_channels = amount_channels
        self.data_dir = data_dir
        self.transform = transform
        self.roobi_size = roobi_size
        self.nth_frame = nth_frame

        min_frame = int(frame_limits[0])
        max_frame = int(frame_limits[1])
        frames_before_and_after = (amount_channels // 2) * nth_frame

        if helpers.is_non_zero_file(csv_file):
            self.gt_frame = pd.read_csv(csv_file, header=None)
        else:
            self.gt_frame = pd.DataFrame(columns=[0, 1, 2, 3])

        if not cconf.cfg.include_stationary:
            filtered_gt = stationary_filer.filter_stationary(
                self.gt_frame.values, 5, cconf.cfg.stationary_car_threshold,
                nth_frame)
            self.gt_frame = pd.DataFrame(data=filtered_gt)

        if not include_empty:
            # Frame idxs of frames with cars in them
            self.frame_numbers = np.unique(self.gt_frame.values[:, 3])
            self.frame_numbers = self.frame_numbers[self
                                                    .frame_numbers.argsort()]
        else:
            self.frame_numbers = np.arange(min_frame, max_frame+1)

        # Fix boundaries
        self.frame_numbers = self.frame_numbers[self.frame_numbers >=
                                                min_frame +
                                                frames_before_and_after]
        self.frame_numbers = self.frame_numbers[self.frame_numbers <=
                                                max_frame -
                                                frames_before_and_after]
        # Remove frames that are on boundary
        if self.frame_numbers.size > 0:
            self.gt_frame = self.gt_frame[self.gt_frame.iloc[:, 3] >=
                                          self.frame_numbers[0]]
            self.gt_frame = self.gt_frame[self.gt_frame.iloc[:, 3] <=
                                          self.frame_numbers[
                                                len(self.frame_numbers)-1]]

    def __len__(self):
        return len(self.frame_numbers)

    @abstractmethod
    def __get_groundtruth__(self, idx):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    def __get_path_stacks(self):
        path_stacks = []
        frames_before_and_after = (self.channels // 2) * self.nth_frame

        for frame_number in self.frame_numbers:
            path_stack = []
            central_frame_number = int(frame_number)
            regex = r""
            for i in range(central_frame_number-frames_before_and_after,
                           central_frame_number+frames_before_and_after+1,
                           self.nth_frame):
                regex = regex + r'((-010*{}-VIS)|(img0*{}.png))'.format(i, i)
                if i < central_frame_number+frames_before_and_after:
                    regex = regex + r'|'
            current_dim = 0
            for _, _, files in os.walk(self.root_dir):
                for file in sorted(files):
                    if re.search(regex, file):
                        file_path = os.path.join(self.root_dir, file)
                        path_stack.append(file_path)
                        current_dim += 1
                        if(current_dim == self.channels):
                            break
            path_stacks.append(path_stack)
        return path_stacks

    def __getitem__(self, idx):
        positions = self.gt_frame.values[:, [1, 2]]
        positions = positions[self.gt_frame.values[:, 3] ==
                              self.frame_numbers[idx]]
        positions = positions/np.float32(cconf.cfg.heatmap_scaling_factor)
        positions = positions.astype(np.float32)
        frame_stack = np.zeros((self.roobi_size[0], self.roobi_size[1],
                                self.channels), dtype=np.uint8)
        path_stack = self.path_stacks[idx]
        for channel, path in enumerate(path_stack):
            frame_stack[:, :, channel] = cv.imread(path, 0)
        groundtruth = self.__get_groundtruth__(idx)
        sample = {'frame': frame_stack, 'heatmap': groundtruth,
                  'positions': positions}
        if self.transform:
            sample['frame'] = self.transform(sample['frame'])
        return sample
