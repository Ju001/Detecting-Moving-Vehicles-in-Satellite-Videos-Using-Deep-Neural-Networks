"""Implementation of FoveanetDataset using gaussian heatmaps as GT"""
import foveanet.config as cconf
from gaussian import draw_gaussian


class HeatmapFoveanetDataset(FoveanetDataset):
    def __init__(self, csv_file, data_dir, size, amount_channels, frame_limits,
                 transform=None, nth_frame=1, include_empty=False):
        super(HeatmapFoveanetDataset, self).__init__(csv_file, data_dir,
                                                     roobi_size,
                                                     amount_channels,
                                                     frame_limits,
                                                     transform,
                                                     nth_frame,
                                                     include_empty)
        scaling_factor = cconf.cfg.heatmap_scaling_factor
        self.heatmaps = np.zeros((len(self.frame_numbers),
                                  self.roobi_size[0]//scaling_factor,
                                  self.roobi_size[1]//scaling_factor),
                                 dtype=np.float32)

        for idx, frame_number in enumerate(self.frame_numbers):
            positions = self.gt_frame.values[:, [1, 2]]
            positions = positions[self.gt_frame.values[:, 3] == frame_number]
            positions = positions//scaling_factor
            positions = positions.astype(np.float32)
            heatmap = np.zeros((self.roobi_size[0]//scaling_factor,
                                self.roobi_size[1]//scaling_factor),
                               dtype=np.float32)

            for position in positions:
                heatmap = draw_gaussian(heatmap, position,
                                        cconf.cfg.gaussian_std,
                                        mode=cconf.cfg.gaussian_reduction)
            self.heatmaps[idx] = heatmap
            heatmap = np.clip(heatmap, 0.0, 1.0)

    def __get_groundtruth__(self, idx):
        heatmap = self.heatmaps[idx]
        heatmap = heatmap[np.newaxis, ...]
        return torch.from_numpy(heatmap)

    @staticmethod
    def collate_fn(batch):
        frames = list()
        heatmaps = list()
        positions = list()

        for b in batch:
            frames.append(b['frame'])
            heatmaps.append(b['heatmap'])
            positions.append(b['positions'])

        frames = torch.stack(frames, dim=0)
        heatmaps = torch.stack(heatmaps, dim=0)

        return frames, heatmaps, positions
