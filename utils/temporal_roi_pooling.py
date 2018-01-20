import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RoIPool(nn.Module):
    def __init__(self, pooled_height, pooled_width, pooled_length, spatial_scale, temporal_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.pooled_length = int(pooled_length)

        self.spatial_scale = float(spatial_scale)
        self.temporal_scale = float(temporal_scale)

    def forward(self, features, rois):
        # assume features: [batch_size, num_channels, length, height, width]
        batch_size, num_channels, data_length, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.pooled_length, self.pooled_height, self.pooled_width)).cuda()

        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data[0])
            # print roi.data[0]
            # batch_ind = int(roi[0])
            # rois data struct:
            # roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(
            #     roi[1:].data.cpu().numpy() * self.spatial_scale).astype(int)
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(
                roi[1:-2].data.cpu().numpy() * self.spatial_scale).astype(int)
            roi_start_l, roi_end_l = np.round(
                roi[-2:].data.cpu().numpy() * self.temporal_scale).astype(int)
            # fixed rois in spatial sapce as only temporal rois are considered.
            roi_start_w = 0
            roi_start_h = 0
            roi_end_w = 6
            roi_end_h = 6

            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            roi_length = max(roi_end_l - roi_start_l + 1, 1)

            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)
            bin_size_l = float(roi_length) / float(self.pooled_length)

            for pl in range(self.pooled_length):
                lstart = int(np.floor(pl * bin_size_l))
                lend = int(np.floor((pl + 1) * bin_size_l))
                lstart = min(data_length, max(0, lstart + roi_start_l))
                lend = min(data_length, max(0, lend + roi_start_l))

                for ph in range(self.pooled_height):
                    hstart = int(np.floor(ph * bin_size_h))
                    hend = int(np.ceil((ph + 1) * bin_size_h))
                    hstart = min(data_height, max(0, hstart + roi_start_h))
                    hend = min(data_height, max(0, hend + roi_start_h))
                    for pw in range(self.pooled_width):
                        wstart = int(np.floor(pw * bin_size_w))
                        wend = int(np.ceil((pw + 1) * bin_size_w))
                        wstart = min(data_width, max(0, wstart + roi_start_w))
                        wend = min(data_width, max(0, wend + roi_start_w))

                        is_empty = (hend <= hstart) or(wend <= wstart) or (lstart <= lend)
                        if is_empty:
                            outputs[roi_ind, :, pl, ph, pw] = 0
                        else:
                            data = features[batch_ind]
                            outputs[roi_ind, :, pl, ph, pw] = torch.max(
                                torch.max(data[:, lstart:lend, hstart:hend, wstart:wend], 2)[0], 3)[0].view(-1)
        return outputs

