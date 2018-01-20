import numpy as np
import torch
from temporal_roi_pooling import RoIPool


if __name__ == '__main__':
    data_path = './v_V9nOM1VWdnc.npy'
    data = np.load(data_path)

    net = RoIPool(7, 7, 2, 1.0/16, 1.0/8)
    features = np.expand_dims(data, axis=0)
    features = torch.from_numpy(np.expand_dims(data, axis=0))
    features = features.cuda()

    rois = [[0, 0, 0, 0, 0, 0, 1]]
    rois = torch.from_numpy(rois)
    rois = rois.cuda()

    RoIPool.forward(features, rois)
