import numpy as np
import torch
from torch.autograd import Variable
from temporal_roi_pooling import RoIPool


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


if __name__ == '__main__':
    data_path = './v_V9nOM1VWdnc.npy'
    data = np.load(data_path)

    net = RoIPool(7, 7, 2, 1.0/16, 1.0/8)
    features = np.expand_dims(data, axis=0)
    features = np_to_variable(np.expand_dims(data, axis=0))

    rois = np.asarray([[0, 0, 0, 0, 0, 0, 1],[0,1,1,1,1,0,1]])
    rois = np_to_variable(rois, dtype=torch.LongTensor)

    outputs = net.forward(features, rois)
    print type(outputs)
    
