import c3d_model
import torch
import numpy as np
from torch.autograd import Variable
import os
from glob import glob
import torch.nn as nn
import scipy.io as sio
import math
import subprocess
import image_io
def get_probs(prediction):
    exp_scores = np.exp(prediction)
    probs = exp_scores / np.sum(exp_scores, keepdims=True)
    return probs
def preproc_img(img, mean, imageSz,crop_size):
    img = img.astype(np.float32, copy=False)
    img = image_io.resize_image(img, imageSz)
    h_off = (imageSz[0] - crop_size)/2
    w_off = (imageSz[1] - crop_size)/2
    img = img[h_off:h_off+crop_size,w_off:w_off+crop_size,:]
    img = img.transpose((2,0,1))
    img = img[(2,1,0), :, :]
    img *= 255 # input image is in range(0,1)
    if mean.ndim == 1:
        mean = mean[:,np.newaxis, np.newaxis]
    img -= mean
    return img

def get_clip(clip, verbose = False):
    '''
    :param clip_name: str, the name of the clip (subfolder in c3d-pytorch/data)
    :param verbose: if True, show the unrolled clip
    :return:
    '''

    img_mean = np.asarray([128])
    imageSz = (128, 171)
    crop_size = 112

    # clip = sorted(glob(os.path.join('./c3d-pytorch/data', clip_name, '*.jpg')))

    clip = np.array([preproc_img(image_io.load_image(frame), mean=img_mean, imageSz=imageSz, crop_size = crop_size) for frame in clip])
    clip = clip.transpose(1,0,2,3)
    clip = clip[np.newaxis, :, :, :, :]
    return torch.from_numpy(clip)

def get_video_list(video_folder):
    videos = os.listdir(video_folder)
    # videos.sort()
    return videos

def check_exist(video_name):
    base_name = video_name.split('.')[0]
    feat_name = base_name+'.npy'
    if_exist = 0
    if os.path.exists(os.path.join('../features', feat_name)):
        if_exist = 1
    elif os.path.exists(os.path.join('../tmp', base_name)):
        if_exist = 1
    return if_exist

def count_imgs(src_folder):
    num_files = len([f for f in os.listdir(src_folder)
                     if os.path.isfile(os.path.join(src_folder,f))])
    return num_files


def decode_frames(video_name, max_frames):
    src_video = os.path.join('../videos', video_name)
    base_name = video_name.split('.')[0]
    dst_folder = os.path.join('../tmp', base_name)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    cmd = 'ffmpeg -i '+ src_video+ ' ' + os.path.join(dst_folder, '%6d.jpg')+ ' -hide_banner -loglevel quiet'
    # os.system(cmd)
    subprocess.call(cmd, shell=True)
    frames = sorted(glob(os.path.join('../tmp', base_name, '*.jpg')))
    # print frames[:10]
    # assert False
    # print len(frames)
    sample_rate = int(math.ceil(1.0 * len(frames) / max_frames))
    if len(frames) == 0:
        return len(frames),frames
    frames = frames[::sample_rate]
    # print len(frames)

    return len(frames),frames

def remove_tmp_folder(video_name):
    base_name = video_name.split('.')[0]
    dst_folder = os.path.join('../tmp', base_name)
    cmd = 'rm -rf '+dst_folder
    subprocess.call(cmd, shell=True)
    # os.system(cmd)

if __name__ == '__main__':
    video_folder = '../videos'
    max_frames = 800

    model = c3d_model.c3d_model
    model_path = './conv3d_deepnetA_sport1m_iter_1900000.pth'
    model.load_state_dict(torch.load(model_path))
    model = nn.Sequential(*list(model.children())[:-10])
    model.cuda()
    model.eval()

    videos = get_video_list(video_folder)
    with open('./video_split_3.txt','rb') as fr:
        lines = fr.readlines()
        for index,line in enumerate(lines):
            video = line.rstrip('\r\n')
            print 'Processing {}th video'.format(index)
            if check_exist(video) == 1:
                continue
            num_frames, frames = decode_frames(video, max_frames)
            if num_frames == 0:
                remove_tmp_folder(video)
                continue
            X = get_clip(frames)
            X = Variable(X)
            X = X.cuda()
            prediction = model(X)
            prediction = prediction.data.cpu().numpy()
            prediction = np.squeeze(prediction)
            np.save(os.path.join('../features', video.split('.')[0]+'.npy'), prediction)
            remove_tmp_folder(video)

