import os

if __name__ == '__main__':
    video_folder = '../videos'
    videos = os.listdir(video_folder)
    videos.sort()
    num_each_set = len(videos) / 3

    split_1 = './video_split_1.txt'
    with open(split_1, 'wb') as fw:
        for video in videos[:num_each_set]:
            fw.write(video+'\n')

    split_2 = './video_split_2.txt'
    with open(split_2, 'wb') as fw:
        for video in videos[num_each_set:2*num_each_set]:
            fw.write(video+'\n')

    split_3 = './video_split_3.txt'
    with open(split_3, 'wb') as fw:
        for video in videos[2*num_each_set:]:
            fw.write(video+'\n')
