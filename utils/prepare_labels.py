"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io

import sys

sys.path.insert(0, '../../coco-caption')  # Hack to allow the import of pycocoeval
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

tokenizer = PTBTokenizer()


def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    print('number of vid: %d' % (len(imgs)))
    cur_res = {}
    ind = 0
    for img in imgs.keys():
        # print(imgs[img]['sentences'])
        for sent in imgs[img]['sentences']:
            # tokens = sent.strip('.').split(' ')
            cur_res[ind] = [{'caption': remove_nonascii(sent)}]
            ind += 1
    tokens_sent = tokenizer.tokenize(cur_res)
    all_tok = []
    max_len = 0
    for tok in tokens_sent:
        tokens_sent[tok] = tokens_sent[tok][0].split(' ')
        # if(len(tokens_sent[tok]) == 73):
        #        print(tokens_sent[tok])
        max_len = max(max_len, len(tokens_sent[tok]))
        all_tok += tokens_sent[tok]
    for w in all_tok:
        counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    # badbad_words = [w for w,n in counts.items() if n <= 0.001]
    # print('bad_words: ', bad_words)
    vocab = [w for w, n in counts.items() if n > count_thr]
    vocab = ['<pad>', '<start>', '<end>'] + vocab
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    cur_res = {}
    ind = 0
    for img in imgs:
        for sent in imgs[img]['sentences']:
            # cur_res = {}
            # cur_res[img] = [{'caption': remove_nonascii(sent)}]
            txt = tokens_sent[ind]  # tokenizer.tokenize(cur_res)
            ind += 1
            # txt = sent.strip('.').split(' ') #sent['tokens']

            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    ind = 0
    for img in imgs:
        imgs[img]['final_captions'] = []
        for sent in imgs[img]['sentences']:
            # txt = sent['tokens']
            txt = tokens_sent[ind]  # tokenizer.tokenize(cur_res)
            ind += 1
            # txt = sent.strip('.').split(' ') #sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            imgs[img]['final_captions'].append(caption)

    return vocab, max_len


def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(imgs[img]['final_captions']) for img in imgs)  # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(imgs[img]['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')

        for j, s in enumerate(imgs[img]['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length


def ship_to_finalcaption(imgs, wtoi, vocab, max_len):
    """
    tokenize the caption and transform some of the words into UNK
    so that we can encode it with one hot fetaure later
    """
    cur_res = {}
    ind = 0
    for img in imgs.keys():
        # print(imgs[img]['sentences'])
        for sent in imgs[img]['sentences']:
            # tokens = sent.strip('.').split(' ')
            cur_res[ind] = [{'caption': '<start> ' + remove_nonascii(sent) + ' <end>'}]
            ind += 1
    tokens_sent = tokenizer.tokenize(cur_res)
    for tok in tokens_sent:
        tokens_sent[tok] = tokens_sent[tok][0].split(' ')
    ind = 0
    for img in imgs:
        imgs[img]['final_captions'] = []
        imgs[img]['final_captions_wordid'] = []
        for sent in imgs[img]['sentences']:
            # txt = sent['tokens']
            txt = tokens_sent[ind]  # tokenizer.tokenize(cur_res)
            ind += 1
            # txt = sent.strip('.').split(' ') #sent['tokens']
            caption = [w if w in vocab else 'UNK' for w in txt]
            wordid = [wtoi[w] for w in caption]
            if (img == 'v_DCYz8p4zH6o'):
                print(wordid)
                print(caption)
            if (len(wordid) < max_len):
                wordid += [0] * (max_len - len(wordid))
            if (len(wordid) > max_len):
                wordid = wordid[:max_len]
            imgs[img]['final_captions'].append(caption)
            imgs[img]['final_captions_wordid'].append(wordid)


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    # imgs = imgs['images']

    seed(123)  # make reproducible

    # create the vocab
    vocab, max_len = build_vocab(imgs, params)
    max_len = min(params['max_length'], max_len)
    print('max_len: ', max_len)
    # json.dump(vocab, open('data/activity_caption_vocab.json', 'w'))

    # normal vacab should start from 1 instead of 0, but we add <pad> so we start from 0
    # itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    # wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
    itow = {i: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table

    data = {'vocabulary': vocab, 'itow': itow, 'wtoi': wtoi}
    json.dump(data, open(params['output_json'], 'w'))
    print('save as %s' % params['output_json'])
    data = {'vocabulary': 'vocab', 'itow': 'itow dict', 'wtoi': 'wtoi dict'}
    print('data format: : ', data)

    # trainsform val_json caption into final caption
    for rawjson in [params['train_json'], params['val1_json'], params['val2_json']]:
        print('transforming rawjson: %s using vocabulary' % rawjson)
        imgs = json.load(open(rawjson, 'r'))
        ship_to_finalcaption(imgs, wtoi, vocab, max_len)
        result_json = params['output_json'].split('.json')[0] + "_transformed_start_end_" + rawjson.split('/')[-1]
        json.dump(imgs, open(result_json, 'w'))
        print('save as %s' % result_json)
    print('max_length of caption: %d' % max_len)
    # encode captions in large arrays, ready to ship to hdf5 file
    '''
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

    # create output h5 file
    N = len(imgs)
    f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['images'] = []
    for i,img in enumerate(imgs):

      jimg = {}
      jimg['split'] = params['input_json'].split('.json')[0] #'train' #imgs[img]['split']
      if 'filename' in img: jimg['file_path'] = os.path.join(img['filepath'], img['filename']) # copy it over, might need
      if 'cocoid' in img: jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)

      out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data.json', help='output json file')
    parser.add_argument('--output_h5', default='data', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    parser.add_argument('--train_json', required=True,
                        help='input json file to process into one hot feature based on the vocab')
    parser.add_argument('--val1_json', required=True,
                        help='input json file to process into one hot feature based on the vocab')
    parser.add_argument('--val2_json', required=True,
                        help='input json file to process into one hot feature based on the vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
