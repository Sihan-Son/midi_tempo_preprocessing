import numpy as np
import glob
import datetime
import math
import random
import os
import shutil
import matplotlib.pyplot as plt
import pretty_midi
from pypianoroll import Multitrack, Track
import write_midi
import librosa.display
from utils import *

import tensorflow as tf

ROOT_PATH = 'D:\Project\PPProcessing\datasets'
# 'D:/dataSets/music/Datasets'
# '/Users/sumuzhao/Downloads/'
test_ratio = 0.1
LAST_BAR_MODE = 'remove'


def get_bar_piano_roll(piano_roll):
    if int(piano_roll.shape[0] % 64) is not 0:
        if LAST_BAR_MODE == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((64 - piano_roll.shape[0] % 64, 128))), axis=0)
        elif LAST_BAR_MODE == 'remove':
            piano_roll = np.delete(piano_roll, np.s_[-int(piano_roll.shape[0] % 64):], axis=0)
    piano_roll = piano_roll.reshape(-1, 64, 128)
    return piano_roll


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track


def save_midis(bars, file_path, tempo=80.0):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
    pause = np.zeros((bars.shape[0], 64, 128, bars.shape[3]))
    images_with_pause = padded_bars
    images_with_pause = images_with_pause.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
    images_with_pause_list = []
    for ch_idx in range(padded_bars.shape[3]):
        images_with_pause_list.append(images_with_pause[:, :, :, ch_idx].reshape(images_with_pause.shape[0],
                                                                                 images_with_pause.shape[1],
                                                                                 images_with_pause.shape[2]))
    # write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[33, 0, 25, 49, 0],
    #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)
    write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[0], is_drum=[False], filename=file_path,
                                         tempo=tempo, beat_resolution=4)


"""1. divide the original set into train and test sets"""
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/rock/rock_midi'))]
# print(l)
# idx = np.random.choice(len(l), int(test_ratio * len(l)), replace=False)
# print(len(idx))
# for i in idx:
#     shutil.move(os.path.join(ROOT_PATH, 'MIDI/rock/rock_midi', l[i]),
#                 os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/origin_midi', l[i]))

"""2. convert_clean_test.py"""

"""3. choose the clean midi from original sets"""
# if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_midi')):
#     os.makedirs(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_midi'))
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner'))]
# print(l)
# print(len(l))
# for i in l:
#     shutil.copy(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/origin_midi', os.path.splitext(i)[0] + '.mid'),
#                 os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_midi', os.path.splitext(i)[0] + '.mid'))

"""4. merge and crop"""
# if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_midi_gen')):
#     os.makedirs(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_midi_gen'))
# if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_npy')):
#     os.makedirs(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_npy'))
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_midi'))]
# print(l)
# count = 0
# for i in range(len(l)):
#     try:
#         multitrack = Multitrack(beat_resolution=4, name=os.path.splitext(l[i])[0])
#         x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_midi', l[i]))
#         multitrack.parse_pretty_midi(x)
#
#         category_list = {'Piano': [], 'Drums': []}
#         program_dict = {'Piano': 0, 'Drums': 0}
#
#         for idx, track in enumerate(multitrack.tracks):
#             if track.is_drum:
#                 category_list['Drums'].append(idx)
#             else:
#                 category_list['Piano'].append(idx)
#         tracks = []
#         merged = multitrack[category_list['Piano']].get_merged_pianoroll()
#         print(merged.shape)
#
#         pr = get_bar_piano_roll(merged)
#         print(pr.shape)
#         pr_clip = pr[:, :, 24:108]
#         print(pr_clip.shape)
#         if int(pr_clip.shape[0] % 4) != 0:
#             pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
#         pr_re = pr_clip.reshape(-1, 64, 84, 1)
#         print(pr_re.shape)
#         save_midis(pr_re, os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_midi_gen', os.path.splitext(l[i])[0] +
#                                        '.mid'))
#         np.save(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_npy', os.path.splitext(l[i])[0] + '.npy'), pr_re)
#     except:
#         count += 1
#         print('Wrong', l[i])
#         continue
# print(count)

"""5. concatenate into a big binary numpy array file"""
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_npy'))]
# print(l)
# train = np.load(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_npy', l[0]))
# print(train.shape, np.max(train))
# for i in range(1, len(l)):
#     print(i, l[i])
#     t = np.load(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/cleaner_npy', l[i]))
#     train = np.concatenate((train, t), axis=0)
# print(train.shape)
# np.save(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/rock_test_piano.npy'), (train > 0.0))

"""6. separate numpy array file into single phrases"""
# if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/phrase_test')):
#     os.makedirs(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/phrase_test'))
# x = np.load(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/rock_test_piano.npy'))
# print(x.shape)
# count = 0
# for i in range(x.shape[0]):
#     if np.max(x[i]):
#         count += 1
#         np.save(os.path.join(ROOT_PATH, 'MIDI/rock/rock_test/phrase_test/rock_piano_test_{}.npy'.format(i+1)), x[i])
#         print(x[i].shape)
#    # if count == 11216:
#    #     break
# print(count)

"""some other codes"""
# filepaths = []
# msd_id_list = []
# for dirpath, _, filenames in os.walk(os.path.join(ROOT_PATH, 'MIDI/Sinfonie Data')):
#     for filename in filenames:
#         if filename.endswith('.mid'):
#             msd_id_list.append(filename)
#             filepaths.append(os.path.join(dirpath, filename))
# print(filepaths)
# print(msd_id_list)
# for i in range(len(filepaths)):
#     shutil.copy(filepaths[i], os.path.join(ROOT_PATH, 'MIDI/classic/classic_midi/{}'.format(msd_id_list[i])))

# x1 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_1.npy'))
# x2 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_2.npy'))
# x3 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_3.npy'))
# x4 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_4.npy'))
# x5 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_5.npy'))
# x = np.concatenate((x1, x2, x3, x4, x5), axis=0)
# print(x.shape)
# np.save(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano.npy'), x)


# multitrack = Multitrack(beat_resolution=4, name='YMCA')
# x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/origin/YMCA.mid'))
# multitrack.parse_pretty_midi(x)
#
# category_list = {'Piano': [], 'Drums': []}
# program_dict = {'Piano': 0, 'Drums': 0}
#
# for idx, track in enumerate(multitrack.tracks):
#     if track.is_drum:
#         category_list['Drums'].append(idx)
#     else:
#         category_list['Piano'].append(idx)
# tracks = []
# merged = multitrack[category_list['Piano']].get_merged_pianoroll()
#
# # merged = multitrack.get_merged_pianoroll()
# print(merged.shape)
#
# pr = get_bar_piano_roll(merged)
# print(pr.shape)
# pr_clip = pr[:, :, 24:108]
# print(pr_clip.shape)
# if int(pr_clip.shape[0] % 4) != 0:
#     pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
# pr_re = pr_clip.reshape(-1, 64, 84, 1)
# print(pr_re.shape)
# save_midis(pr_re, os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/merged_midi/YMCA.mid'), 127)
# np.save(os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/merged_npy/YMCA.npy'), (pr_re > 0.0))
