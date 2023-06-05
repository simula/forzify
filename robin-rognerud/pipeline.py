from sbd_test import SBD
from logoDetection import LogoDetection
from loadAndPredict import Predict
from readMatchTxt import MatchTXT
from imageHandling import ImageHandling
from clipVideos import Clip

import os


path_to_frames = './images/prediction_set/'
path_to_game = '/Users/robin/Documents/master/full_game/'
path_to_clip = 'card_vids/'
filename = 'game_1824'
txt = 'meta_1824.txt'


def handle_full_game(txt_file):
    """
    Clips a full game to yellow card events based on meta text file

    :param txt_file: name of the corresponding text-file
    :return:
    """
    mt = MatchTXT()
    mt.fetch_yellow_card_timestamps(path_to_game + txt_file)
    mt.fetch_clips(path_to_game, filename)


def clip_to_frames():
    """
    Splits each of the clips in the full game into frames, if not already done.

    :return:
    """

    im = ImageHandling()
    for i in os.listdir('./card_vids/' + filename):
        if not os.path.exists('./images/prediction_set/' + filename + '/' + i[:-4]) and i.endswith('.mp4'):
            print(i[:-4])
            im.video_to_frames(filename + '/' + i)


def get_logo_list(clip_name):
    """
    Makes a model based on a local dataset if the model is not apparent.
    Then predicts logo or game on a dataset of frames

    :return: a list of frames that begins a logo transition
    """
    if not os.path.exists('./models/logo-detection-simple.h5'):
        model = LogoDetection()
        model.generate_data()
        model.create_model()
        model.save_model()

    pred = Predict()
    pred.predict_images(filename + '/' + clip_name)
    pred.find_seq_v2()
    pred.handle_stretches(pred.sequence_list)
    print(pred.logo_list)
    return pred.startframes


def get_sbd_list(clip_name):
    """
    Detects scene changes in a clip

    :return: a list of frames beginning a scene change (poor on logo transitions)
    """
    sbd = SBD()
    sbd.predict_sbd(filename + '/' + clip_name)
    return sbd.read_predicted_sbd(filename + '/' + clip_name)


def make_clip(logo_list, sbd_list, card):
    clip = Clip()
    clip.set_lists(logo_list, sbd_list)
    clip.decide_start()
    clip.decide_end()
    clip.construct_clip(filename, card)




def main():
    handle_full_game(txt)
    clip_to_frames()

    while True:
        lst = os.listdir('images/prediction_set/' + filename)
        print(lst)
        q1 = input('Choose a clip from above list, use index: (exit to break) ')
        if q1.lower() == 'exit':
            break
        logo_list = get_logo_list(lst[int(q1)-1])
        sbd_list = get_sbd_list(lst[int(q1)-1])
        print('Timestamp of start of logo transitions (frame nr): ')
        print(logo_list)
        print('Timestamp of sbd (frame nr) : ')
        print(sbd_list)
        make_clip(logo_list, sbd_list, lst[int(q1)-1])


main()
