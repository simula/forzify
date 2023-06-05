import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import os


class SBD:

    def __init__(self):
        self.path = 'card_vids/'
        self.scene_list = ''

    def predict_sbd(self, filename):
        final_path = self.path + filename + '.mp4'
        subprocess.run(["transnetv2_predict", final_path], stderr=subprocess.PIPE, shell=True, text=True)

    def read_predicted_sbd(self, filename):
        with open(self.path + filename + '.mp4.scenes.txt') as f:
            lines = f.readlines()
            new_scenes = []
            for i in lines:
                tmp_str = i.replace('\n', '')
                tmp_str = tmp_str.split(' ')[1]
                new_scenes.append(int(tmp_str))
            # new_scenes.pop()  # Remove last occurrence as it is the end of the video
            self.scene_list = new_scenes
        return new_scenes

    def plot_stretch(self, video_number):
        length = len(self.scene_list)
        store = input("Which scene? (max: " + str(length) + ")")
        if int(store) < 1 or int(store) > length:
            print("Pick another number")
        frame_nr = int(self.scene_list[int(store) - 1])
        print(frame_nr)
        frame_stretch = []
        for i in range(frame_nr - 2, frame_nr + 3):
            file = "images/prediction_set/" + str(video_number) + "/frame" + str(i) + ".jpg"
            frame_stretch.append(file)
        print(frame_stretch)
        f, axarr = plt.subplots(1, 5, figsize=(50, 50))
        axarr[0].imshow(Image.open(frame_stretch[0]))
        axarr[1].imshow(Image.open(frame_stretch[1]))
        axarr[2].imshow(Image.open(frame_stretch[2]))
        axarr[3].imshow(Image.open(frame_stretch[3]))
        axarr[4].imshow(Image.open(frame_stretch[4]))
        plt.show()

