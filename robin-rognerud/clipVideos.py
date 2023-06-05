import os
import cv2
import ffmpeg


def static_clip(filename, card):
    start_end = [520, 1145]  # Change when necessary
    fps, width, height = 25, 1280, 720

    target_path = 'new_clips/avc1/' + filename + '_' + card + '_static.mp4'
    out = cv2.VideoWriter(target_path, cv2.VideoWriter_fourcc(*'avc1'),
                          fps, (width, height))
    image_folder = './images/prediction_set/' + filename + '/' + card
    images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]

    get_number = lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else float(
        'inf')
    sorted_img_names = sorted(images, key=get_number)

    for image in sorted_img_names:
        num = int(''.join(filter(str.isdigit, image)))
        if start_end[0] <= num <= start_end[1]:
            img = cv2.imread(os.path.join(image_folder, image))
            out.write(img)
    out.release()
    cv2.destroyAllWindows()


class Clip:

    def __init__(self):

        self.logo_list = []
        self.sbd_list = []
        self.frames_to_clip = []

    def set_lists(self, logo_list, sbd_list):
        self.logo_list = logo_list
        self.sbd_list = sbd_list

    def decide_start(self):
        # Fallback option
        start_frame = 450

        # If first scene change is within 12 seconds of annotated event, and before first LT
        for item in self.sbd_list:
            if self.logo_list:
                if 450 <= item <= 750 and item < self.logo_list[0]:
                    start_frame = item
                    break
        self.frames_to_clip.append(start_frame)

    def decide_end(self):
        # Fallback value
        end_frame = 1125
        if len(self.logo_list) > 2:
            self.logo_list = self.logo_list[-2:]

        # 2 LT, add 20 for finishing the LT in the clip
        if len(self.logo_list) == 2:
            end_frame = self.logo_list[1] + 20

        # 1 LT, also add 20
        if len(self.logo_list) == 1:
            tmp_list = []
            for item in self.sbd_list:
                if item > self.logo_list[0]:
                    tmp_list.append(item)
                    if len(tmp_list) == 2:
                        break
            end_frame = tmp_list[-1] + 20
        self.frames_to_clip.append(end_frame)

    def construct_clip(self, filename, card):
        sorted_frames_list = sorted(self.frames_to_clip)
        print('Start: ' + str(sorted_frames_list[0]) + ' and end: ' + str(sorted_frames_list[1]) + ' of new clip')
        fps, width, height = 25, 1280, 720
        target_path = 'new_clips/' + filename + '_' + card + '_edit.mp4'
        out = cv2.VideoWriter(target_path, cv2.VideoWriter_fourcc(*'avc1'),
                              fps, (width, height))
        image_folder = './images/prediction_set/' + filename + '/' + card
        images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]

        get_number = lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else float(
            'inf')
        sorted_img_names = sorted(images, key=get_number)

        for image in sorted_img_names:
            num = int(''.join(filter(str.isdigit, image)))
            if sorted_frames_list[0] <= num <= sorted_frames_list[1]:
                img = cv2.imread(os.path.join(image_folder, image))
                out.write(img)
        out.release()
        cv2.destroyAllWindows()

