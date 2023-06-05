import os
import cv2
import matplotlib.pyplot as plt


class ImageHandling:

    def __init__(self):
        self.path = ''

    def video_to_frames(self, filename):
        cam = cv2.VideoCapture(self.path + filename)
        tmp_filename = filename.split('.')[0]
        try:
            if not os.path.exists('images'):
                os.makedirs('images')
            if not os.path.exists('images/prediction_set/' + tmp_filename):
                os.makedirs('images/prediction_set/' + tmp_filename)
        except OSError:
            print('Error: OSError creating dir')

        currentframe = 0

        while True:
            bools, frame = cam.read()
            if bools:
                # if video is still left continue creating images
                name = './images/prediction_set/' + tmp_filename + '/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)

                cv2.imwrite(name, frame)
                currentframe += 1
            else:
                break

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

    def set_path(self, path):
        self.path = path

    @staticmethod
    def show_image(path):
        for i, img_path in enumerate(path):
            sp = plt.subplot(1, 3, i + 1)
            sp.axis('Off')
            img = cv2.imread(path)
            plt.imshow(img)
            plt.waitforbuttonpress()

