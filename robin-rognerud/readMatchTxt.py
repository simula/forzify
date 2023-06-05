import re
import os
from datetime import datetime
from moviepy.editor import *


class MatchTXT:

    def __init__(self):

        self.path = '/Users/robin/Documents/master/full_game/'

        self.start_time = ''
        self.timestamps = []

    def set_path(self, path):
        self.path = path

    def fetch_yellow_card_timestamps(self, path):
        regex_time = r'\d\d:\d\d:\d\d'
        regex_event_yellow = 'yellow card'
        regex_event_red = 'red card'
        card_events_time = []
        with open(path, 'r') as file:
            for line in file:
                if re.search('start timestamp', line):
                    self.start_time = re.search(regex_time, line).group()
                if re.search(regex_event_yellow, line) or re.search(regex_event_red, line):
                    card_events_time.append(re.search(regex_time, line).group())
            # print(card_events_time)

        # Handle datetime -> video time, returns seconds after start
        video_timestamps = []
        if self.start_time == '':
            print('No start timestamp')
        dt_start = datetime.strptime(self.start_time, '%H:%M:%S')
        for i in card_events_time:
            video_event_time = datetime.strptime(i, '%H:%M:%S')
            diff = video_event_time-dt_start
            video_timestamps.append(diff.seconds)
        self.timestamps = video_timestamps
        if len(video_timestamps) == 0:
            print('No cards this game')

    def fetch_clips(self, path, filename):
        timestamps = self.timestamps
        clip = VideoFileClip(path + filename + '.mp4')
        try:
            if not os.path.exists('card_vids'):
                os.makedirs('card_vids')
            if not os.path.exists('card_vids/' + filename):
                os.makedirs('card_vids/' + filename)
                for i in timestamps:
                    subclip = clip.subclip(i - 30, i + 30)
                    subclip.write_videofile('./card_vids/' + filename + '/yellow_sec_' + str(i) + '.mp4')
        except OSError:
            print('Error: OSError creating dir')

