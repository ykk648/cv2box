# -- coding: utf-8 --
# @Time : 2021/12/29
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import os
import re
import logging
import cv2
import shutil
from tqdm import tqdm
from ..utils import os_call
import numpy as np
from pathlib import Path


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


class CVVideo:
    def __init__(self, video_p, verbose=True):
        self.video_path = video_p
        assert Path(self.video_path).exists()
        self.video_dir, self.video_name = os.path.split(self.video_path)
        self.prefix, self.suffix = os.path.splitext(self.video_name)
        if verbose:
            self.print_video_info()

    def print_video_info(self):

        # 获得视频的格式
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        # 获得码率及尺寸
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频文件的帧数
        duration = frame_number / fps  # 帧速率/视频总帧数 是s

        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        print(
            'video info:\nname: {}\nfourcc: {}\nfps: {}\nsize: {}'.format(self.video_name, decode_fourcc(fourcc), fps,
                                                                          size))

    def show_video_cv(self, delay=100):
        cap = cv2.VideoCapture(self.video_path)
        success = True
        while success:
            success, frame = cap.read()
            cv2.namedWindow("First Frame", 0)
            cv2.imshow('First Frame', frame)
            # cv2.waitKey(99999)
            if cv2.waitKey(delay) == 27 or 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def video_2_h264(self, inplace=True):
        if not inplace:
            os_call('ffmpeg -i {} -vcodec h264 {}'.format(self.video_path,
                                                          self.video_path.replace(self.suffix, '_h264_out.mp4')))
        else:
            temp_p = self.video_path.replace(self.suffix, '_temp_copy.mp4')
            video_path_new = self.video_path.replace(self.suffix, '.mp4')
            os_call(
                'mv {} {} && ffmpeg -i {} -vcodec h264 {} && rm {}'.format(self.video_path, temp_p, temp_p,
                                                                           video_path_new, temp_p))

    def change_video_speed(self, speed=1):
        assert 0.5 <= speed <= 2.0, 'Speed must between 0.5-2.0 .'
        command = 'ffmpeg -i {} -filter_complex "[0:v]setpts={}*PTS[v];[0:a]atempo={}[a]" -map "[v]" -map "[a]" {}.mp4'.format(
            self.video_path, 1 / speed, speed, self.video_dir + '/' + self.prefix + '_speed_out')
        os_call(command)

    @staticmethod
    def concat_multi_video(video_dir):
        file_list = []
        for video_n in os.listdir(video_dir):
            file_list.append('file \'{}\'\n'.format(video_n))
        filelist_p = video_dir + '/filelist.txt'
        with open(filelist_p, 'w') as f:
            f.writelines(file_list)
        command = 'ffmpeg -f concat -i {} -c copy {}.mp4 && rm {}'.format(filelist_p,
                                                                          video_dir + '/' + 'multi_video_concat_result',
                                                                          filelist_p)
        os_call(command)

    def crop_video(self, rec_list: tuple, format='libx264'):
        """
        out_w is the width of the output rectangle
        out_h is the height of the output rectangle
        x and y specify the top left corner of the output rectangle
        """
        x, y, out_w, out_h = rec_list
        size_str = '{}:{}:{}:{}'.format(str(out_w), str(out_h), str(x), str(y))
        command = 'ffmpeg -y -i {} -filter:v "crop={}" -c:v {} -crf 17 -c:a copy {}.mp4'.format(self.video_path,
                                                                                                size_str, format,
                                                                                                self.video_dir + '/' + self.prefix + '_out')
        os_call(command)

    def cut_video(self, start, last_time, accurate=False):
        assert re.match(r"(\d{1,2}:\d{1,2}:\d{1,2})",
                        start) is not None, 'The time format: start:00:00:15 last_time:00:00:15 etc.'
        assert re.match(r"(\d{1,2}:\d{1,2}:\d{1,2})",
                        last_time) is not None, 'The time format: start:00:00:15 last_time:00:00:15 etc.'
        if not accurate:
            command = 'ffmpeg -y -ss {} -t {} -i {} -codec copy {}.mp4'.format(start, last_time, self.video_path,
                                                                               self.video_dir + '/' + self.prefix + '_cut_out')
        else:
            command = 'ffmpeg -y -ss {} -t {} -i {} {}.mp4'.format(start, last_time, self.video_path,
                                                                   self.video_dir + '/' + self.prefix + '_cut_out')
        os_call(command)

    def add_text(self, text, left_top_coord: tuple, fontsize=20):

        command = 'ffmpeg -i {} -vf drawtext="text={}:x={}:y={}:fontsize={}:fontcolor=white:box=1:boxcolor=blue" -y {}.mp4'.format(
            self.video_path, text, left_top_coord[0], left_top_coord[1], fontsize,
            self.video_dir + '/' + self.prefix + '_add_text_out')
        os_call(command)

    def reverse_crop_video(self, vp_overlay, rec_list: tuple, format='libx264rgb'):
        """
        x and y specify the top left corner of the output rectangle
        """
        x, y = rec_list
        size_str = '{}:{}'.format(str(x), str(y))
        # libx265 libx264 libx264rgb
        # command = 'ffmpeg -y -i {} -i {} -filter_complex overlay={} -c:v {} -crf 17 -c:a copy {}.mp4'.format(
        command = 'ffmpeg -y -i {} -i {} -filter_complex overlay={} -c:v {} -c:a copy {}.mp4'.format(
            self.video_path, vp_overlay, size_str, format,
            self.video_dir + '/' + self.prefix + '_reverse_out')
        os_call(command)

    def video_2_frame(self, per_sec=None, out_path=None, silent=False, rename=False):
        """
        :param per_sec: frame extract per sec
        :param out_path:
        :param rename: out_path include file rename part
        :return:
        """
        if per_sec is None:
            cap = cv2.VideoCapture(self.video_path)
            per_sec = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        suffix = Path(self.video_path).suffix
        # print(suffix)
        if out_path is None:
            save_path = self.video_path.split(suffix)[0] + '/'
            Path.mkdir(Path(save_path), exist_ok=True)
        else:
            if not rename:
                save_path = out_path + '/'
                Path.mkdir(Path(save_path), exist_ok=True)
            else:
                save_path = out_path
                Path.mkdir(Path(save_path).parent, exist_ok=True)
        command = 'ffmpeg -i {} -r {} -q:v 2 -f image2 {}%08d.jpg'.format(self.video_path, per_sec, save_path)
        os_call(command, silent=silent)

    def video_2_frame_cv(self, interval=1, out_path=None, compress=False, verbose=True):
        suffix = Path(self.video_path).suffix
        # print(suffix)
        if out_path is None:
            save_path = self.video_path.split(suffix)[0] + '/'
        else:
            save_path = out_path + '/'

        is_exists = os.path.exists(save_path)
        if not is_exists:
            os.makedirs(save_path)
            if verbose:
                print('path of %s is build' % save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
            if verbose:
                print('path of %s already exist and rebuild' % save_path)

        # 开始读视频
        video_capture = cv2.VideoCapture(self.video_path)
        i = 0
        j = 0

        while True:
            success, frame = video_capture.read()
            i += 1
            # print(frame)
            if not success:
                if verbose:
                    print('done!')
                break
            if i % interval == 0:
                # 保存图片
                j += 1
                if compress:
                    save_name = save_path + str(j) + '_' + str(i) + '.jpg'
                else:
                    save_name = save_path + str(j) + '_' + str(i) + '.png'
                cv2.imwrite(save_name, frame)
                if verbose:
                    print('image of %s is saved' % save_name)
        video_capture.release()

    def resize_video(self, out_size=(768, 1024), inplace=False):

        out_p = self.video_path.replace('.mp4', '_{}x{}.mp4'.format(out_size[0], out_size[1]))
        if inplace:
            os.system('mv {} {}'.format(self.video_path, out_p))
            self.video_path, out_p = out_p, self.video_path

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        video_writer = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)

        while True:
            s, img1 = cap.read()
            if not s:
                break
            img = cv2.resize(img1, out_size, cv2.INTER_LINEAR)
            video_writer.write(img)
        cap.release()

        if inplace:
            os.system('rm {} &'.format(self.video_path))

    def video_concat(self, video_path_2, concat_mode=None, copy_audio=True):
        assert concat_mode in ['vstack', 'hstack'], 'Need name concat_mode to \'vstack\' or \'hstack\' !'
        img = None
        reader1 = cv2.VideoCapture(self.video_path)
        fps = reader1.get(cv2.CAP_PROP_FPS)
        video1_pre_path, video1_suffix = os.path.splitext(self.video_path)
        video_out_p = video1_pre_path + '_concat_out.mp4'
        if video1_suffix != '.mp4':
            print('Will output mp4 format video file !')
            print('Output path is {}'.format(video_out_p))

        reader2 = cv2.VideoCapture(video_path_2)
        width1 = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2 = int(reader2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(reader2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if concat_mode == 'hstack':
            assert height1 == height2, 'height1: {} height2: {}'.format(height1, height2)
            output_size = (width1 + width2, height1)
        elif concat_mode == 'vstack':
            assert width1 == width2
            output_size = (width1, height1 + height2)
        else:
            output_size = (width1 + width2, height1)

        writer = cv2.VideoWriter(video_out_p,
                                 cv2.VideoWriter_fourcc(*'mp4v'),  # (*"mp4v") for mp4 output
                                 fps,  # fps
                                 output_size)  # resolution

        if not reader1.isOpened() or not reader2.isOpened():
            print('video1 read {}, video2 read {}'.format(reader1.isOpened(), reader2.isOpened()))
            return

        have_more_frame = True

        while have_more_frame:
            have_more_frame, frame1 = reader1.read()
            have_more_frame_2, frame2 = reader2.read()
            if not have_more_frame_2:
                break
            # frame1 = cv2.resize(frame1, (width, height))
            # frame2 = cv2.resize(frame2, (width, height))
            try:
                if concat_mode == 'hstack':
                    img = np.hstack((frame1, frame2))
                elif concat_mode == 'vstack':
                    img = np.vstack((frame1, frame2))
                else:
                    img = np.hstack((frame1, frame2))
            except ValueError:
                print('Got ValueError')
            writer.write(img)

        writer.release()
        reader1.release()
        reader2.release()

        if copy_audio:
            os_call('ffmpeg -i {} -vn -codec copy {}'.format(self.video_path, './temp.m4a'))
            os_call("ffmpeg -i {} -i {} -vcodec copy -acodec copy {}".format(video_out_p, './temp.m4a',
                                                                             video_out_p.replace('_concat_out.mp4',
                                                                                                 '_concat_out_audio.mp4')))
            os_call('rm ./temp.m4a')

        return video_out_p


class CVVideoLoader(object, ):
    """
    based on OpenCV
    """

    def __init__(self, video_p):
        self.video_p = video_p

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_p)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frames_num = self.cap.get(7)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get(self):
        """

        Returns: success, frame

        """
        return self.cap.read()


class CVVideoLoaderFF(object, ):
    """
    based on https://github.com/abhiTronix/deffcode, hope faster
    """

    def __init__(self, video_p):
        self.video_p = video_p

    def __enter__(self):
        try:
            from deffcode import FFdecoder
        except Exception as e:
            logger = logging.getLogger('cv2box')
            logger.error('got exception: {}, {}'.format(e, 'cv_video: pip install deffcode'))

        self.decoder = FFdecoder(self.video_p).formulate()
        self.fps = self.decoder.metadata["source_video_framerate"]
        self.size = self.decoder.metadata["source_video_resolution"]
        self.frames_num = self.decoder.metadata["approx_video_nframes"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.decoder.terminate()

    def __len__(self):
        return int(self.decoder.metadata["approx_video_nframes"])

    def get(self):
        """
        Returns: success, frame
        """
        return self.decoder.generateFrame()


class CVVideoMaker(object, ):
    @staticmethod
    def frame_2_video(frame_path_name, frame_rate=30, output_video_path=None):
        """
        :param frame_path_name:  .../lb_%d_graphormer_pred.jpg
        :param frame_rate:
        :param output_video_path:
        :return:
        """
        if not output_video_path:
            output_video_path = str(Path(frame_path_name).parent / 'output.mp4')
        os_call(
            'ffmpeg -framerate {} -i {} -c:a copy -shortest -c:v libx264 -pix_fmt yuv420p {}'.format(
                # 'ffmpeg -f image2 -i {} -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,fps={},format=yuv420p" {}'.format(
                frame_rate, frame_path_name, output_video_path))


'''
ffmpeg -y -threads 4 -start_number 0 -r 30 -i vis_results/dancecut_pare.mp4_output_temp/%06d.png -frames:v 900 -profile:v baseline -level 3.0 -c:v libx264 -pix_fmt yuv420p -an -v error -loglevel error vis_results/dancecut_pare.mp4
'''
