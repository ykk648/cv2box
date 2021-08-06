import os
import cv2
import shutil
from tqdm import tqdm
from .utils import oscall
import numpy as np


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


class VideoTools:
    def __init__(self, video_p, verbose=True):
        self.video_path = video_p
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
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        print(
            'video info:\nname: {}\nfourcc: {}\nfps: {}\nsize: {}'.format(self.video_name, decode_fourcc(fourcc), fps,
                                                                          size))

    def video_2_h264(self, inplace=True):
        # ffmpeg -i "xx.avi" -vcodec h264 "xx.mp4"
        if not inplace:
            oscall('ffmpeg -i {} -vcodec h264 {}'.format(self.video_path,
                                                         self.video_path.replace(self.suffix, '_h264_out.mp4')))
        else:
            # print(self.extension)
            temp_p = self.video_path.replace(self.suffix, '_temp_copy.mp4')
            video_path_new = self.video_path.replace(self.suffix, '.mp4')
            oscall(
                'mv {} {} && ffmpeg -i {} -vcodec h264 {} && rm {}'.format(self.video_path, temp_p, temp_p,
                                                                           video_path_new, temp_p))
            # oscall()

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
        oscall(command)

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
        oscall(command)

    def show_video_cv(self):
        cap = cv2.VideoCapture(self.video_path)
        success = True
        while success:
            success, frame = cap.read()
            cv2.namedWindow("First Frame", 0)
            cv2.imshow('First Frame', frame)
            # cv2.waitKey(99999)
            if cv2.waitKey(100) == 27 or 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def video_2_frame(self, interval=1, out_path=None):
        if out_path is None:
            save_path = self.video_path.split('.mp4')[0] + '/'
        else:
            save_path = out_path

        is_exists = os.path.exists(save_path)
        if not is_exists:
            os.makedirs(save_path)
            print('path of %s is build' % save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
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
                print('done!')
                break
            if i % interval == 0:
                # 保存图片
                j += 1
                save_name = save_path + str(j) + '_' + str(i) + '.png'
                cv2.imwrite(save_name, frame)
                print('image of %s is saved' % save_name)
        video_capture.release()

    def resize_video(self, inplace=False, out_size=(768, 1024)):

        out_p = self.video_path.replace(self.extension, '_{}x{}.mp4'.format(out_size[0], out_size[1]))
        if inplace:
            os.system('mv {} {}'.format(self.video_path, out_p))
            self.video_path, out_p = out_p, self.video_path

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        video_writer = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)

        s = True
        while s:
            s, img1 = cap.read()
            img = cv2.resize(img1, out_size, cv2.INTER_LINEAR)
            video_writer.write(img)
        cap.release()

        if inplace:
            os.system('rm {} &'.format(self.video_path))

    # def avi2mp4(self):
    #
    #     # 指定写视频的格式, I420-avi, MJPG-mp4
    #     videoWriter = cv2.VideoWriter(self.root_path.split('.')[0] + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
    #                                   fps, size)
    #
    #     # 读帧
    #     success, frame = cap.read()
    #
    #     while success:
    #         # cv2.waitKey(1000 / int(fps))  # 延迟
    #         videoWriter.write(frame)  # 写视频帧
    #         success, frame = cap.read()  # 获取下一帧

    def video_concat(self, video_path_2, concat_mode=None):
        if concat_mode is None:
            print('Need name concat_mode to \'vstack\' or \'hstack\' !')
            return

        reader1 = cv2.VideoCapture(self.video_path)
        video1_pre_path, video1_suffix = os.path.splitext(self.video_path)
        video_out_p = video1_pre_path + '_out.mp4'
        if video1_suffix != '.mp4':
            print('Will output mp4 format video file !')
            print('Output path is {}'.format(video_out_p))

        reader2 = cv2.VideoCapture(video_path_2)
        width = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(video_out_p,
                                 cv2.VideoWriter_fourcc(*'mp4v'),  # (*"mp4v") for mp4 output
                                 30,  # fps
                                 (width * 2, height))  # resolution

        if not reader1.isOpened() or not reader2.isOpened():
            print('video1 read {}, video2 read {}'.format(reader1.isOpened(), reader2.isOpened()))
            return

        have_more_frame = True

        while have_more_frame:
            have_more_frame, frame1 = reader1.read()
            have_more_frame_2, frame2 = reader2.read()
            if not have_more_frame_2:
                break
            frame1 = cv2.resize(frame1, (width, height))
            frame2 = cv2.resize(frame2, (width, height))
            if concat_mode == 'hstack':
                img = np.hstack((frame1, frame2))
            elif concat_mode == 'vstack':
                img = np.vstack((frame1, frame2))
            else:
                img = np.hstack((frame1, frame2))
            writer.write(img)

        writer.release()
        reader1.release()
        reader2.release()
