# -- coding: utf-8 --
# @Time : 2021/12/29
# @LastEdit : 2025/4/25
# @Author : ykk648

import os
import re
import json
import cv2
import subprocess
import shutil
from ..utils import os_call, try_import, check_ffmpeg
from .cv_image import CVImage
import numpy as np
from pathlib import Path
from ..utils import cv_print as print


class CVVideo:
    def __init__(self, video_p, verbose=True, ffmpeg_path='ffmpeg'):
        self.video_path = video_p
        self.ffmpeg_path = ffmpeg_path
        assert check_ffmpeg(self.ffmpeg_path), 'ffmpeg is not installed.'

        video_path_posix = Path(self.video_path)
        assert video_path_posix.exists(), 'video path does not exist.'
        self.video_dir = video_path_posix.parent
        self.video_name = video_path_posix.name
        self.prefix = video_path_posix.stem
        self.suffix = video_path_posix.suffix
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

        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        def decode_fourcc(cc):
            return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

        print(f'video info:\nname: {self.video_name}\nfourcc: {decode_fourcc(fourcc)}\nframe_number: {frame_number}\nfps: {fps}\nduration: {duration}\nsize: {size}')

    def show_video_cv(self, delay=99999):
        with CVVideoLoader(self.video_path) as cvvl:
            for _ in range(len(cvvl)):
                success, frame = cvvl.get()
                CVImage(frame).show(delay)

    def get_video_info_ffmpeg(self):
        # 使用 ffmpeg 获取视频信息并直接输出 JSON
        ffmpeg_command = [
            self.ffmpeg_path,
            "-i", self.video_path,  # 输入文件
            "-f", "null",  # 输出格式为空（不生成输出文件）
            "/dev/null"  # 输出到空设备
        ]

        # 通过 subprocess 调用 ffmpeg 并捕获输出（不写入文件）
        process = subprocess.Popen(ffmpeg_command, stderr=subprocess.PIPE)
        out, err = process.communicate()

        # 检查 ffmpeg 的执行是否出错
        if process.returncode != 0:
            print(f"Error: {err.decode('utf-8')}")
            return None

        # 从 stderr 中提取视频信息
        info = err.decode('utf-8')

        # 正则表达式提取所需信息
        width_height = re.search(r'(\d+)x(\d+)', info)
        fps = re.search(r'(\d+(?:\.\d+)?) fps', info)
        pix_fmt = re.search(r'(?<=pix_fmt:\s)(\S+)', info)
        color_space = re.search(r'(?<=color_space:\s)(\S+)', info)
        color_primaries = re.search(r'(?<=color_primaries:\s)(\S+)', info)
        color_transfer = re.search(r'(?<=color_transfer:\s)(\S+)', info)

        # 提取信息
        if width_height:
            width, height = map(int, width_height.groups())
        else:
            width, height = None, None

        fps_value = eval(fps.group(1)) if fps else 25
        pix_fmt_value = pix_fmt.group(0) if pix_fmt else 'bgr24'
        color_space_value = color_space.group(0) if color_space else "bt709"
        color_primaries_value = color_primaries.group(0) if color_primaries else "bt709"
        color_transfer_value = color_transfer.group(0) if color_transfer else "bt709"

        # 处理特定的像素格式情况
        if pix_fmt_value == "yuv420p10le":
            pix_fmt_value = "yuv420p"
            color_space_value = "bt709"
            color_primaries_value = "bt709"
            color_transfer_value = "bt709"

        video_info = {
            "fps": fps_value,
            "height": height,
            "width": width,
            "pix_fmt": pix_fmt_value,
            "color_space": color_space_value,
            "color_primaries": color_primaries_value,
            "color_transfer": color_transfer_value
        }
        return video_info

    def video_2_h264(self, inplace=True):
        # ffmpeg -i input.mp4 -c:v libx264 -tag:v avc1 -movflags faststart -crf 30 -preset superfast output.mp4
        # https://tools.rotato.app/compress
        if not inplace:
            os_call('ffmpeg -i {} -vcodec h264 {}'.format(self.video_path, self.video_path.replace(self.suffix, '_h264_out.mp4')))
        else:
            temp_p = self.video_path.replace(self.suffix, '_temp_copy.mp4')
            video_path_new = self.video_path.replace(self.suffix, '.mp4')
            os_call('mv {} {} && ffmpeg -i {} -vcodec h264 {} && rm {}'.format(self.video_path, temp_p, temp_p, video_path_new, temp_p))

    def change_video_speed(self, speed=1):
        assert 0.5 <= speed <= 2.0, 'Speed must between 0.5-2.0 .'
        command = 'ffmpeg -i {} -filter_complex "[0:v]setpts={}*PTS[v];[0:a]atempo={}[a]" -map "[v]" -map "[a]" {}.mp4'.format(
            self.video_path, 1 / speed, speed, self.video_dir + '/' + self.prefix + '_speed_out')
        os_call(command)

    @staticmethod
    def concat_multi_video_from_dir(video_dir):
        file_list = []
        for video_n in os.listdir(video_dir):
            file_list.append('file \'{}\'\n'.format(video_n))
        filelist_p = video_dir + '/filelist.txt'
        with open(filelist_p, 'w') as f:
            f.writelines(file_list)
        command = 'ffmpeg -f concat -i {} -c copy {}.mp4 && rm {}'.format(filelist_p, video_dir + '/' + 'multi_video_concat_result', filelist_p)
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
                                                                                                self.video_dir / f"{self.prefix}_out")
        os_call(command)

    def cut_video(self, start, last_time, accurate=False):
        """
        :param start: start time(s)
        :param last_time: video clip length(s)
        :param accurate: using keyframe or not
        :return:
        """
        assert re.match(r"(\d{1,2}:\d{1,2}:\d{1,2})",
                        start) is not None, 'The time format: start:00:00:15 last_time:00:00:15 etc.'
        assert re.match(r"(\d{1,2}:\d{1,2}:\d{1,2})",
                        last_time) is not None, 'The time format: start:00:00:15 last_time:00:00:15 etc.'
        cut_out_video_path = str(self.video_dir) + '/' + self.prefix + '_cut_out.mp4'
        if not accurate:
            command = 'ffmpeg -y -ss {} -t {} -i "{}" -codec copy "{}"'.format(start, last_time, self.video_path,
                                                                               cut_out_video_path)
        else:
            command = 'ffmpeg -y -ss {} -t {} -i "{}" "{}"'.format(start, last_time, self.video_path, cut_out_video_path)
        os_call(command)
        return cut_out_video_path

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
            save_path = self.video_path.split(suffix)[0]
        else:
            save_path = out_path

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
                # 支持中文
                j += 1
                if compress:
                    save_name = str(Path(save_path) / f"{j}_{i}.jpg")
                    cv2.imencode(ext='.jpg', img=frame)[1].tofile(save_name)
                else:
                    save_name = str(Path(save_path) / f"{j}_{i}.png")
                    cv2.imencode(ext='.png', img=frame)[1].tofile(save_name)
                # cv2.imwrite(save_name, frame)
                if verbose:
                    print('image of %s is saved' % save_name)
        video_capture.release()

    def resize_video(self, out_size=(768, 1024), inplace=False):
        out_p = self.video_path.replace('.mp4', '_{}x{}.mp4'.format(out_size[0], out_size[1]))
        if inplace:
            shutil.move(self.video_path, out_p)
            self.video_path, out_p = out_p, self.video_path
            # os.remove(self.video_path)
        os_call(
            f'ffmpeg -y -loglevel error -i {self.video_path} -s {out_size[0]}x{out_size[1]} -c:a copy {out_p}')
        if inplace:
            os.remove(self.video_path)

    def resize_video_cv(self, out_size=(768, 1024), inplace=False):
        out_p = self.video_path.replace('.mp4', '_{}x{}.mp4'.format(out_size[0], out_size[1]))
        if inplace:
            shutil.move(self.video_path, out_p)
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
            os.remove(self.video_path)

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
            os_call('ffmpeg -y -i "{}" -vn -codec copy "{}"'.format(self.video_path, './temp.m4a'))
            os_call(' ffmpeg -y -i "{}" -i {} -vcodec copy -acodec copy "{}"'.format(video_out_p, './temp.m4a',
                                                                             video_out_p.replace('_concat_out.mp4',
                                                                                                 '_concat_out_audio.mp4')))
            Path("./temp.m4a").unlink()
            Path(video_out_p).unlink()
        return video_out_p

    def extract_audio(self, output_path=None, quiet=False, sample_rate=44100):
        if not output_path:
            output_path = Path(self.video_path).with_suffix('.wav')

        if quiet:
            ffmpeg_params = '-loglevel error'
        else:
            ffmpeg_params = '-loglevel info'
        ffmpeg_log = os_call(f'{self.ffmpeg_path} -y -i {self.video_path} -vn -acodec pcm_s16le -ar {sample_rate} -ac 1 {ffmpeg_params} {output_path}')
        return ffmpeg_log

    def copy_audio(self, audio_src):
        """
        support mp4 or wav
        ffmpeg -y -hide_banner -i "input.mp4" -i "input1.wav" -i "input2.wav" -filter_complex "[1:a]volume=1.5[a1];[2:a]volume=1[a2];[a1][a2]amix=inputs=2:duration=first:dropout_transition=0" -c:v "libx264" -c:a 'aac' -y "mix.mp4"
        :param audio_src:
        :return:
        """
        # mod to mkv when met "Could not find tag for codec pcm_s24le in stream #1"
        # output_path = self.video_path.replace('.mp4', '_audio_replaced.mkv')
        output_path = self.video_path.replace('.mp4', '_audio_replaced.mp4')
        if Path(audio_src).suffix == '.mp4':
            audio_temp_path = audio_src.replace('.mp4', '_temp.mp4')
            os_call(f'ffmpeg -i {audio_src} -vn -codec copy {audio_temp_path}')
            os_call(
                f'ffmpeg -i {self.video_path} -i {audio_temp_path} -vcodec copy -acodec copy -map 0:v:0 -map 1:a:0 {output_path}')
            os.remove(audio_temp_path)
        else:
            os_call(
                f'ffmpeg -i {self.video_path} -i {audio_src} -c copy -map 0:v -map 1:a -shortest {output_path}')

    def bt2020_2_bt709(self):
        """
        ref https://www.bilibili.com/read/cv3936575/
        :return:
        """
        output_path = self.video_path.replace('.mp4', '_bt709.mp4')
        ffmpeg_command = f"{self.ffmpeg_path} -i {self.video_path} -vf zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p -c:v libx264 -preset slow -crf 18 -c:a copy {output_path}"
        os_call(ffmpeg_command)


class CVVideoLoader(object):
    """
    based on OpenCV
    """

    def __init__(self, video_p):
        self.video_p = video_p

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_p)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frames_num = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        codec = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.codec = chr(codec & 0xFF) + chr((codec >> 8) & 0xFF) + chr((codec >> 16) & 0xFF) + chr((codec >> 24) & 0xFF)
        return self

    def reset(self):
        """重置视频到起始位置
        Returns:
            bool: 是否重置成功
        """
        if self.cap is None:
            return False
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get(self):
        """获取下一帧
        Returns:
            tuple: (success, frame)
        """
        return self.cap.read()

    def __len__(self):
        return int(self.frames_num)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()


class CVVideoLoaderVidgear(object):
    """
    based on Vidgear https://github.com/abhiTronix/vidgear
    internal queue to save opencv frame, similar as CVVideoThread
    """

    def __init__(self, video_p):
        self.video_p = video_p

    def __enter__(self):
        vidgear = try_import('vidgear')
        from vidgear.gears import VideoGear
        self.stream = VideoGear(source=self.video_p).start()
        self.fps = self.stream.framerate
        self.size = (int(self.stream.stream.frame.shape[1]), int(self.stream.stream.frame.shape[0]))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.stop()

    def __len__(self):
        return int(self.frames_num)

    def get(self):
        """
        Returns: success, frame
        """
        return self.stream.read()


class CVVideoLoaderFF(object, ):
    """
    pip install deffcode
    based on https://github.com/abhiTronix/deffcode, hope faster
    default ffmpeg 6.1.1 ref https://abhitronix.github.io/deffcode/latest/installation/ffmpeg_install/#a-auto-installation
    """

    def __init__(self, video_p, custom_ffmpeg=None):
        self.video_p = video_p
        self.custom_ffmpeg = custom_ffmpeg

    def __enter__(self):
        deffcode = try_import('deffcode')

        self.decoder = deffcode.FFdecoder(self.video_p, frame_format="bgr24", custom_ffmpeg=self.custom_ffmpeg).formulate()
        self.metadata = json.loads(self.decoder.metadata)
        self.fps = self.metadata["source_video_framerate"]
        self.size = self.metadata["source_video_resolution"]
        self.frames_num = self.metadata["approx_video_nframes"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.decoder.terminate()

    def __len__(self):
        return int(self.frames_num)

    def get(self):
        """
        Returns: success, frame
        """
        return None, next(self.decoder.generateFrame())


class CVVideoLoaderFFHWACCL(object, ):
    """
    pip install deffcode
    based on https://github.com/abhiTronix/deffcode, hope faster
    default ffmpeg 6.1.1 ref https://abhitronix.github.io/deffcode/latest/installation/ffmpeg_install/#a-auto-installation
    ref https://abhitronix.github.io/deffcode/latest/recipes/advanced/transcode-hw-acceleration/
    """

    def __init__(self, video_p, custom_ffmpeg=None):
        self.video_p = video_p
        self.custom_ffmpeg = custom_ffmpeg

        self.ffparams = {
            "-vcodec": None,  # skip source decoder and let FFmpeg chose
            "-enforce_cv_patch": True,  # enable OpenCV patch for YUV(NV12) frames
            "-ffprefixes": [
                "-vsync",
                "0",  # prevent duplicate frames
                "-hwaccel",
                "cuda",  # accelerator
                "-hwaccel_output_format",
                "cuda",  # output accelerator
            ],
            "-custom_resolution": "null",  # discard source `-custom_resolution`
            "-framerate": "null",  # discard source `-framerate`
            "-vf": "scale_cuda=640:360,"  # scale to 640x360 in GPU memory
                   + "crop=80:60:200:100,"  # crop a 80×60 section from position (200, 100) in GPU memory
                   + "hwdownload,"  # download hardware frames to system memory
                   + "format=nv12",  # convert downloaded frames to NV12 pixel format
        }

    def __enter__(self):
        deffcode = try_import('deffcode')

        self.decoder = deffcode.FFdecoder(self.video_p, frame_format="null", verbose=False, custom_ffmpeg=self.custom_ffmpeg, **self.ffparams).formulate()
        self.metadata = json.loads(self.decoder.metadata)
        self.fps = self.metadata["source_video_framerate"]
        self.size = self.metadata["source_video_resolution"]
        self.frames_num = self.metadata["approx_video_nframes"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.decoder.terminate()

    def __len__(self):
        return int(self.frames_num)

    def get(self):
        """
        Returns: success, frame
        """
        return None, cv2.cvtColor(next(self.decoder.generateFrame()), cv2.COLOR_YUV2BGR_NV12)


class CVVideoLoaderAV(object, ):
    """
    pip install av
    based on https://github.com/PyAV-Org/PyAV, without hwaccel support, pure cpu, random lost frames
    """

    def __init__(self, video_p):
        self.video_p = video_p

    def __enter__(self):
        av = try_import('av')

        self.container = av.open(self.video_p)
        self.video_stream = self.container.streams.video[0]
        self.video_stream.thread_type = "AUTO"  # FRAME AUTO
        self.fps = self.video_stream.base_rate  # 帧率
        self.size = (self.video_stream.height, self.video_stream.width)
        self.frames_num = self.video_stream.frames  # 视频总帧数
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __len__(self):
        return int(self.frames_num)

    def get(self):
        """
        Returns: frame
        """
        return None, next(self.container.decode(self.video_stream)).to_ndarray(format='bgr24')


class CVVideoLoaderTorch(object):
    """
    Video loader based on torchcodec
    """

    def __init__(self, video_p):
        self.video_p = video_p

    def __enter__(self):
        VideoDecoder = try_import('torchcodec.decoders')
        self.decoder = VideoDecoder(self.video_p)
        # Get video metadata
        self.frames_num = len(self.decoder)
        self.fps = self.decoder.fps
        self.size = (self.decoder.width, self.decoder.height)
        # Note: torchcodec doesn't expose codec info directly
        self.codec = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # VideoDecoder will handle cleanup automatically
        pass

    def __len__(self):
        return self.frames_num

    def get(self):
        """
        Get next frame from video
        Returns: success (bool), frame (torch.Tensor)
        """
        try:
            # Get next frame - note this is simplified as torchcodec doesn't have
            # a direct frame-by-frame reading mechanism like OpenCV
            frame = self.decoder[self.current_frame:self.current_frame + 1]
            self.current_frame += 1
            return True, frame.squeeze(0)  # Remove batch dimension
        except IndexError:
            return False, None

    def get_frames_at(self, indices):
        """
        Get frames at specific indices
        Args:
            indices: List of frame indices to retrieve
        Returns: torch.Tensor of frames
        """
        return self.decoder.get_frames_at(indices=indices)

    def get_frames_by_time(self, seconds):
        """
        Get frames at specific timestamps
        Args:
            seconds: List of timestamps in seconds
        Returns: torch.Tensor of frames
        """
        return self.decoder.get_frames_played_at(seconds=seconds)


class CVVideoMaker(object):
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
