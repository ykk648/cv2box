# -- coding: utf-8 --
# @LastEdit : 2024/11/23
# @Author : ykk648
"""
ref https://github.com/OpenTalker/SadTalker/blob/main/src/utils/audio.py
"""
import re
from pathlib import Path
from ..utils import try_import, os_call


sd = try_import('sounddevice', 'cv_audio: need sounddevice')
torchaudio = try_import("torchaudio", 'cv_audio: need torchaudio')


class CVAudio:
    def __init__(self, ffmpeg_path='ffmpeg'):
        self.ffmpeg_path = ffmpeg_path

    def get_duration(self, audio_path):
        """

        :return: seconds
        """
        result = os_call(f"{self.ffmpeg_path} -i {audio_path}", silent=False, asyncio=False)
        # print(result)
        # Use regular expression to extract duration
        duration_str = re.search(r"Duration:\s{1}(.*?),", result, re.DOTALL).group(1)
        # Convert duration string to seconds
        duration = sum(float(x) * 60 ** i for i, x in enumerate(reversed(duration_str.split(":"))))
        return duration

    @staticmethod
    def load_audio_torch(audio_path, target_sr):
        speech, sample_rate = torchaudio.load(audio_path)
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech

    def crop_audio(self, audio_path, start, end, output_p=None):
        if not output_p:
            output_p = Path(audio_path).parent / (Path(audio_path).stem + "_cut" + Path(audio_path).suffix)
        os_call(f'{self.ffmpeg_path} -y -i {audio_path} -ss {start} -to {end} -c copy {output_p}')

    def speedup(self, audio_path, speed_up_x, output_p=None):
        if not output_p:
            output_p = Path(audio_path).parent / (Path(audio_path).stem + "_speedup" + Path(audio_path).suffix)
        os_call(f"{self.ffmpeg_path} -y -i {audio_path} -filter:a atempo={speed_up_x} -vn {output_p}")

    def make_silence(self, silence_duration, output_p):
        os_call(f"{self.ffmpeg_path} -y -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -t {silence_duration} -vn {output_p}")

    def concat_2audio(self, audio_path_1, audio_path_2, output_p=None):
        if not output_p:
            output_p = Path(audio_path_1).parent / (Path(audio_path_1).stem + "_concat" + Path(audio_path_1).suffix)
        os_call(f"{self.ffmpeg_path} -y -i {audio_path_1} -i {audio_path_2} -filter_complex \"[0:0][1:0]concat=n=2:v=0:a=1[out]\" -map \"[out]\" {output_p}")

    def play_audio(self, audio_path, target_device_name=None):
        sf = try_import('soundfile', 'cv_audio: need soundfile')
        # 读取音频文件
        data, samplerate = sf.read(audio_path)

        # 获取所有可用的音频设备列表
        devices = sd.query_devices()

        # 打印每个音频设备的信息和ID
        for idx, device in enumerate(devices):
            print(f"Device {idx}: {device['name']}")

        device_id = None
        for device in devices:
            if device['name'] == target_device_name:
                device_id = device['index']
                break

        if device_id is not None:
            # 播放音频到指定设备
            sd.play(data, samplerate, device=device_id)
            sd.wait()  # 等待音频播放完毕
        else:
            print(f"Device '{target_device_name}' not found")
