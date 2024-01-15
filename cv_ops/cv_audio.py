# -- coding: utf-8 --
# @Time : 2023/7/24
# @Author : ykk648
# @Project : https://github.com/ykk648/cv_toolbox
"""
ref https://github.com/OpenTalker/SadTalker/blob/main/src/utils/audio.py
"""
from ..utils import try_import
from ..utils.audio.hparams import hparams as hp
import numpy as np
from scipy import signal
from tqdm import tqdm

librosa = try_import('librosa', 'cv_audio: need librosa')


def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)


_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def _lws_processor():
    lws = try_import('lws', 'cv_audio: need lws')
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")


def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)


def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)

    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))


def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)

    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


class CVAudio:
    def __init__(self, audio_path, sr=16000, mono=True):
        self.audio_path = audio_path
        self.sr = sr
        self.num_frames = None
        self.fps = None
        if audio_path:
            self.audio = librosa.core.load(self.audio_path, sr=sr, mono=mono)[0]
            """
            # audio_all, sr = sf.read(audio_path, dtype='int16')
            # if len(audio_all.shape) > 1:
            #     audio_all = 0.5 * (audio_all[:, 0] + audio_all[:, 1])
            # audio_all = resampy.resample(audio_all, sr, 16000)
            """

    @property
    def data(self):
        return self.audio

    def mel_spectrogram(self, crop_pad=False, fps=None):
        """
        :param crop_pad:
        :param fps:
        :return: (N,80) N 采样点个数
        """
        if crop_pad:
            assert fps
            self.fps = fps
            bit_per_frames = self.sr / self.fps
            audio_length = len(self.audio)
            self.num_frames = int(audio_length / bit_per_frames)
            cal_audio_length = int(self.num_frames * bit_per_frames)

            if audio_length > cal_audio_length:
                self.audio = self.audio[:cal_audio_length]
            elif audio_length < cal_audio_length:
                self.audio = np.pad(self.audio, [0, cal_audio_length - audio_length], mode='constant',
                                    constant_values=0)

        D = _stft(preemphasis(self.audio, hp.preemphasis, hp.preemphasize))
        S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db

        if hp.signal_normalization:
            return _normalize(S).T
        return S.T

    def individual_mel_spec(self, orig_mel, mel_window=16):
        """
        slide window, ref wav2lip
        :param orig_mel:
        :param mel_window:
        :return: (num_frames,80,16)
        """
        indiv_mels = []
        for i in tqdm(range(self.num_frames), 'mel:'):
            start_frame_num = i - 2
            start_idx = int(80. * (start_frame_num / float(self.fps)))
            end_idx = start_idx + mel_window
            seq = list(range(start_idx, end_idx))
            seq = [min(max(item, 0), orig_mel.shape[0] - 1) for item in seq]
            m = orig_mel[seq, :]
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels)
        return np.asarray(indiv_mels)

class CVAudioRecThread(Factory):
    def __init__(self, queue_list: list, input_device_name, samplerate=40000, blocksize=4096, fps_counter=False):
        super().__init__(queue_list, fps_counter)
        if input_device_name is not None:
            self.set_devices(input_device_name)
        else:
            print(self.get_devices())
            print('Input device not set, use default.')
        self.stream = sd.InputStream(device=sd.default.device[0], channels=1, blocksize=blocksize,
                                     samplerate=samplerate, dtype="float32")
        self.stream.start()
        self.blocksize = blocksize

        self.stream_out = sd.OutputStream(device=sd.default.device[1], channels=1, samplerate=samplerate)
        self.stream_out.start()

    def exit_func(self):
        """
        If something is None, enter exit func, set `pass` if you want deal with exit by yourself.
        """
        self.exit_signal = False
        # self.stream.stop()
        # self.stream.close()

    def get_devices(self, update: bool = True):
        """获取设备列表"""
        if update:
            sd._terminate()
            sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0
        ]
        return (input_devices, input_devices_indices,)

    def set_devices(self, input_device):
        """设置输出设备"""
        (
            input_devices,
            input_device_indices,
        ) = self.get_devices()
        sd.default.device[0] = input_device_indices[
            input_devices.index(input_device)
        ]

        print(f"Input device: {str(sd.default.device[0])}:{input_device}")

    def forward_func(self):
        data = self.stream.read(self.blocksize)
        self.stream_out.write(data[0].astype(np.float32))
        # print(data)
        return [data[0]]


class CVAudioPlayThread(Consumer):
    def __init__(self, queue_list: list[Queue], output_device_name=None, samplerate=40000, fps_counter=False):
        super().__init__(queue_list, fps_counter)
        if output_device_name is not None:
            self.set_devices(output_device_name)
        else:
            print(self.get_devices())
            print('Output device not set, use default.')
        self.stream = sd.OutputStream(device=sd.default.device[1], channels=1, samplerate=samplerate)
        self.stream.start()

    def get_devices(self, update: bool = True):
        """获取设备列表"""
        if update:
            sd._terminate()
            sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        output_devices = [f"{d['name']} ({d['hostapi_name']})" for d in devices if d["max_output_channels"] > 0]
        output_devices_indices = [d["index"] if "index" in d else d["name"] for d in devices if
                                  d["max_output_channels"] > 0]
        return (output_devices, output_devices_indices)

    def set_devices(self, output_device):
        """设置输出设备"""
        (
            output_devices,
            output_device_indices,
        ) = self.get_devices()
        sd.default.device[1] = output_device_indices[
            output_devices.index(output_device)
        ]
        print(f"Output device: {str(sd.default.device[1])}:{output_device}")

    def exit_func(self):
        """
        If something is None, enter exit func, set `pass` if you want deal with exit by yourself.
        """
        print('{} {} exit !'.format(self.class_name(), self.pid_number))
        self.exit_signal = True
        self.stream.stop()
        self.stream.close()

    def forward_func(self, something_in):
        output_audio = something_in[0]
        self.stream.write(output_audio.astype(np.float32))
