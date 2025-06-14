### CV Threads Base

Human-like API , making multi-thread AI program more easier.

Multithread waterfall sequence: Factory -> Linker -> Consumer

If you need circle or tree structure, just add more queue to queue_list.

Linker support pass queue list to forward_func, which is useful for multi-output.

#### multi-mode switch

Default mode is multi-thread, which is most economical.

Make sure your model is pickable if you want use torch-process.

```python
import os
os.environ['CV_MULTI_MODE'] = 'torch-process'  # multi-thread multi-process torch-process

# for multi-process or torch-process
import multiprocessing
multiprocessing.set_start_method('spawn')  # fork
```


#### inherent base class

```python
from cv2box.cv_gears import Factory, Linker, Consumer

class SecondModel(Linker):
    def __init__(self, queue_list: list, block=True, fps_counter=False):
        super().__init__(queue_list, block, fps_counter)
        # model init
        self.model = SomeNet()

    def exit_func(self):
        # rewrite exit condition
        pass

    def forward_func(self, something_in):
        if something_in is None:
            return
        in_1, in_2 = something_in
        out_1, out_2 = self.model(in_1, in_2)
        return [out_1, out_2]
```

#### init class in main func

```python
from cv2box.cv_gears import Queue

q1 = Queue(2)
q2 = Queue(2)
m1 = FirstModel([q1], fps_counter=True)
m2 = SecondModel([q1, q2], fps_counter=True)
m3 = FinalModel([q2], fps_counter=True)
m1.start()
m2.start()
m3.start()
```

### CV Video Thread

Supply thread reading to a video file, addition of VidGear.

#### example

```python
from cv2box.cv_gears import CVVideoThread, Queue

frame_q = Queue(5)
c1 = CVVideoThread(video_p, [frame_q], fps_counter=False)
c1.start()
while True:
    frame = frame_q.get()
```

### CVVideoCacheThread

WIP

### CVCamThread

WIP

### CV Video Writer Thread

Supply thread writing rgb frame to a video file, support ffmpeg and opencv.

#### example

```python
from cv2box.cv_gears import CVVideoWriterThread, Queue

frame_q = Queue(5)
if opencv_flag:
    video_writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'avc1'), video_fps, video_size)
else:
    video_info = CVVideo(video).get_video_info_ffmpeg()
    ffmpeg_command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{video_size[0]}x{video_size[1]}", "-pix_fmt", "bgr24", "-r", str(video_fps), "-i", "-", "-an", "-c:v", "h264_nvenc", "-preset", "medium",
                      "-qp", "10", "-pix_fmt", f"{video_info['pix_fmt']}", "-colorspace", f"{video_info['color_space']}", "-color_primaries", f"{video_info['color_primaries']}", "-color_trc", f"{video_info['color_transfer']}",
                      "-color_range", "tv", f"{str(video_out_path)}"]
    video_writer = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
c1 = CVVideoWriterThread(video_writer, [frame_q], fps_counter=False)
c1.start()
frame_q.put(some_frame)
c1.join()
if opencv_flag:
    video_writer.release()
else:
    stdout, stderr = video_writer.communicate()
```

### CV Multi Video Thread

Supply multiple video/stream read synchronous based on vidgear, and will reconnect when stream get fail.

```python
from cv2box.cv_gears import CVMultiVideoThread
source_list = [0, 2, 4, 6]
q1 = Queue(5)
cvmt = CVMultiVideoThread(source_list, [q1])
cvmt.start()
```