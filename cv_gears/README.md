### CV Threads Base

Human-like API , making multi-thread AI program more easy.

Multithread waterfall sequence: Factory -> Linker -> Consumer

If you need circle or tree structure, just add more queue to queue_list.

#### multi-mode switch

Make sure your model is pickable.

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

q1 = Queue(5)
c1 = CVVideoThread(video_p, [q1], silent=False)
c1.start()
```

### CV Multi Video Thread

Supply multiple video/stream read synchronous based on vidgear.

```python
from cv2box.cv_gears import CVMultiVideoThread
source_list = [0, 2, 4, 6]
q1 = Queue(5)
cvmt = CVMultiVideoThread(source_list, [q1])
cvmt.start()
```