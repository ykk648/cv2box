### CVAudio

Audio preprocessing.

#### example

```python
from cv2box.cv_ops.cv_audio import CVAudio
audio_p = ''
ca = CVAudio(audio_p, sr=16000)
print(ca.data.shape)
mel = ca.mel_spectrogram(True, 25)
print(mel.shape)
individual_mel = ca.individual_mel_spec(mel)
print(individual_mel.shape)
```

### CVImage

Human-like API to unite different image format, convenient way to convert/show/save one pic or even pic path.

Makes `BGR RGB PIL cv2 sk-image bytes base64 tensor` to be one **CVImage**.

#### example

```python
from cv2box import CVImage
img_in = 'img path' # or img_in = cv2.imread('img path')
img_bgr = CVImage(img_in).bgr
img_rgb = CVImage(img_in).rgb
img_pil = CVImage(img_in).pillow
img_base64 = CVImage(img_in).base64
img_bytes = CVImage(img_in).bytes

# show image using opencv
CVImage(img_in).resize(256,256).show()

# save image using opencv
from PIL import Image
img_in_pillow = Image.open('image path')
CVImage(img_in_pillow, image_format='pillow').resize(256,256).save('image save path')

# image2tensor
trans = {}
CVImage(img_in).set_transform(trans).tensor.cuda()
# image2cvblob
onnx_input = CVImage(img_in).set_blob(127.5,127.5,(112,112)).blob_rgb

```

### CVVideo

Basic video operate process using cv2 & ffmpeg.

#### example

```python
from cv2box import CVVideo, CVImage, CVVideoLoader, CVVideoMaker
from tqdm import tqdm

# basic video ops
vp = 'test.mp4'
vt = CVVideo(vp).video_2_frame(interval=30)
CVVideo(vp).video_2_h264(inplace=False)
CVVideo(vp).cut_video('00:00:15', '00:00:35', accurate=True)

# load a video to process by cv2
with CVVideoLoader(vp) as cvvl:
    # print(len(cvvl))
    for _ in tqdm(range(len(cvvl))):
        _, frame = cvvl.get()
        CVImage(frame).show(0)

# cluster frames to one video
cvvm = CVVideoMaker().frame_2_video('frame_path/something_%d.jpg')
```


### CVQueue

A queue-like high-level class which can be used for two different python projects on same host machine.

Based on ShareMemory , which is more efficient than socket or grpc.

#### idea

![CVQueue](./src/cvqueue.png)

#### fixed problems

- Deal with memory release;
- Sync without lock;
- Multi size buffer trans;
- Add queue-like api like full/empty/close;
- Mem access delay to avoid access fail;
- More retry logic;

#### cam stream example

```python
import cv2
from cv2box import CVQueue, CVImage

# project A
cvq = CVQueue(10, mem_name='cv2box', max_data_size=3*1920*1080)
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    img_buffer = CVImage(frame).format_bytes
    cvq.push(img_buffer)
    
# project B
get_cvq = CVQueue(10, mem_name='cv2box')
while True:
    get_buf, get_buffer_len = get_cvq.get()
    image_to_show = bytes(get_buf.buf[:get_buffer_len])
    get_q.get_ok()
    frame_get = CVImage(image_to_show, image_format='buffer',image_size=(frame_height, frame_width, 3)).bgr
    CVImage(frame_get).show()
```

### CVFile

Make pickle/txt/json/yaml be one CVFile.

#### example

```python
from cv2box import CVFile
import numpy as np

# load from file
file_p = '*.pkl/*.json/*.npz/*.h5 etc'
print(CVFile(file_p).data)
# recursion show data structure
CVFile(file_p).show(iter_times=3)   


# write to file
dummy_dict = {}
dummy_numpy = np.array([])
CVFile(dummy_dict).json_write('write path')
CVFile(dummy_numpy).pickle_write('write path')
```

### CVExcel

basic ops for insert data to excel file.

#### example

```python
from cv2box import CVExcel
excel_p = 'test.xls'
cve = CVExcel(excel_p)
cve.create_sheet()
cve.insert_image('A1', image_path='test.jpg', image_new_size=(256,256))
cve.insert_words('B2', 'this is test words.')
```

### CVBbox

filter bboxes by center/area/center+area condition.

#### example

```python
from cv2box import CVBbox
boxes = [[],[]]
print(CVBbox(boxes).area())
boxes_result = CVBbox(boxes).area_center_filter(frame.shape, max_num=1)
```

### CVCamera

convert multical calibrate pkl to needed matrix

#### example
```python
from cv2box import CVCamera
pkl_p = 'multical_calibrate.pkl'
cvc = CVCamera(pkl_p)
print(cvc.world_view_extri_matrix())
print(cvc.multi_cam_stack(cvc.pall_rotate()))
print(cvc.intri_matrix())
print(cvc.matrix_2_rt(cvc.world_view_extri_matrix()['cam1']))
```