## cv2box - cv toolbox

```shell
# RECOMMEND - inclue cv2/numpy/tqdm
pip install cv2box

# clean install - no other dependencies
pip install cv2box[light] 

# full install - referring requirements.txt
pip install cv2box[full]

# clone codes to your project without install
git clone git@github.com:ykk648/cv2box.git --depth 1
```

```python
# Example
from cv2box import CVImage
img_in = 'img path' # or img_in = cv2.imread('img path')
img_bgr = CVImage(img_in).bgr
CVImage(img_in).resize(256,256).show()
```

A gather of tools or funcs frequently using in my work.

Now including these functions:

### cv ops
- [CVBbox](./cv_ops#cvbox) , different bbox filter.
- [CVCamera](./cv_ops#cvcamera) , load [multical](https://github.com/makerdao/multicall) calibrate result and give any format of camera matrix.
- [CVExcel](./cv_ops#cvexcel) , basic ops for insert data to excel file.
- [CVFile](./cv_ops#cvfile) , for multi-class file io.
- [CVFolder](./cv_ops) , higher level folder op to do batch file operate.
- [CVImage](./cv_ops#cvimage) , a image class support multi format convert.
- [CVQueue](./cv_ops#cvqueue) , queue-like high-level class which can be used for two different python projects communicate on same host machine.
- [CVRotate](./cv_ops) , under dev.
- [CVVideo](./cv_ops#cvvideo), a video class support multi kinds of ops.


### cv gears

- [CV threads parts](./cv_gears#CV-Threads-Parts) , fast construct your multithread AI program.
- [CV video thread](./cv_gears#CV-Video-Thread) , give a video path to get a frame thread.
- [CV multi video thread](./cv_gears#CV-Multi-Video-Thread) , give a video path to get a frame thread.

### code templates

- multiprocess example by concurrent
- keyboard listener in separate thread
- full-stack web demo for deep learning http distribute

### shell scripts

- rsync between different servers
- batch uncompress script support different types(zip tar 7z etc.)

### utils

- fps counter
- apscheduler
- cal distances

### AI power (moved to [AI_power](https://github.com/ykk648/AI_power))
