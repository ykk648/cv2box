## cv2box - cv toolbox

```shell
# clean install - no other dependencies
pip install cv2box

# RECOMMEND - inclue cv2/numpy/.. 
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

- [CVImage](./cv_ops#cv-image) , a image class support multi format convert.

- [CVVideo](./cv_ops#cv-video), a video class support multi kinds of ops.

- [CVQueue](./cv_ops#cv-queue) , A queue-like high-level class which can be used for two different python projects on same host machine.

- [CVFile](./cv_ops#cv-file) , for multi-class file io.

- [CVExcel](./cv_ops#cv-excel) , basic ops for insert data to excel file.

- [CVBbox](./cv_ops) , different bbox filter.

- [CVRotate](./cv_ops) , under dev.

### cv gears

- [CV threads parts](./cv_gears#CV-Threads-Parts) , fast construct your multithread AI program.
- [CV video thread](./cv_gears#CV-Video-Thread) , give a video path to get a frame thread.

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

### AI power (moved to [AI_power](https://github.com/ykk648/AI_power))
