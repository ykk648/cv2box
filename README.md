## cv2box - cv toolbox

```
# clean install - no other dependencies
pip install cv2box

# RECOMMEND - inclue cv2/numpy/.. 
pip install cv2box[light] 

# full install - referring requirements.txt
pip install cv2box[full]
```


A gather of tools or funcs frequently using in my work.

more info and code example plz read [wiki](https://github.com/ykk648/cv2box/wiki)

Now including these functions:

### AI power (move to [AI_power](https://github.com/ykk648/AI_power))

### cv ops

- [CVImage](./cv_ops#cv-image) , a image class support multi format convert, usage: todo

- [CVQueue](./cv_ops#cv-queue) , A queue-like high-level class which can be used for two different python projects on same host machine.

- [CVFile](./cv_ops#cv-file) , for multi-class file io.

- [CVExcel](./cv_ops#cv-excel) , basic ops for insert data to excel file.

- [CVVideo](./cv_ops#cv-video), a video class support multi kind of ops.


### code templates

- multiprocess example by concurrent
- multiprocess consumer example
- keyboard listener in separate thread
- full-stack web demo for deep learning http distribute

### shell scripts

- rsync between different servers
- batch uncompress script support different types(zip tar 7z etc.)

### functions in py file (deprecated)