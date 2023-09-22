### CVEncrypt

Encrypt file/model with AES.

#### example

```python
from cv2box.utils.encrypt import CVEncrypt

key = "test"
file_path = ''

CVEncrypt(key).encrypt_file(file_path)
```