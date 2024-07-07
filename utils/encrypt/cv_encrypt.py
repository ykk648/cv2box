# -- coding: utf-8 --
# @Time : 2021/11/26
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
"""
24.7.7 update EBC to CBC
"""
from ... import try_import

AES = try_import('Crypto.Cipher.AES', 'cv_encrypt need crypto: pip install pycryptodome')
from Crypto.Util.Padding import pad, unpad
import operator
import time
from pathlib import Path

AES_BLOCK_SIZE = AES.block_size  # AES 加密数据块大小, 只能是16
AES_KEY_SIZE = 16  # AES 密钥长度（单位字节），可选 16、24、32，对应 128、192、256 位密钥


def _cipher(key, iv):
    key = key.encode('ascii')
    return AES.new(key=key, mode=AES.MODE_CBC, iv=iv)


def pad_key(key):
    key += 'scNwy/x9h0sGBb/kQYsMREQJHwDT34wYLCnMRQ6EC14VPLd7VvO6esfYuOF0EjEI'
    if len(key) > AES_KEY_SIZE:
        return key[:AES_KEY_SIZE]
    return key


def EnCrypt(key, bytes, iv):
    myCipher = _cipher(key, iv)
    encryptData = myCipher.encrypt(bytes)
    return encryptData


# AES 解密
def DeCrypt(key, encryptData, iv):
    myCipher = _cipher(key, iv)
    bytes = myCipher.decrypt(encryptData)
    return bytes


class CVEncrypt:
    def __init__(self, key='cv2box', iv='cv2boxcv2box123456'):
        self.key = key
        self.iv = iv.encode('utf-8')

    def encrypt_file(self, input_path_, output_path_=None):

        if type(input_path_) == type(b''):
            bytes_ori = input_path_
        else:
            if not output_path_:
                output_path_ = Path(input_path_).parent / Path(input_path_).stem / '.bin'
            with open(input_path_, 'rb') as f:
                bytes_ori = f.read()
        print('origin_length: {}'.format(len(bytes_ori)))

        key = pad_key(self.key)
        bytes = pad(bytes_ori, AES_KEY_SIZE)
        print('new_length: {}'.format(len(bytes)))

        encryptTest = EnCrypt(key, bytes, self.iv)

        start_time = time.time()
        decryptTest = unpad(DeCrypt(key, encryptTest, self.iv), AES_KEY_SIZE)
        dt = time.time() - start_time
        print("cost_time:  ", dt)

        print('recover_length: {} '.format(len(decryptTest)))
        if operator.eq(bytes_ori, decryptTest):
            print('AES success!')
            with open(output_path_, "wb") as fo:
                fo.write(encryptTest)
        else:
            print('AES failure')
        return

    def load_encrypt_file(self, model_path):
        """
        file name: xxx_{pad_len}.bin
        :param model_path:
        :return:
        """
        if type(model_path) == str:
            with open(model_path, 'rb') as f_:
                bytes_aes = f_.read()
        else:
            bytes_aes = model_path
        bytes = DeCrypt(pad_key(self.key), bytes_aes, self.iv)
        return unpad(bytes, AES_KEY_SIZE)
