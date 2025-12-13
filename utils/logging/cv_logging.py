# -- coding: utf-8 --
# @Time : 2022/8/18
# @LastEdit : 2025/10/14
# @Author : ykk648


import logging
import os
import base64
from tqdm.contrib.logging import logging_redirect_tqdm

CV_LOG_LEVEL = os.environ['CV_LOG_LEVEL']
LOGGER_NAME = 'cv2box'

# 日志加密配置
CV_LOG_ENCRYPT_KEY = 'cv2box_log_secret_key'
CV_LOG_IV = 'cv2boxcv2box1234'

LEVEL_DICT = {
    'debug': 10,
    'info': 20,
    'warning': 30,
    'error': 30,
    'critical': 50,
}

# 简化的加密实现，避免循环依赖
class SimpleLogEncryptor:
    """简化的日志加密器，专用于日志加密，避免循环依赖"""
    
    def __init__(self, key='cv2box_log_secret_key', iv='cv2boxcv2box123456'):
        self.key = key
        self.iv = iv.encode('utf-8')
        self._cipher_cache = None
    
    def _get_cipher(self):
        """延迟导入AES模块，避免循环依赖"""
        if self._cipher_cache is None:
            try:
                from Crypto.Cipher import AES
                from Crypto.Util.Padding import pad, unpad
                self._AES = AES
                self._pad = pad
                self._unpad = unpad
                
                # 处理密钥长度
                key = self.key + 'scNwy/x9h0sGBb/kQYsMREQJHwDT34wYLCnMRQ6EC14VPLd7VvO6esfYuOF0EjEI'
                if len(key) > 16:
                    key = key[:16]
                key = key.encode('ascii')
                
                self._cipher_cache = self._AES.new(key=key, mode=self._AES.MODE_CBC, iv=self.iv)
                return True
            except ImportError:
                # 如果没有安装crypto库，返回False
                return False
        return True
    
    def encrypt(self, message):
        """加密消息"""
        if not self._get_cipher():
            return f"[ENCRYPT_UNAVAILABLE]{message}"
        
        try:
            message_bytes = message.encode('utf-8')
            padded_bytes = self._pad(message_bytes, 16)
            
            # 每次加密都创建新的cipher对象，避免重复使用
            key = self.key + 'scNwy/x9h0sGBb/kQYsMREQJHwDT34wYLCnMRQ6EC14VPLd7VvO6esfYuOF0EjEI'
            if len(key) > 16:
                key = key[:16]
            key = key.encode('ascii')
            cipher = self._AES.new(key=key, mode=self._AES.MODE_CBC, iv=self.iv)
            
            encrypted_bytes = cipher.encrypt(padded_bytes)
            encrypted_base64 = base64.b64encode(encrypted_bytes).decode('utf-8')
            return f"[ENCRYPTED]{encrypted_base64}"
        except Exception as e:
            return f"[ENCRYPT_FAILED]{message} (Error: {str(e)})"
    
    def decrypt(self, encrypted_message, remove_prefix=True):
        """解密消息"""
        if not self._get_cipher():
            return f"[DECRYPT_UNAVAILABLE]{encrypted_message}"
        
        try:
            # 移除前缀标识
            if remove_prefix and encrypted_message.startswith('[ENCRYPTED]'):
                encrypted_message = encrypted_message[11:]
            
            # base64解码
            encrypted_bytes = base64.b64decode(encrypted_message.encode('utf-8'))
            
            # 每次解密都创建新的cipher对象
            key = self.key + 'scNwy/x9h0sGBb/kQYsMREQJHwDT34wYLCnMRQ6EC14VPLd7VvO6esfYuOF0EjEI'
            if len(key) > 16:
                key = key[:16]
            key = key.encode('ascii')
            cipher = self._AES.new(key=key, mode=self._AES.MODE_CBC, iv=self.iv)
            
            # 解密并移除padding
            decrypted_bytes = cipher.decrypt(encrypted_bytes)
            unpadded_bytes = self._unpad(decrypted_bytes, 16)
            return unpadded_bytes.decode('utf-8')
        except Exception as e:
            return f"[DECRYPT_FAILED] {encrypted_message} (Error: {str(e)})"

# 全局加密器实例
_log_encryptor = SimpleLogEncryptor(key=CV_LOG_ENCRYPT_KEY, iv=CV_LOG_IV)


def cv_print(message, *args, level='info', encrypt=False):
    """
    日志打印函数，支持加密输出
    
    Args:
        message: 日志消息
        *args: 额外参数
        level: 日志级别
        encrypt: 是否加密日志内容
    """
    logger = logging.getLogger(LOGGER_NAME)
    
    # 格式化完整的日志消息
    if args:
        full_message = message % args
    else:
        full_message = str(message)
    
    # 如果启用加密，则加密日志内容
    if encrypt:
        full_message = _encrypt_log_message(full_message)
    
    with logging_redirect_tqdm(loggers=[logger]):
        if level == 'debug':
            logger.debug(full_message)
        elif level == 'info':
            logger.info(full_message)
        elif level == 'warning':
            logger.warning(full_message)
        elif level == 'error':
            logger.error(full_message)
        elif level == 'critical':
            logger.critical(full_message)


def _encrypt_log_message(message):
    """
    加密日志消息
    
    Args:
        message: 待加密的日志消息字符串
        
    Returns:
        str: 加密后的base64编码字符串，带有[ENCRYPTED]前缀标识
    """
    return _log_encryptor.encrypt(message)


def decrypt_log_message(encrypted_message, remove_prefix=True):
    """
    解密日志消息
    
    Args:
        encrypted_message: 加密的日志消息（包含[ENCRYPTED]前缀或不包含）
        remove_prefix: 是否自动移除[ENCRYPTED]前缀
        
    Returns:
        str: 解密后的原始消息，如果解密失败则返回错误信息
    """
    return _log_encryptor.decrypt(encrypted_message, remove_prefix)


def decrypt_log_file(log_file_path, output_file_path=None):
    """
    解密日志文件中的加密内容
    
    Args:
        log_file_path: 包含加密日志的文件路径
        output_file_path: 输出解密后日志的文件路径，如果为None则覆盖原文件
        
    Returns:
        bool: 解密是否成功
    """
    try:
        if output_file_path is None:
            output_file_path = log_file_path + '.decrypted'
            
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        decrypted_lines = []
        for line in lines:
            if '[ENCRYPTED]' in line:
                # 提取加密部分并解密
                parts = line.split('[ENCRYPTED]')
                if len(parts) >= 2:
                    encrypted_part = parts[1].strip()
                    # 移除可能的日志时间戳等前缀，只保留加密内容
                    decrypted_content = decrypt_log_message(encrypted_part, remove_prefix=False)
                    # 重新构造日志行
                    decrypted_line = parts[0] + decrypted_content + '\n'
                    decrypted_lines.append(decrypted_line)
                else:
                    decrypted_lines.append(line)
            else:
                decrypted_lines.append(line)
                
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.writelines(decrypted_lines)
            
        return True
        
    except Exception as e:
        print(f"解密日志文件失败: {str(e)}")
        return False


def set_log_encryption_key(key, iv=None):
    """
    设置日志加密的密钥和IV
    
    Args:
        key: 加密密钥字符串
        iv: 初始向量字符串，如果为None则使用默认值
    """
    global _log_encryptor
    if iv is None:
        iv = CV_LOG_IV
    _log_encryptor = SimpleLogEncryptor(key=key, iv=iv)


def get_log_encryption_info():
    """
    获取当前日志加密设置信息
    
    Returns:
        dict: 包含密钥和IV的字典（用于调试，不包含实际密钥值）
    """
    return {
        'key_set': bool(_log_encryptor.key),
        'iv_set': bool(_log_encryptor.iv),
        'key_length': len(_log_encryptor.key) if _log_encryptor.key else 0
    }


def set_log_level(level='info'):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LEVEL_DICT[level])


def judge_log_level(level='info'):
    logger = logging.getLogger(LOGGER_NAME)
    level_now = logger.getEffectiveLevel()
    return level_now == LEVEL_DICT[level]


def cv_logging_init():
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LEVEL_DICT[CV_LOG_LEVEL])
    logger.propagate = False

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(LEVEL_DICT[CV_LOG_LEVEL])
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger
