# -- coding: utf-8 --
# @Time : 2022/4/22
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box


import gradio as gr


def greet(name):
    return "Hello " + name + "!!"


iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch(server_name='0.0.0.0', share=False)
