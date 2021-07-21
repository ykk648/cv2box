import os
import cv2
import shutil
from tqdm import tqdm


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


class VideoTools:
    def __init__(self, video_p, verbose=True):
        self.video_path = video_p
        _, self.video_name = os.path.split(self.video_path)
        _, self.extension = os.path.splitext(self.video_name)
        if verbose:
            self.print_video_info()

    def print_video_info(self):

        # 获得视频的格式
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        # 获得码率及尺寸
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        print(
            'video info:\nname: {}\nfourcc: {}\nfps: {}\nsize: {}'.format(self.video_name, decode_fourcc(fourcc), fps,
                                                                          size))

    def video_2_h264(self, inplace=True):
        # ffmpeg -i "xx.avi" -vcodec h264 "xx.mp4"
        if not inplace:
            os.system('ffmpeg -i {} -vcodec h264 {}'.format(self.video_path,
                                                            self.video_path.replace(self.extension, '_h264_out.mp4')))
        else:
            temp_p = self.video_path.replace(self.extension, '_temp_copy.mp4')
            os.system(
                'mv {} {} && ffmpeg -i {} -vcodec h264 {} && rm {}'.format(self.video_path, temp_p, temp_p,
                                                                           self.video_path, temp_p))

    # def frame_2_video(self, fps, img_size, num_frames=394):
    #     # fps = 30  # 帧率
    #     img_array = []
    #     img_width = img_size[0]
    #     img_height = img_size[1]
    #     for i in range(num_frames + 1):
    #         filename = self.root_path + '/' + "%04d" % i + ".png"
    #         # filename = self.root_path + '/' + str(i) + ".png"
    #         img = cv2.imread(filename)
    #
    #         if img is None:
    #             print(filename + " is non-existent!")
    #             continue
    #         img_array.append(img)
    #
    #     out = cv2.VideoWriter(self.root_path + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (img_width, img_height))
    #
    #     for i in range(len(img_array)):
    #         out.write(img_array[i])
    #     out.release()

    def video_2_frame(self, interval=1, out_path=None):
        if out_path is None:
            save_path = self.video_path.split('.mp4')[0] + '/'
        else:
            save_path = out_path

        is_exists = os.path.exists(save_path)
        if not is_exists:
            os.makedirs(save_path)
            print('path of %s is build' % save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
            print('path of %s already exist and rebuild' % save_path)

        # 开始读视频
        video_capture = cv2.VideoCapture(self.video_path)
        i = 0
        j = 0

        while True:
            success, frame = video_capture.read()
            i += 1
            # print(frame)
            if not success:
                print('done!')
                break
            if i % interval == 0:
                # 保存图片
                j += 1
                save_name = save_path + str(j) + '_' + str(i) + '.png'
                cv2.imwrite(save_name, frame)
                print('image of %s is saved' % save_name)
        video_capture.release()

    def resize_video(self, inplace=False, out_size=(768, 1024)):

        out_p = self.video_path.replace(self.extension, '_{}x{}.mp4'.format(out_size[0], out_size[1]))
        if inplace:
            os.system('mv {} {}'.format(self.video_path, out_p))
            self.video_path, out_p = out_p, self.video_path

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        video_writer = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)

        s = True
        while s:
            s, img1 = cap.read()
            img = cv2.resize(img1, out_size, cv2.INTER_LINEAR)
            video_writer.write(img)
        cap.release()

        if inplace:
            os.system('rm {} &'.format(self.video_path))

    # def avi2mp4(self):
    #
    #     # 指定写视频的格式, I420-avi, MJPG-mp4
    #     videoWriter = cv2.VideoWriter(self.root_path.split('.')[0] + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
    #                                   fps, size)
    #
    #     # 读帧
    #     success, frame = cap.read()
    #
    #     while success:
    #         # cv2.waitKey(1000 / int(fps))  # 延迟
    #         videoWriter.write(frame)  # 写视频帧
    #         success, frame = cap.read()  # 获取下一帧


if __name__ == '__main__':
    root_p = '/workspace/codes/DeepFaceLab_Real_Time/out_video/pyy-ljt-liae-ud-stage1.avi'
    # root_p = '/workspace/codes/cv_toolbox/tmp'
    save_p = '/workspace/data/mysterybox/mysterybox_origin/210427/out'
    video_tools = VideoTools(root_p, save_p)
    # video_tools.frame_2_video(fps=30, img_size=(1920, 1080), num_frames=1149)
    # video_tools.video_2_frame()
    # video_tools.resize_video()
    video_tools.avi2mp4()
