import os
import numpy as np
import cv2
import tensorflow as tf
from model_matrixs import calc_IOU, calc_IOU_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_PATH = "E:/Projects/videos"
HEIGHT = 256
WIDTH = 256

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    VIDEO_PATH = "videos/raw/video-3.mp4"
    # CAMVID_MODEL = "Camvid-Weights1.hdf5"
    KITTI_MODEL = "model_v1.h5"

    # video_path = os.path.join(base_path, "videos/video-1.mp4")

    # print(os.listdir(video_path))

    video_filename = VIDEO_PATH.split('/')[-1]
    print(video_filename)
    """ Loading model """
    with tf.keras.utils.CustomObjectScope({"calc_IOU": calc_IOU, "calc_IOU_loss": calc_IOU_loss}):
        model = tf.keras.models.load_model(KITTI_MODEL)
    print(model.summary())

    vs = cv2.VideoCapture(VIDEO_PATH)
    _, frame = vs.read()
    # print(frame)
    H, W, _ = frame.shape
    vs.release()
    # print(H, W)

    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')


    out = cv2.VideoWriter((os.path.join("videos/output", video_filename)), fourcc, 10, (W, H), True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    idx = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            out.release()
            print("------------All frames are processed of the video.---------")
            break

        ori_frame = frame.copy()
        frame = cv2.resize(frame, (256, 256))   # frame size (256, 256,3)
        frame = np.expand_dims(frame, axis=0)   # Creating a 1 batch of image frame (1, 256,256,3)
        frame = frame / 255.0

        mask = model.predict(frame)     # output size (1, 256, 256, 1)
        # print(mask.shape)
        mask = mask[0]      # taking the predicted mask without the batch (256, 256, 1)

        # creating the thresholding for the predicted image.
        mask = mask > 0.5
        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, (W, H))

        # print(mask.shape)

        mask = np.expand_dims(mask, axis=-1)
        # print(mask.shape)
        # print("Original frame size: ", ori_frame.shape)
        # print("Mask size: ", mask.shape)
        # exit(0)
        # if camvidmodel:
        #     mask = np.mean(mask, axis=2)
            # print("Mask size: ", mask.shape)
            # exit(0)

        # mask = mask*255       # for camvid
        combine_frame = ori_frame * mask
        combine_frame = combine_frame.astype(np.uint8)

        # cv2.imwrite(f"videos/output/{idx}.png", combine_frame)
        # idx += 1
        out.write(combine_frame)

