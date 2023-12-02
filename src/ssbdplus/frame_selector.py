import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import os



MODEL_PATH = 'model/path/here'
movenet_model = tf.lite.Interpreter(model_path=MODEL_PATH)
movenet_model.allocate_tensors()



"""
@param path: Path of the video file need to be processed.
Extracts the frames and body joint coordinate of the child in each frame from the video
Returns:
    - Joint Coordinates (keypts) [shape: (40, 17, 3)]
    - Video Frames [shape: (40, 256, 341)] 
"""
def get_movenet_data(path):
    estimator = tf.lite.Interpreter(model_path='/content/drive/MyDrive/lite-model_movenet_singlepose_thunder_3.tflite')
    estimator.allocate_tensors()
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("Error")
        return []

    frames = []
    vid_keypts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img1 = frame.copy()
            img1 = tf.image.resize_with_pad(np.expand_dims(img1, axis=0), 256, 341)
            frames.append(np.squeeze(img1))

            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
            input_image = tf.cast(img, dtype=tf.float32)

            #  Input/Output
            input_details = estimator.get_input_details()
            output_details = estimator.get_output_details()

            # predict
            estimator.set_tensor(input_details[0]['index'], np.array(input_image))
            estimator.invoke()

            keypts = estimator.get_tensor(output_details[0]['index'])
            vid_keypts.append(keypts)

        else:
            break

    cap.release()
    return vid_keypts, frames



"""
@param keypts: Numpy vector of shape (40, 17), denoting coordinates of 17 body joints 
    detected by Movenet
Finds the frame index at which the change of key point coordinates is maximum between itself
and it's next frame. 
"""
def frame_with_max_change(keypts):
    max_diff = 0
    max_loc = 0
    for frame in range(len(keypts) - 1):
        shaped1 = np.squeeze(keypts[frame])
        shaped2 = np.squeeze(keypts[frame + 1])

        diff_mag = np.linalg.norm(shaped2 - shaped1)

        if max_diff < diff_mag:
            max_diff = diff_mag
            max_loc = frame

    return max_diff, max_loc



"""
@params: keyoints [shape: (40, 17, 3)]
Combines the x and y coordinate of keypt for each frame giving a vector of shape (40, 34) 
"""
def processed(keypts):
  proc_keypts = []

  for keypt in keypts:
    keypt = np.squeeze(keypt)[:, :2].flatten()
    proc_keypts.append(keypt)

  return proc_keypts



"""
@param: video_dir: path of the video directory to be analysed
@param: test: Boolean flag, set if eval set is being processed. It is false if directory points 
to the train set

Get all best frames and the keypoint coordinates corresponding to each video in the directory
"""

def get_best_frames(video_dir, test = False):
    best_frames = []

    # Set Label
    set_label = np.array([1.0, 0.0, 0.0], dtype = np.float32)
    if 'headbanging' in video_dir and not test:
        set_label = np.array([0.0, 1.0, 0.0], dtype = np.float32)
    elif 'spinning' in video_dir and not test:
        print("SPINNING -- TRAIN")
        set_label = np.array([0.0, 0.0, 1.0], dtype = np.float32)
    
    y = []
    all_keypts = []
    for video in os.listdir(video_dir):
        if 'noclass' in video:
            continue

        keypts, frames = get_movenet_data(video_dir + '/' + video)
        all_keypts.append(processed(keypts))

        max_diff, max_loc = frame_with_max_change(keypts)
        best_frames.append(frames[max_loc + 1])
        
        if not test:
            y.append(set_label)

        else:
            if 'armflapping' in video:
                y.append(np.array([1.0, 0.0, 0.0], dtype = np.float32))
            elif 'headbanging' in video:
                y.append(np.array([0.0, 1.0, 0.0], dtype = np.float32))
            elif 'spinning' in video:
                y.append(np.array([0.0, 0.0, 1.0], dtype = np.float32))

    if test:
        test_keys = []
        test_best_frames = []
        test_labels = []
        for frame, keypt, label in list(zip(best_frames, all_keypts, y)):
            if len(keypt) == 40:
                test_keys.append(keypt)
                test_best_frames.append(frame)
                test_labels.append(label)

        return (np.array(test_best_frames, dtype = np.float32), np.array(test_keys, dtype = np.float32)), np.array(test_labels, dtype = np.float32)

    return (best_frames, all_keypts), y




