import os
import cv2
import tqdm
import yaml

import numpy as np

from os.path import isfile, join
from src.rcnn_coco_obj_det import Model
from src.utils import get_human_detection
from src.reid import ReID
from src.reid_model import ReID_Model


if __name__=="__main__":

    # open and read in config file
    with open("confs/config.yml", "r") as f:
        config = yaml.safe_load(f)

    input_path = config["input"]["frames_path"]
    output_path = config["output"]["save_path"]
    fps = config["output"]["fps"]

    inception_model_path = config["model"]["inception"]
    reid_model_path = config["model"]["reid"]

    frame_array = []
    files = [f for f in os.listdir(input_path) if isfile(join(input_path, f))]

    # instantiate inception model
    model = Model(inception_model_path)

    # instantiate ReID object
    reid_model = ReID_Model()
    reid_model.load_weights(weight_path=reid_model_path)
    reid_model = reid_model.model
    reid = ReID(reid_model)

    # bbox colors
    COLOR_GREEN = (0, 255, 0)
    COLOR_ORANGE = (255, 128, 0)

    # sort frames
    files.sort(key=lambda x: int(x[6:-4]))

    # query frame
    query_frame = 0

    for i in tqdm.tqdm(range(len(files))):
        filename=input_path + files[i]

        # read each file
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # extract dims
        dims = height, width
        # predict
        output = model.predict(img)
        # post-process detections filter everything except human dets
        human_dets = get_human_detection(output, dims)


        # draw bboxes around hum dets
        # and reid query_frame human detection in subsequent frames

        for j, bbox in enumerate(human_dets):

            # filter unreasonable dimensions
            height = abs(bbox[0] - bbox[2])
            width = abs(bbox[1] - bbox[3])
            if width > 50 or height > 150: continue

            # let's crop the detections
            bbox_crop = img[bbox[0]: bbox[2], bbox[1]: bbox[3]]

            # choose the query detection
            # TODO: automate this conditional statement
            if i == query_frame and j == 0:
                reid.add_feature(bbox_crop, feature_type="query")
            # after query_frame, we add targets
            if i > query_frame:
                reid.add_feature(bbox_crop, feature_type="target", bbox=bbox)

            cv2.rectangle(img, (bbox[1],bbox[0]), (bbox[3],bbox[2]), COLOR_GREEN, 2)

        if i > query_frame:
            # compute ranks
            reid.compute_distances()
            argmax_indices = reid.get_rank()
            argmax_index = argmax_indices[0]  # since we have only one query
            reid_bbox = reid.target_bboxes[argmax_index]

            # clear
            reid.reset_targets()
            reid.reset_dists()

            # highlight ReID
            cv2.rectangle(img, (reid_bbox[1],reid_bbox[0]), (reid_bbox[3],reid_bbox[2]), COLOR_ORANGE, 2)

        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
