import torch
import cv2

import torchvision.transforms as transforms

import numpy as np


class ReID:

    def __init__(
        self,
        model=None,
        queries=[],
        targets=[],
        target_bboxes=[],
        input_width=128,
        input_height=256
    ):

        self.queries = queries  # numpy array (feature)
        self.targets = targets
        self.target_bboxes = target_bboxes
        self.model = model  # torch model

        self.euc_dist_mat = None
        self.input_width = input_width
        self.input_height = input_height


    def reset_targets(self):
        self.targets = []
        self.target_bboxes = []

    def reset_dists(self):
        self.euc_dist_mat = None

    def compute_distances(self):
        # now we compute euclidean distance
        # we use cdist method from torch
        # each column i corresponds to query feature i
        # and each entry i, j corresponds to the distance
        # b/w query feature i against target sample feature j

        # features
        query_features = torch.vstack(self.queries)
        target_features = torch.vstack(self.targets)

        # distances
        self.euc_dists_mat = torch.cdist(target_features, query_features)


    def get_rank(self):
        rank1_matches = torch.argmin(
            self.euc_dists_mat,
            dim=0
        )
        return rank1_matches

    def preprocess_crop(self, crop):

        crop_RGB = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        crop_RGB_resized = cv2.resize(
            crop_RGB,
            (self.input_width ,self.input_height),
            cv2.INTER_CUBIC
        )

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # Convert the image to Torch tensor
        tensor = transform(crop_RGB_resized)
        input_tensor = torch.unsqueeze(tensor, 0)

        return input_tensor

    def add_feature(self, crop, feature_type="query", bbox=None):
        # preprocess crop image
        crop_input = self.preprocess_crop(crop)

        # compute feature
        feature = self.model(crop_input)

        if feature_type == "query":
            self.queries.append(
                feature
            )
        else:
            self.targets.append(
                feature
            )
            self.target_bboxes.append(bbox)
