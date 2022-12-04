import numpy as np


def get_human_detection(pred_output, dims):
    """Return bounding boxes corresp. to humans given frame (inception model)
        Args:
            pred_output (tuple): each triple (i.e. given j, output[i][j] for i = 1,2,3)
                            represents a detection with bbox coordinates, score, and class
                            values
            dims (tuple): width and height of image

        Returns:
            list: list of tuples corresponding to human detections
    """

    boxes, scores, classes = pred_output
    # process scores and classes
    scores = scores[0].tolist()
    classes = classes[0].tolist()
    # dimensions of video
    height, width = dims

    # initialize empty list to append human det bboxes
    hum_dets = []
    for i in range(boxes.shape[1]):
        if classes[i] == 1 and scores[i] > 0.70:
            box =[boxes[0, i, j] for j in range(4)]
            # scale normalized bbox coordinates to frame dims
            box *= np.array([height, width, height, width])
            # append scaled bbox coordinates
            hum_dets.append(tuple(map(int, box)))
    return hum_dets
