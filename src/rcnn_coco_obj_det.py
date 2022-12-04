import numpy as np
import tensorflow as tf
import cv2
import time


class Model:
    """Model class for Faster RCNN Inception v2 model
    """

    def __init__(self, model_path):
        """ Load and create model tensorflow session graph
            Args:
                model_path (str): path to model weights
        """

        self.det_graph = tf.Graph()

        # load model
        with self.det_graph.as_default():
            # define serialized version of tensorflow graph
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as file:
                serialized_graph = file.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # create session
        self.session = tf.compat.v1.Session(graph=self.det_graph)

    def predict(self, img):
        """Generate predictions of bbox detections given frame
            Args:
                img (np.ndarray): 3-tensor containing image
        """

        # expand dimensions to [1, None, None, 3]
        img_expd = np.expand_dims(img, axis=0)

        # first declare the functions we want to use
        bbox_fun = self.det_graph.get_tensor_by_name('detection_boxes:0')
        scores_fun = self.det_graph.get_tensor_by_name('detection_scores:0')
        cls_fun = self.det_graph.get_tensor_by_name('detection_classes:0')

        # perform graph computations
        return self.session.run([bbox_fun, scores_fun, cls_fun], \
            feed_dict={self.det_graph.get_tensor_by_name('image_tensor:0'): img_expd})
