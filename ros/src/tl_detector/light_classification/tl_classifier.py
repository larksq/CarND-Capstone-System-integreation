from styx_msgs.msg import TrafficLight

import tensorflow as tf
import rospy
import numpy as np
from utils import label_map_util
from google.protobuf import text_format


PATH_TO_GRAPH = 'udacity_models/ssd_sim/frozen_inference_graph.pb'  # load SSD trained on udacity's simulation images
# PATH_TO_GRAPH = r'models/ssd_udacity/frozen_inference_graph.pb' ## load SSD trained on udacity's parking lot images
PATH_TO_LABELS = 'data/udacity_label_map.pbtxt'
NUM_CLASSES = 13
CLASSIFICATION_THRESHOLD = 0.7

class TLClassifier(object):

    def __init__(self):
        self.detection_graph = self.load_graph(PATH_TO_GRAPH)
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        # category_index = label_map_util.create_category_index(categories)
        # print(category_index)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detect_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detect_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detect_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image_np = self.load_image_into_numpy_array(image)
                image_expanded = np.expand_dims(image_np, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detect_boxes, detect_scores, detect_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

                # labels: {1: {'id': 1, 'name': u'Green'}, 2: {'id': 2, 'name': u'Red'}, 3: {'id': 3, 'name': u'Yellow'}, 4: {'id': 4, 'name': u'off'}}
                # logic checking: if any class > Threshold, then return

                score = scores[0]

                if score[0] > CLASSIFICATION_THRESHOLD:
                    label = classes[0][0]
                    if label == 1:
                        return 2
                    elif label == 2:
                        return 0
                    elif label == 3:
                        return 1

                return TrafficLight.UNKNOWN

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def load_image_into_numpy_array(self, image):
        (im_width, im_height, channels) = image.shape
        np_image = np.array(image.ravel().reshape((im_width, im_height, 3)).astype(np.uint8))
        return np_image

