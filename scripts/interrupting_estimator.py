#! /usr/bin/env python
import pickle
import json
import os
import math
import sys
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import rospy
import rospkg
from interruptibility_msgs.msg import FeatureVector, Interruptibility
from interruptibility_srvs.srv import InterruptibilityAllQuery, InterruptibilityAllQueryResponse
from geometry_msgs.msg import PointStamped


# Debug Helpers
FAIL_COLOR = '\033[91m'
ENDC_COLOR = '\033[0m'


def eprint(error):
    sys.stderr.write(
        FAIL_COLOR
        + type(error).__name__
        + ": "
        + error.message
        + ENDC_COLOR
    )
# End Debug Helpers


class MLPEstimator(object):
    """
    This class makes interruptibility estimations based on incoming feature vectors
    """

    def __init__(self):
        rospy.init_node('MLPEstimator')
        pkg_path = rospkg.RosPack().get_path('int_estimator')
        clf_path = rospy.get_param('~model_filename', default=pkg_path+'/scripts/MLPrel.pkl')
        scal_path = rospy.get_param('~scaler_filename', default=pkg_path+'/scripts/scaler.pkl')
        self.clf = joblib.load(clf_path)
        self.scaler = joblib.load(scal_path)
        self.debug = rospy.get_param('~debug', default=False)
        self.feature_vector_sub_topic_name = rospy.get_param('~features_topic_name',
                                                             default='/data_filter/feature_vector')
        self.service_handle = rospy.Service("~interruptibilities", InterruptibilityAllQuery, self.make_predictions)
        self.data_storage_duration = rospy.get_param('~data_storage_duration', default=4.0)
        self.model_pos_frame = rospy.get_param('~model_pos_frame', default=None)
        self.should_interpolate = rospy.get_param('~should_interpolate', default=False)
        self.headers_in_scene = {}
        self.people_in_scene = {}
        self.times_added = {}
        self.positions_in_scene = {}

    def buffer_data(self, feature_vector):
        """
        Take feature vectors and run them through the MLP classifier
        """

        x_in = []
        x_in.append(feature_vector.pose.right_wrist_angle)                # 0
        x_in.append(feature_vector.pose.right_elbow_angle)                # 1
        x_in.append(feature_vector.pose.left_wrist_angle)                 # 2
        x_in.append(feature_vector.pose.left_elbow_angle)                 # 3
        x_in.append(feature_vector.pose.left_eye_angle)                   # 4
        x_in.append(feature_vector.pose.right_eye_angle)                  # 5
        x_in.append(feature_vector.pose.right_shoulder_angle)             # 6
        x_in.append(feature_vector.pose.left_shoulder_angle)              # 7
        x_in.append(feature_vector.pose.nose_vec_y)                       # 8
        x_in.append(feature_vector.pose.nose_vec_x)                       # 9

        x_in.append(feature_vector.gaze_boolean.gaze_at_robot)            # 10

        # TODO: undo the missing data numbers from the data filter, set to -5
        for val in range(len(x_in)):
            if x_in[val] >= 1.7e300:
                x_in[val] = -5

        x_in.append(0)  # book 11
        x_in.append(0)  # bottle 12
        x_in.append(0)  # bowl 13
        x_in.append(0)  # cup 14
        x_in.append(0)  # laptop 15
        x_in.append(0)  # cell phone 16
        x_in.append(0)  # blocks 17
        x_in.append(0)  # tablet 18
        x_in.append(0)  # unknown 19
        foi = 11  # first object index to make it easier to change stuff
        for item in feature_vector.object_labels.object_labels:
            if item == 'book':
                x_in[foi] += 1
            elif item == 'bottle':
                x_in[foi + 1] += 1
            elif item == 'bowl':
                x_in[foi + 2] += 1
            elif item == 'cup':
                x_in[foi + 3] += 1
            elif item == 'laptop':
                x_in[foi + 4] += 1
            elif item == 'cell phone':
                x_in[foi + 5] += 1
            elif item == 'blocks':
                x_in[foi + 6] += 1
            elif item == 'tablet':
                x_in[foi + 7] += 1
            else:
                x_in[foi + 8] += 1

        # TODO: save the person in self.people_in_scene
        self.people_in_scene[feature_vector.person_id] = x_in
        self.times_added[feature_vector.person_id] = time.time()
        self.headers_in_scene[feature_vector.person_id] = feature_vector.header
        self.positions_in_scene[feature_vector.person_id] = feature_vector.body_position.position
        # TODO: clean people out after 4 seconds
        self.clean_people()
        # TODO: Do I need to propagate specific values?

    def clean_people(self):
        for person in self.people_in_scene:
            # Haven't seen the person in at least 4 seconds
            if time.time() - self.times_added[person] > self.data_storage_duration:
                del self.people_in_scene[person]
                del self.headers_in_scene[person]
                del self.positions_in_scene[person]
                del self.times_added[person]

    def make_predictions(self):
        res = InterruptibilityAllQueryResponse()
        for person in self.people_in_scene:
            x_in = self.scaler.transform(self.people_in_scene[person])
            prediction = self.clf.predict(x_in)

            pnt = PointStamped()
            pnt.header = self.headers_in_scene[person]
            pnt.point = self.positions_in_scene[person]

            msg = Interruptibility()
            msg.header = self.headers_in_scene[person]
            msg.person_id = person
            msg.estimate = prediction
            msg.position = pnt
            res.interruptibilities.append(msg)
        return res

    def run(self):
        # subscribe to sub_image_topic and callback parse
        rospy.Subscriber(self.feature_vector_sub_topic_name, FeatureVector, self.buffer_data)
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = MLPEstimator()
        detector.run()
    except rospy.ROSInterruptException:
        pass
