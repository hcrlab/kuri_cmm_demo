#!/usr/bin/env python3
import rospy

import math, random
# from kuri_api.utils import interp
# from kuri_api.anim import AnimationGroup
# from kuri_api.anim import Track
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

import numpy as np
import logging
from enum import Enum
import cv2
import threading

from scipy.ndimage.measurements import label

# import tf.transformations
# from kuri_api.utils import timeouts
from threading import Event
# import kuri_api

class ViewTuner(object):
    def __init__(self, img_topic):
        # joint_state = kuri_api.JointStates()
        # head_mux = kuri_api.HeadMux(joint_state, tf.TransformListener())
        # self.animations = AnimationGroup(head_mux.animations)

        self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

        self.bridge = CvBridge()
        self.img_pub_with_saliency_center = rospy.Publisher('img_with_saliency_center', Image, queue_size=1)
        self.img_pub_saliency = rospy.Publisher('saliency_map', Image, queue_size=1)
        self.img_pub_connected_components = rospy.Publisher('connected_components', Image, queue_size=1)
        self.img_sub = rospy.Subscriber(
            img_topic, CompressedImage, self.img_callback, queue_size=1)

    def img_callback(self, img_msg, threshold=0.05, duration=0.1):
        """
        Called when a new img_msg is received. Computes the saliency map of the
        image, computes the center of saliency of the image, and moves the head
        to center that.

        Inputs:
        - img_msg is the ROS CompressedImage message
        - if the x and y of the center of saliency is within threshold
          proportion of the width/height to the true center of the image, stop
          moving the head.
        """
        # Convert to a cv2 image
        img_data = np.fromstring(img_msg.data, np.uint8)
        img_cv2 = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        w = img_cv2.shape[1]
        h = img_cv2.shape[0]

        # Get the center of saliency (cx, cy) in the image
        (success, saliency_map) = self.saliency.computeSaliency(img_cv2)
        total_saliency = np.sum(saliency_map)
        if not success:
            rospy.logwarn("Failed to get saliency_map")
            return

        # Turn it into a binary map
        binary_saliency_map = saliency_map >= total_saliency/w/h
        connected_components, n_components = label(binary_saliency_map, np.ones((3,3), dtype=np.int32))
        connected_components_relative_entropy = np.zeros(connected_components.shape, dtype=np.uint8)
        # print("connected_components", connected_components, n_components)
        # For each connected components, compute its relative entropy
        max_relative_entropy, max_relative_entropy_comp_i = None, None
        for comp_i in range(1,n_components+1):
            # If saliency was uniformly distributed, what proportion of it
            # would fall into this connected component?
            region_prob = np.count_nonzero(connected_components == comp_i)/w/h
            # What proportion of the saliency actually falls into this connected
            # component?
            region_density = np.sum(saliency_map[np.where(connected_components == comp_i)])/total_saliency
            if region_density <= region_prob:
                relative_entropy = 0.0
            else:
                relative_entropy = (region_density*math.log(region_density/region_prob) +
                    (1-region_density)*math.log((1-region_density)/(1-region_prob)))

            print("region_density", region_density)
            connected_components_relative_entropy = np.where(connected_components == comp_i, int(round(region_density*255)), connected_components_relative_entropy)

            if max_relative_entropy is None or relative_entropy > max_relative_entropy:
                max_relative_entropy = relative_entropy
                max_relative_entropy_comp_i = comp_i

        print("connected_components_relative_entropy", connected_components_relative_entropy)

        saliency_of_region_of_interest = np.where(connected_components == max_relative_entropy_comp_i, saliency_map, 0)
        weights = saliency_of_region_of_interest/np.sum(saliency_of_region_of_interest)
        cy = int(round(np.dot(np.arange(h), np.sum(weights, axis=1))))
        cx = int(round(np.dot(np.arange(w), np.sum(weights, axis=0))))

        true_cx = weights.shape[1]/2
        true_cy = weights.shape[0]/2
        w = weights.shape[1]*0.1
        h = weights.shape[0]*0.1

        img_annotated = cv2.circle(img_cv2, (cx, cy), 5, color=(0, 0, 255), thickness=-1)
        img_annotated = cv2.rectangle(img_annotated, (int(round(true_cx-w/2)), int(round(true_cy-h/2))),  (int(round(true_cx+w/2)), int(round(true_cy+h/2))), color=(255, 0, 0), thickness=2)
        # print("img_annotated", img_annotated)
        img_msg_annotated = self.bridge.cv2_to_imgmsg(img_annotated, encoding="passthrough")
        # print("img_msg_annotated.step", img_msg_annotated.step)
        img_msg_annotated.step = int(img_msg_annotated.step)
        self.img_pub_with_saliency_center.publish(img_msg_annotated)

        img_saliency = (saliency_map * 255).astype("uint8")
        img_saliency = cv2.circle(img_saliency, (cx, cy), 5, color=(0, 0, 255), thickness=-1)
        img_saliency = cv2.rectangle(img_saliency, (int(round(true_cx-w/2)), int(round(true_cy-h/2))),  (int(round(true_cx+w/2)), int(round(true_cy+h/2))), color=(255, 0, 0), thickness=2)
        img_saliency_annotated = self.bridge.cv2_to_imgmsg(img_saliency, encoding="passthrough")
        # print("img_msg_annotated.step", img_msg_annotated.step)
        img_saliency_annotated.step = int(img_saliency_annotated.step)
        self.img_pub_saliency.publish(img_saliency_annotated)

        img_conn_comp = connected_components_relative_entropy
        img_conn_comp = cv2.circle(img_conn_comp, (cx, cy), 5, color=(0, 0, 255), thickness=-1)
        img_conn_comp = cv2.rectangle(img_conn_comp, (int(round(true_cx-w/2)), int(round(true_cy-h/2))),  (int(round(true_cx+w/2)), int(round(true_cy+h/2))), color=(255, 0, 0), thickness=2)
        img_conn_comp_annotated = self.bridge.cv2_to_imgmsg(img_conn_comp, encoding="passthrough")
        # print("img_msg_annotated.step", img_msg_annotated.step)
        img_conn_comp_annotated.step = int(img_conn_comp_annotated.step)
        self.img_pub_connected_components.publish(img_conn_comp_annotated)

        # Generate the head motion to center (cx, cy)
        dx_to_center = cx/weights.shape[1]-0.5
        dy_to_center = cy/weights.shape[0]-0.5
        if abs(dx_to_center) <= threshold and abs(dy_to_center) <= threshold:
            rospy.loginfo("Done tuning head position")
            return
        init_tilt = self.animations.head.cur_tilt
        init_pan = self.animations.head.cur_pan
        tk = Track()
        tk.add(0.0, self.head_mot.pantilt(
            init_pan,
            init_tilt,
            duration,
        ))
        tk.add(
            curr_time,
            self.head_mot.pantilt(
                init_pan - dx_to_center/2.0,
                init_tilt + dy_to_center/2.0,
                duration,
            ),
        )
        tk.play()
        # # Run the head motion in another thread to not block the img_callback
        # thread = threading.Thread(
        #     target=tk.play,
        #     args=(,)
        # )
        # thread.start()

if __name__ == '__main__':
    rospy.init_node('view_tuner')

    img_topic = rospy.get_param('~img_topic', '/upward_looking_camera/compressed')
    view_tuner = ViewTuner(img_topic)

    rospy.spin()
