#!/usr/bin/env python3

import rospy
import zed_interfaces.msg as zed_msgs

import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import shutil
import os

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped


lock = Lock()
run_signal = False
exit_signal = False
class_names = []
global svo, img_size, conf_thres, model_name
svo = None
img_size = 416
conf_thres = 0.4
model_name = 'yolov8m-ch'

CAMERA_NAME = "zed2i"

def xywh2abcd(xywh):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max
    return output

def detections_to_custom_box(detections):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output


def torch_thread(model_name, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, class_names

    print("Intializing Network...")

    script_path = os.path.dirname(os.path.realpath(__file__))
    models_path = script_path+'/../models/'
    model_path = models_path+model_name+'.pt'
    model_labels_path = models_path+model_name+'_labels.txt'

    # Check if the model does not exists
    if not os.path.isfile(model_path):
        print("Model not found, downloading it...")

        # Get the PyTorch model
        model = YOLO(model_name+'.pt')
        class_names = model.names

        # Export the model as an engine
        #model.export(format='engine')    

        # Copy the models into the correct directory
        shutil.copy(model_name+'.pt', models_path+model_name+'.pt')
        #shutil.copy(model_name+'.engine', models_path+model_name+'.engine')

        # Export the class names dictionary as a file
        with open(model_labels_path, 'w') as f:
            for i in range(len(class_names)):
                f.write("%d, %s\n" % (i, class_names[i]))

    # Export the class names dictionary as a file
    with open(model_labels_path, 'w') as f:
        for i in range(len(class_names)):
            f.write("%d, %s\n" % (i, class_names[i]))       
    # Load the model
    model = YOLO(models_path+model_name+'.pt')

    # Load class names as a list
    with open(model_labels_path, 'r') as f:
        class_names = f.read().splitlines()
    class_names = [name.split(', ')[1] for name in class_names]

    while not exit_signal:
        if run_signal:
            lock.acquire()

            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
            # https://docs.ultralytics.com/modes/predict/#video-suffixes
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det)
            lock.release()
            run_signal = False
        sleep(0.001)

# Wrap data into ROS ObjectStamped message
def ros_wrapper(objects):
    ros_msg = zed_msgs.ObjectsStamped()
    ros_msg.header.stamp = rospy.Time.now()

    # When working on chair, uncomment 1st line and comment 2nd line
    # ros_msg.header.frame_id = CAMERA_NAME + "_left_camera_frame"
    ros_msg.header.frame_id = CAMERA_NAME

    obj_list = []
    for obj in objects.object_list:
        obj_msg = zed_msgs.Object()
        obj_msg.label = class_names[obj.raw_label]
        obj_msg.label_id = obj.raw_label
        obj_msg.sublabel = repr(obj.id)
        obj_msg.instance_id = obj.id
        obj_msg.confidence = obj.confidence
        pos = obj.position
        obj_msg.position = [pos[0], pos[1], pos[2]]
        pos_cov = obj.position_covariance
        obj_msg.position_covariance = [pos_cov[0], pos_cov[1], pos_cov[2], pos_cov[3], pos_cov[4], pos_cov[5]]
        vel = obj.velocity
        obj_msg.velocity = [vel[0], vel[1], vel[2]]
        obj_msg.tracking_available = True
        if repr(obj.tracking_state) == "OFF":
            obj_msg.tracking_state = 0
        elif repr(obj.tracking_state) == "OK":
            obj_msg.tracking_state = 1
        else:
            obj_msg.tracking_state = 2
        bbox_3d = obj.bounding_box
        if len(bbox_3d) == 8:
            obj_msg.bounding_box_3d.corners[0].kp = [bbox_3d[0][0], bbox_3d[0][1], bbox_3d[0][2]]
            obj_msg.bounding_box_3d.corners[1].kp = [bbox_3d[1][0], bbox_3d[1][1], bbox_3d[1][2]]
            obj_msg.bounding_box_3d.corners[2].kp = [bbox_3d[2][0], bbox_3d[2][1], bbox_3d[2][2]]
            obj_msg.bounding_box_3d.corners[3].kp = [bbox_3d[3][0], bbox_3d[3][1], bbox_3d[3][2]]
            obj_msg.bounding_box_3d.corners[4].kp = [bbox_3d[4][0], bbox_3d[4][1], bbox_3d[4][2]]
            obj_msg.bounding_box_3d.corners[5].kp = [bbox_3d[5][0], bbox_3d[5][1], bbox_3d[5][2]]
            obj_msg.bounding_box_3d.corners[6].kp = [bbox_3d[6][0], bbox_3d[6][1], bbox_3d[6][2]]
            obj_msg.bounding_box_3d.corners[7].kp = [bbox_3d[7][0], bbox_3d[7][1], bbox_3d[7][2]]
        obj_list.append(obj_msg)         
    ros_msg.objects = obj_list
    return ros_msg  

def local_to_map_transform(msg, tfBuffer):
    try:
        # Uncomment when working on chair
        # lct = tfBuffer.get_latest_common_time("map", CAMERA_NAME + "_left_camera_frame")
        # transform = tfBuffer.lookup_transform("map", CAMERA_NAME + "_left_camera_frame", lct, rospy.Duration(0.1))
        for obj in msg.objects:
            p = PointStamped()
            p.point.x = obj.position[0]
            p.point.y = obj.position[1]
            p.point.z = obj.position[2]

            # Uncomment 2 lines below when working on chair, comment 3rd line out
            # obj.position = tf2_geometry_msgs.do_transform_point(p, transform)
            # obj.position = [obj.position.point.x, obj.position.point.y, obj.position.point.z]
            obj.position = [p.point.x, p.point.y, p.point.z]

            for corner in obj.bounding_box_3d.corners:
                p.point.x = corner.kp[0]
                p.point.y = corner.kp[1]
                p.point.z = corner.kp[2]

                # Uncomment 2 lines below when working on chair, comment 3rd line out
                # corner.kp = tf2_geometry_msgs.do_transform_point(p, transform)
                # corner.kp = [corner.kp.point.x, corner.kp.point.y, corner.kp.point.z]
                corner.kp = [p.point.x, p.point.y, p.point.z]
        # uncomment 1st line when working on chair
        # msg.header.frame_id = "map"
        msg.header.frame_id = CAMERA_NAME
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Failed to transform object from local to map frame")
    return msg


def main():
    global image_net, exit_signal, run_signal, detections, class_names, svo, img_size, conf_thres, model_name

    # Define ROS publisher 
    pub_l = rospy.Publisher(CAMERA_NAME+'/od_yolo', zed_msgs.ObjectsStamped, queue_size=50)
    pub_g = rospy.Publisher(CAMERA_NAME+'/od_yolo_map_frame', zed_msgs.ObjectsStamped, queue_size=50)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    br = tf2_ros.TransformBroadcaster()

    capture_thread = Thread(target=torch_thread, kwargs={'model_name': model_name, 'img_size': img_size, "conf_thres": conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    # Convert empty string to None for SVO parameter
    if svo == '':
        svo = None
    if svo is not None:
        input_type.set_from_svo_file(svo)
        
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()


    while rospy.is_shutdown() is False:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)
            
            # Publish in ROS as an ObjectStamped message
            ros_msg = ros_wrapper(objects)
            pub_l.publish(ros_msg)
            pub_g.publish(local_to_map_transform(ros_msg, tfBuffer))
    zed.close()


if __name__ == '__main__':
    rospy.init_node("zed_yolo_ros", anonymous=False)
    rospy.loginfo("ZED YOLO node started")
    
    # Namespace for parameters
    namespace = rospy.get_name() + "/"
    
    # Read parameters from the parameter server
    model_name = rospy.get_param(namespace + 'model_name', 'yolov8m-ch')
    svo = rospy.get_param(namespace + 'svo', None)
    img_size = rospy.get_param(namespace + 'img_size', 416)
    conf_thres = rospy.get_param(namespace + 'conf_thres', 0.4)

    rospy.loginfo(f"Model Name: {model_name}")
    rospy.loginfo(f"SVO File: {svo}")
    rospy.loginfo(f"Image Size: {img_size}")
    rospy.loginfo(f"Confidence Threshold: {conf_thres}")

    with torch.no_grad():
        main()
