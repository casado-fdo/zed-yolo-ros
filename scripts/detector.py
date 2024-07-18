#!/usr/bin/env python3

import rospy
import math
import numpy as np
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
import tf
from geometry_msgs.msg import PointStamped, TransformStamped
import zed_yolo_ros.msg as zed_msgs
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg



lock = Lock()
run_signal = False
img_size = 416
conf_thres = 0.4
model_name = 'yolov8m-pose'
zed_location = 'detached'
skeletons = None

CAMERA_NAME = "zed2i"

# Convert 2-D Pose keypoints to 3-D points
def ingest_skeletons(skeletons, labels, point_cloud):
    # Check to see if there are any skeletons
    if skeletons is None:
        people = None
        labels = None
        return people, labels
    
    # Create a list of people, each with 17 4-d keypoints
    num_people = skeletons.shape[0]
    people = np.zeros((num_people, 17, 4))
    
    # Iterate through the people
    for i in range(num_people):
        person = skeletons[i]
        
        # Iterate through the keypoints
        for j in range(person.shape[0]):
            keypoint = person[j]

            # Extract the keypoint's data:
            # x, y: pixel coordinates (cast as an int)
            # conf: confidence score
            x = int(keypoint[0])
            y = int(keypoint[1])
            conf = keypoint[2]
            
            # Find the corresponding 3D point in the point cloud
            err,point_3d = point_cloud.get_value(x, y)
            
            # Save the 3D point and confidence score
            people[i, j] = [point_3d[0], point_3d[1], point_3d[2], conf]
    
    # Return the people and their IDs/labels
    return people, labels

# Perform YOLO pose detection
def torch_thread(model_name, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_np, run_signal, skeletons, ids

    print("Intializing Model...")

    script_path = os.path.dirname(os.path.realpath(__file__))
    models_path = script_path+'/../models/'
    model_path = models_path+model_name+'.pt'
    model_labels_path = models_path+model_name+'_labels.txt'

    # Download model if not found
    if not os.path.isfile(model_path):
        print("Model not found, downloading it...")

        # Get the PyTorch model
        model = YOLO(model_name+'.pt')  

        # Copy the models into the correct directory
        shutil.copy(model_name+'.pt', models_path+model_name+'.pt')
    
    print("Model initialized")

    # Load the model
    print("Model loading...") 
    model = YOLO(models_path+model_name+'.pt')
    print("Model loaded")
    
    while rospy.is_shutdown() is False:
        if run_signal:
            # Wait for an image to be grabbed from the zed camera
            lock.acquire()

            # Convert the image to RGB for YOLO
            img = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
            
            # Get the inference from the pose model with tracking enabled
            # Add show=True to display the pose estimation results (slows to ~2 Hz)
            results = model.track(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres, persist=True, tracker="bytetrack.yaml")
            
            # Save the results
            if results[0].boxes.id is None:
                ids = None
                skeletons = None
            else:
                ids = results[0].boxes.id.int().cpu().tolist()
                skeletons = results[0].cpu().numpy().keypoints.data

            # Return lock for result processing in main thread
            lock.release()
            run_signal = False
        sleep(0.001)

# Wrap skeleton data into ROS ObjectStamped message
def objects_wrapper(objects, labels):
    global zed_location
    
    ros_msg = zed_msgs.ObjectsStamped()
    ros_msg.header.stamp = rospy.Time.now()

    if zed_location == 'detached':
        ros_msg.header.frame_id = CAMERA_NAME
    elif zed_location == 'chairry':
        ros_msg.header.frame_id = CAMERA_NAME + "_left_camera_frame"

    obj_list = []
    if objects is None:
        ros_msg.objects = []
    else:
        for i in range(objects.shape[0]):
            person = objects[i]
            obj_msg = zed_msgs.Object()
            
            obj_msg.label_id = labels[i]
            obj_msg.position = [0,0,0]
            obj_msg.velocity = [0,0,0]          # TODO: figure out how to update velocity
            obj_msg.tracking_available = True   # TODO: figure out how to turn off tracking
            obj_msg.tracking_state = 1          # TODO: figure out how to implement this
            obj_msg.skeleton_available = True
            
            for j in range(person.shape[0]):
                keypoint = person[j]
                x = keypoint[0]
                y = keypoint[1]
                z = keypoint[2]
                conf = keypoint[3]
                obj_msg.skeleton_3d.keypoints[j].kp = [x, y, z, conf]
                print(obj_msg.skeleton_3d.keypoints[j].kp)
            
            obj_list.append(obj_msg)
            
    ros_msg.objects = obj_list
    return ros_msg  

# Wrap skeleton data into ROS PointCloud2 message for RViz Visualisation
def point_cloud_wrapper(ros_msg):
    # Create a header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = CAMERA_NAME + "_left_camera_frame"
    
    # Define the fields
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]
    
    # Set the color of the points to blue (arbitrary choice)
    r = 0
    g = 0
    b = 255
    color = (r << 16) | (g << 8) | b
    
    # Define the confidence threshold for keypoints
    #   This can be quite high-- when the keypoint is visible in 
    #   the image, the confidence is usually ~0.95 or higher
    kp_conf_thresh = 0.95

    # Display all valid keypoints
    points = []
    for obj_msg in ros_msg.objects:
        skeleton = obj_msg.skeleton_3d
        for keypoint in skeleton.keypoints:
            confidence = keypoint.kp[3]

            # Only add the keypoint if:
            # - the confidence is above the confidence threshold
            # - the keypoint is finite (not inf, -inf, nor NaN)
            # - the keypoint is not at the origin (sign of an error)
            if confidence > kp_conf_thresh and math.isfinite(keypoint.kp[0]) and keypoint.kp[0] != 0:
                # Convert point coordinates from millimeters to meters
                point = [keypoint.kp[0]/1000.0, keypoint.kp[1]/1000.0, keypoint.kp[2]/1000.0, color]
                points.append(point)
    
    point_cloud_msg = pc2.create_cloud(header, fields, points)
    return point_cloud_msg
    
# Transform skeleton data from local to map frame
# TODO: 
# - confirm if tfBuffer is needed
# - clean up code
# - add comments
def local_to_map_transform(msg, tfBuffer, frame):
    global zed_location
    try:
        if zed_location == 'chairry':
            #lct = tfBuffer.get_latest_common_time(str(frame), CAMERA_NAME + "_left_camera_frame")
            #transform = tfBuffer.lookup_transform(str(frame), CAMERA_NAME + "_left_camera_frame", lct, rospy.Duration(0.1))
            # lct = tfBuffer.get_latest_common_time("map", "zed2i_left_camera_frame")
            # transform = tfBuffer.lookup_transform("map", "zed2i_left_camera_frame", time=lct, timeout=rospy.Duration(0.1))
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "chairry_base_link"
            transform.child_frame_id = "zed2i_left_camera_frame"
            transform.transform.translation.x = -0.25
            transform.transform.translation.y = 0.0
            transform.transform.translation.z = 1.53
            quat = tf.transformations.quaternion_from_euler(0, 0.05, 0)
            transform.transform.rotation.x = quat[0]
            transform.transform.rotation.y = quat[1]
            transform.transform.rotation.z = quat[2]
            transform.transform.rotation.w = quat[3]



        for obj in msg.objects:
            p = PointStamped()
            p.point.x = obj.position[0]
            p.point.y = obj.position[1]
            p.point.z = obj.position[2]

            if zed_location == 'chairry':
                obj.position = tf2_geometry_msgs.do_transform_point(p, transform)
                obj.position = [obj.position.point.x, obj.position.point.y, obj.position.point.z]
            elif zed_location == 'detached':
                obj.position = [p.point.x, p.point.y, p.point.z]

            for corner in obj.bounding_box_3d.corners:
                p.point.x = corner.kp[0]
                p.point.y = corner.kp[1]
                p.point.z = corner.kp[2]

                if zed_location == 'chairry':    
                    corner.kp = tf2_geometry_msgs.do_transform_point(p, transform)
                    corner.kp = [corner.kp.point.x, corner.kp.point.y, corner.kp.point.z]
                elif zed_location == 'detached':
                    corner.kp = [p.point.x, p.point.y, p.point.z]
        
        if zed_location == 'chairry':
            msg.header.frame_id = frame
        elif zed_location == 'detached':
            msg.header.frame_id = CAMERA_NAME


    except tf2_ros.LookupException as e:
        print("Failed to transform object from local to", frame, "frame due to LookupException: ", e)
    except tf2_ros.ConnectivityException:
        print("Failed to transform object from local to", frame, "frame due to ConnectivityException")
    except tf2_ros.ExtrapolationException:
        print("Failed to transform object from local to", frame, "frame due to ExtrapolationException")

    return msg

# Receive data from zed camera, ingest YOLO pose detections, and publish the results
# TODO:
# - confirm if tfBuffer is needed
# - clean up code
# - add comments
def main():
    global image_np, run_signal, img_size, conf_thres, model_name, zed_location, skeletons, ids

    ######################
    ##### Publishers #####
    ######################
    # Publish the objects in the zed2i & chairry_base_link frames
    pub_z = rospy.Publisher(CAMERA_NAME+'/skeletons_zed2i', zed_msgs.ObjectsStamped, queue_size=50)   # zed2i frame
    pub_c = rospy.Publisher(CAMERA_NAME+'/skeletons_cbl', zed_msgs.ObjectsStamped, queue_size=50)     # chairry_base_link frame
    # Publish the point cloud in the zed2i_left_camera_frame frame
    pub_pc = rospy.Publisher(CAMERA_NAME+'/skeleton_point_cloud', PointCloud2, queue_size=10)
    
    # Confirm if this is needed
    # tfBuffer = tf2_ros.Buffer()

    ###################
    ##### Threads #####
    ###################
    # Start YOLO pose detection thread
    capture_thread = Thread(target=torch_thread, kwargs={'model_name': model_name, 'img_size': img_size, "conf_thres": conf_thres})
    capture_thread.start()


    #####################################
    ##### ZED Camera Initialisation #####
    #####################################
    # Initialize ZED camera
    print("Initializing Camera...")

    # Initialize parameters
    zed = sl.Camera()
    input_type = sl.InputType()
    runtime_params = sl.RuntimeParameters()
        
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA        # Use depth ultra mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER   # Use millimeter units (for depth measurements)
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance = 10000           # Set the maximum depth distance to 10 meters

    # Try opening the camera, print error code if unsuccessful
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    print("Initialized Camera")


    # positional_tracking_parameters = sl.PositionalTrackingParameters()
    # if zed_location == 'detached':
    #     # improve static performance and have boxes stuck to the ground
    #     positional_tracking_parameters.set_as_static = True
    # zed.enable_positional_tracking(positional_tracking_parameters)

    # obj_param = sl.ObjectDetectionParameters()
    # obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    # obj_param.enable_tracking = True
    # obj_param.filtering_mode = sl.OBJECT_FILTERING_MODE.NONE
    # zed.enable_object_detection(obj_param)

    # objects = sl.Objects()
    # obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # camera_infos = zed.get_camera_information()
    # camera_res = camera_infos.camera_configuration.resolution


    #######################################
    ##### Get, Ingest, & Publish Data #####
    #######################################

    # Initialise image and point cloud
    image_left_temp = sl.Mat()
    point_cloud = sl.Mat()

    while rospy.is_shutdown() is False:
        # Grab data from the camera
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Wait for previous iteration to finish
            lock.acquire()

            # Retrieve the image
            zed.retrieve_image(image_left_temp, sl.VIEW.LEFT)       # image_left_temp is a Mat object
            image_np = image_left_temp.get_data()                   # image_np is a numpy array (formatting for YOLO)
            
            # Retrieve the point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Release the lock for YOLO thread to use
            lock.release()
            run_signal = True
            while run_signal:
                sleep(0.001)

            # Wait for skeleton detections
            lock.acquire()
            
            # Ingest the skeletons and their IDs
            objects, labels = ingest_skeletons(skeletons, ids, point_cloud)
        
            # Release the lock for the next iteration
            lock.release()

            # Publish skeletons in ROS as a custom ObjectsStamped message
            ros_msg = objects_wrapper(objects, labels)
            pub_z.publish(ros_msg)
            # pub_c.publish(local_to_map_transform(ros_msg, tfBuffer, "chairry_base_link"))
            
            # Publish a point cloud with all keypoints for visualisation
            pc_msg = point_cloud_wrapper(ros_msg)
            pub_pc.publish(pc_msg)
    
    # Close the camera when the node is shutdown
    zed.close()


if __name__ == '__main__':
    # Initialize the node
    rospy.init_node("zed_yolo_ros", anonymous=False)
    rospy.loginfo("ZED YOLO node started")
    
    # Read parameters from the parameter server
    namespace = rospy.get_name() + "/"
    model_name = rospy.get_param(namespace + 'model_name', 'yolov8m-pose')
    img_size = rospy.get_param(namespace + 'img_size', 416)
    conf_thres = rospy.get_param(namespace + 'conf_thres', 0.4)
    zed_location = rospy.get_param(namespace + 'zed_location', 'detached')
    rospy.loginfo(f"Model Name: {model_name}")
    rospy.loginfo(f"Image Size: {img_size}")
    rospy.loginfo(f"Confidence Threshold: {conf_thres}")
    rospy.loginfo(f"ZED Location: {zed_location}")

    with torch.no_grad():
        main()
