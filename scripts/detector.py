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

# Define the confidence threshold for keypoints
    #   This can be quite high-- when the keypoint is visible in 
    #   the image, the confidence is usually ~0.95 or higher
kp_conf_thresh = 0.95

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
# TODO:
# - find current position
# - find current velocity
# - find tracking state
# - update tracking state
def objects_wrapper(objects, labels, pose_history):
    global zed_location, kp_conf_thresh
    
    ros_msg = zed_msgs.ObjectsStamped()
    ros_msg.header.stamp = rospy.Time.now()

    if zed_location == 'detached':
        ros_msg.header.frame_id = CAMERA_NAME
    elif zed_location == 'chairry':
        ros_msg.header.frame_id = CAMERA_NAME + "_left_camera_frame"

    # Clean pose history of any ids that are no longer present
    for i in range(len(pose_history)):
        id = pose_history[i][0]
        if id not in labels:
            # Remove the id's entry from the pose history
            pose_history.pop(i)

    obj_list = []
    if objects is None:
        ros_msg.objects = []
    else:
        for i in range(objects.shape[0]):
            person = objects[i]
            obj_msg = zed_msgs.Object()
            obj_msg.label_id = labels[i]

            
            position = [0,0,0]
            for j in range(person.shape[0]):
                keypoint = person[j]
                x = keypoint[0]
                y = keypoint[1]
                z = keypoint[2]
                conf = keypoint[3]
                obj_msg.skeleton_3d.keypoints[j].kp = [x, y, z, conf]
                # print(obj_msg.skeleton_3d.keypoints[j].kp)

                # Sum the average position
                # Only add the keypoint if:
                # - the confidence is above the confidence threshold
                # - the keypoint is finite (not inf, -inf, nor NaN)
                # - the keypoint is not at the origin (sign of an error)
                num_kps = 0
                if conf > kp_conf_thresh and math.isfinite(keypoint.kp[0]) and keypoint.kp[0] != 0:
                    position[0] += x
                    position[1] += y
                    position[2] += z
                    num_kps +=1

            # Calculate the average position
            obj_msg.position = [position[0]/num_kps, position[1]/num_kps, position[2]/num_kps]

            # Get index of the person in the pose history
            # pose_history = [[id, [x, y, z]], [id, [x, y, z]], ...]
            id = labels[i]
            if id in pose_history:
                index = pose_history.index(id)
                old_pose = pose_history[index][1]
                new_pose = obj_msg.position
                # Calculate velocity
                obj_msg.velocity = [new_pose[0] - old_pose[0], new_pose[1] - old_pose[1], new_pose[2] - old_pose[2]]

                # Save new pose in pose history at same index (replace old pose)
                pose_history[index] = [id, new_pose]

            obj_list.append(obj_msg)
            
    ros_msg.objects = obj_list
    return ros_msg, pose_history  

# Wrap skeleton data into ROS PointCloud2 message for RViz Visualisation
def point_cloud_wrapper(ros_msg):
    global kp_conf_thresh

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
    
    # Set the color of the keypoints to blue (arbitrary choice)
    r = 0
    g = 0
    b = 255
    blue = (r << 16) | (g << 8) | b

     # Set the color of the position to black (arbitrary choice)
    r = 255
    g = 255
    b = 255
    black = (r << 16) | (g << 8) | b

    # Display all valid keypoints & skeleton's position
    points = []
    for obj_msg in ros_msg.objects:
        # Display the keypoints of the skeleton
        skeleton = obj_msg.skeleton_3d
        for keypoint in skeleton.keypoints:
            confidence = keypoint.kp[3]

            # Only add the keypoint if:
            # - the confidence is above the confidence threshold
            # - the keypoint is finite (not inf, -inf, nor NaN)
            # - the keypoint is not at the origin (sign of an error)
            if confidence > kp_conf_thresh and math.isfinite(keypoint.kp[0]) and keypoint.kp[0] != 0:
                # Convert point coordinates from millimeters to meters
                point = [keypoint.kp[0]/1000.0, keypoint.kp[1]/1000.0, keypoint.kp[2]/1000.0, blue]
                points.append(point)

        # Display the position of the skeleton
        pose = obj_msg.position
        point = [pose[0], pose[1], pose[2], black]
    
    point_cloud_msg = pc2.create_cloud(header, fields, points)
    return point_cloud_msg
    
# Transform skeleton data from local to map frame
def local_to_map_transform(ros_msg):
    try:
        # Create a transform from chairry_base_link to zed2i_left_camera_frame
        # Hard-coded because tfBuffer wasn't working
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "chairry_base_link"
        transform.child_frame_id = "zed2i_left_camera_frame"
        transform.transform.translation.x = -0.25
        transform.transform.translation.y = 0.25
        transform.transform.translation.z = 1.53
        quat = tf.transformations.quaternion_from_euler(0, 0.05, 0)
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]

        for obj_msg in ros_msg.objects:
            # Transform position
            orig_pose = PointStamped()
            orig_pose.point.x = obj_msg.position[0]
            orig_pose.point.y = obj_msg.position[1]
            orig_pose.point.z = obj_msg.position[2]

            new_pose = tf2_geometry_msgs.do_transform_point(orig_pose, transform)
            obj_msg.position = [new_pose.point.x, new_pose.point.y, new_pose.point.z]

            # Transform velocity
            orig_vel = PointStamped()
            orig_vel.point.x = obj_msg.velocity[0]
            orig_vel.point.y = obj_msg.velocity[1]
            orig_vel.point.z = obj_msg.velocity[2]

            new_vel = tf2_geometry_msgs.do_transform_point(orig_vel, transform)
            obj_msg.velocity = [new_vel.point.x, new_vel.point.y, new_vel.point.z]

            # Transform keypoints
            skeleton = obj_msg.skeleton_3d
            for keypoint in skeleton.keypoints:
                orig_kp = PointStamped()
                orig_kp.point.x = keypoint.kp[0]
                orig_kp.point.y = keypoint.kp[1]
                orig_kp.point.z = keypoint.kp[2]
                conf = keypoint.kp[3]

                new_point = tf2_geometry_msgs.do_transform_point(orig_kp, transform)
                keypoint.kp = [new_point.point.x, new_point.point.y, new_point.point.z, conf]

        # Update the frame_id
        ros_msg.header.frame_id = "chairry_base_link"  

    # Handle errors
    except tf2_ros.LookupException or tf2_ros.ConnectivityException or tf2_ros.ExtrapolationException as e:
        print("Failed to transform object to chairry_base_link frame due to Exception: ", e)

    return ros_msg

# Receive data from zed camera, ingest YOLO pose detections, and publish the results
def main():
    global image_np, run_signal, img_size, conf_thres, model_name, zed_location, skeletons, ids

    ######################
    ##### Publishers #####
    ######################
    # Publish the objects in the zed2i & chairry_base_link frames
    pub_z = rospy.Publisher(CAMERA_NAME+'/skeletons/objects/z', zed_msgs.ObjectsStamped, queue_size=50)   # zed2i frame
    pub_c = rospy.Publisher(CAMERA_NAME+'/skeletons/objects/c', zed_msgs.ObjectsStamped, queue_size=50)     # chairry_base_link frame
    # Publish the point cloud in the zed2i_left_camera_frame frame
    pub_pc_z = rospy.Publisher(CAMERA_NAME+'/skeletons/point_cloud/z', PointCloud2, queue_size=10)
    pub_pc_c = rospy.Publisher(CAMERA_NAME+'/skeletons/point_cloud/c', PointCloud2, queue_size=10)

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


    #######################################
    ##### Get, Ingest, & Publish Data #####
    #######################################

    # Initialise image, point cloud, and pose history
    image_left_temp = sl.Mat()
    point_cloud = sl.Mat()
    pose_history = []

    while rospy.is_shutdown() is False:
        # Grab data from the camera
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            ####################
            ##### Get Data #####
            ####################
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


            #######################
            ##### Ingest Data #####
            #######################
            # Wait for skeleton detections
            lock.acquire()
            
            # Ingest the skeletons and their IDs
            objects, labels = ingest_skeletons(skeletons, ids, point_cloud)
        
            # Release the lock for the next iteration
            lock.release()


            ########################
            ##### Publish Data #####
            ########################
            # Publish skeletons in ROS as a custom ObjectsStamped message
            ros_msg, pose_history = objects_wrapper(objects, labels, pose_history)
            pub_z.publish(ros_msg)
            # Publish a point cloud with all keypoints for visualisation
            pc_msg = point_cloud_wrapper(ros_msg)
            pub_pc_z.publish(pc_msg)

            # Transform the skeletons to the chairry_base_link frame
            if zed_location == 'chairry':
                # Publish skeletons in ROS as a custom ObjectsStamped message
                transform_msg = local_to_map_transform(ros_msg)
                pub_c.publish(transform_msg)

                # Publish a point cloud with all keypoints for visualisation
                pc_msg = point_cloud_wrapper(transform_msg)
                pub_pc_c.publish(pc_msg)
            
            
    
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
