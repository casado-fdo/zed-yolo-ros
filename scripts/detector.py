#!/usr/bin/env python3

'''
This script performs YOLO pose detection on images from the ZED camera.
It currently runs at approximately 16 Hz.

'''


import rospy
import math
import numpy as np
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
import ultralytics
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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from people_msgs.msg import People, Person
from nav_msgs.msg import Path
import datetime


lock = Lock()
run_signal = False
img_size = 416
conf_thres = 0.4
model_name = 'yolov8m-pose'
zed_location = 'detached'
skeletons = None
window_size = 4

CAMERA_NAME = "zed2i"

# Define the confidence threshold for keypoints
    #   This can be quite high-- when the keypoint is visible in 
    #   the image, the confidence is usually ~0.95 or higher
kp_conf_thresh = 0.95

# Convert 2-D Pose keypoints to 3-D points
def ingest_skeletons(skeletons, labels, point_cloud, transform):

    # Define the frame based on the zed location
    if zed_location == 'chairry':
        frame = "odom"
    else:
        frame = CAMERA_NAME + "_left_camera_frame"

    # Check to see if there are any skeletons
    if skeletons is None:
        people = None
        labels = []
        return people, labels, frame
    
    # Create a list of people, each with 17 4-d keypoints
    num_people = skeletons.shape[0]
    people = np.zeros((num_people, 17, 4))

    # Define the connections between keypoints (see zed_yolo_ros/msg/Skeleton3D)
    connections = [ 
        (0,1), (1,3), 
        (0,2), (2,4),
        (0,5), (5,7), (7,9), 
        (0,6), (6,8), (8,10), 
        (5,11), (11,13), (13,15),
        (6,12), (12,14), (14,16),
        (5,6), (11,12) # connect shoulders and hips 
    ] # 18 connections here
    
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
            
            # Save the 3D point and confidence score, converting to meters
            point_3d = [point_3d[0]/1000.0, point_3d[1]/1000.0, point_3d[2]/1000.0]

            # If the zed is on chairry, transform the 3D point into the odom frame
            if zed_location == 'chairry':
                point_3d = transform_point(point_3d, CAMERA_NAME + "_left_camera_frame", transform)

            point_3d.append(conf)
            people[i][j] = point_3d

        # Iterate through connections to ensure no "bones" are too long
        for start_idx, end_idx in connections:
            start = people[i][start_idx]
            end = people[i][end_idx]
            distance = np.linalg.norm([start[0] - end[0], start[1] - end[1], start[2] - end[2]])
            # If any bone is greater than 1 meter long, set the confidence of the keypoints to 0
            if distance > 1:
                people[i][start_idx][3] = 0
                people[i][end_idx][3] = 0

    # Return the people and their IDs/labels, as well as the frame
    return people, labels, frame

def transform_point(point, original_frame, transform):
    # Create a PointStamped for the position
    point_stamped = tf2_geometry_msgs.PointStamped()
    point_stamped.header.stamp = rospy.Time.now()
    point_stamped.header.frame_id = original_frame
    point_stamped.point.x = point[0]
    point_stamped.point.y = point[1]
    point_stamped.point.z = point[2]

    try:
        # Apply the transform to the point
        transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)

        return [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]

    except tf2_ros.LookupException as e:
        rospy.logwarn(f"Transform lookup failed: {e}")
        
    except tf2_ros.ExtrapolationException as e:
        rospy.logwarn(f"Transform extrapolation failed: {e}")

# Perform YOLO pose detection
def torch_thread(model_name, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_np, run_signal, skeletons, ids

    print("Intializing Model...")

    script_path = os.path.dirname(os.path.realpath(__file__))
    models_path = script_path+'/../models/'
    model_path = models_path+model_name+'.pt'

    # Download model if not found
    if not os.path.isfile(model_path):
        print("Model not found, downloading it...")

        # Get the PyTorch model
        model = YOLO(model_name+'.pt')  

        # Copy the models into the correct directory
        shutil.copy(model_name+'.pt', model_path)
    
    print("Model initialized")

    # Load the model
    print("Model loading...") 
    model = YOLO(model_path)
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
def objects_wrapper(objects, labels, pos_history, vel_history, frame):
    global zed_location, kp_conf_thresh, window_size
    
    # Create a ROS message
    ros_msg = zed_msgs.ObjectsStamped()
    ros_msg.header.stamp = rospy.Time.now()
    ros_msg.header.frame_id = frame
    obj_list = []   # List for storing each person's object message

    # Prepare the histories, removing old data
    pos_history, vel_history, removed_ids = prep_histories(labels, pos_history, vel_history) 
    
    # If there are no detections, return an empty list
    if objects is None:
        ros_msg.objects = obj_list
        
    # Otherwise, create an object message for each person
    else:
        for i in range(objects.shape[0]):
            person = objects[i]
            obj_msg = zed_msgs.Object()
            obj_msg.label_id = labels[i]

            # Initialisations
            position = [0,0,0]
            num_kps = 0
            
            # Iterate through the person's keypoints
            for j in range(person.shape[0]):
                keypoint = person[j]
                x = keypoint[0]
                y = keypoint[1]
                z = keypoint[2]
                conf = keypoint[3]

                # Find VALID keypoints:
                #   - the confidence is above the confidence threshold
                #       - keypoint is not an outlier (checked in ingest_skeletons)
                #       - the model is pretty confident about it (usually > 0.95)
                #   - the keypoint is finite (not inf, -inf, nor NaN)
                #   - the keypoint is not at the origin (sign of an error)
                if conf > kp_conf_thresh and math.isfinite(keypoint[0]) and keypoint[0] != 0:
                    # Store the 3d keypoint data (for RViz visualisation)
                    obj_msg.skeleton_3d.keypoints[j].kp = [x, y, z, conf]

                    # Calculate the position of the person using  
                    # the average of the valid keypoints from the 
                    # shoulders up & hips -> more stable
                    if j <= 6 or j == 11 or j == 12:
                        position[0] += x
                        position[1] += y
                        position[2] += z
                        num_kps += 1 
            # Calculate the position of the person: average of valid keypoints from shoulders up & hips
            if num_kps > 0:
                position = [position[0]/num_kps, position[1]/num_kps, position[2]/num_kps]
                obj_msg.position = position
                print("object position: ", position)
            else:
                # No valid keypoints: skip this person
                continue  

            # Transform position into odom frame if we're on chairry (currently in zed2i frame)
            # if zed_location == 'chairry':
            #     position = transform_point(position, CAMERA_NAME + "_left_camera_frame", "odom", tf_buffer)
            #     obj_msg.position = position
            # else:
            #     obj_msg.position = position


            # Get indices of the person in the position and velocity histories
            id = labels[i]
            pos_index, vel_index = get_indices(id, pos_history, vel_history)
            
            # print("id: ", id, "pos_index: ", pos_index, "vel_index: ", vel_index)
            # If the person has a valid current position & has been detected before
            # Then update the position and velocity histories accordingly
            if num_kps > 0 and pos_index != -1 and vel_index != -1:
                # Update the position and velocity histories & find average velocity
                pos_history, vel_history, averaged_velocity = update_pos_vel(pos_history, vel_history, obj_msg, pos_index, vel_index, id)

                # Set the velocity of the person
                obj_msg.velocity = averaged_velocity
                
            # If the person has a valid current position and has no position history
            # Then add the person to the position history
            if num_kps > 0 and pos_index == -1:
                # Add new position to position history
                position = obj_msg.position
                pos_history.append([id, [position[0], position[1], position[2], rospy.Time.now().to_sec()]])
                
            # If the person has a valid current position and has no vel history
            # Then add the person to the vel history
            if num_kps > 0 and vel_index == -1:
                # Add new velocity to velocity history
                vel_history.append([id, [[0,0,0]]])

            # Append the object message to the list
            obj_list.append(obj_msg)
    
    # Update the object list in the ROS message    
    ros_msg.objects = obj_list
    
    # Return the ROS message and the updated position and velocity histories
    return ros_msg, pos_history, vel_history, removed_ids

def people_wrapper(ros_msg):
    people_msg = People()
    people_msg.header.stamp = ros_msg.header.stamp
    people_msg.header.seq = ros_msg.header.seq
    people_msg.header.frame_id = ros_msg.header.frame_id
    people_msg.people = []

    for obj_msg in ros_msg.objects:
        person = Person()
        person.name = str(obj_msg.label_id)
        person.position.x = obj_msg.position[0]
        person.position.y = obj_msg.position[1]
        person.position.z = obj_msg.position[2]
        person.velocity.x = obj_msg.velocity[0]
        person.velocity.y = obj_msg.velocity[1]
        person.velocity.z = obj_msg.velocity[2]
        print("person position: ", person.position)

        people_msg.people.append(person)

    return people_msg


# Prepare the histories, removing old data
def prep_histories(labels, pos_history, vel_history):
    global window_size
    
    # Format of pos_history and vel_history (assuming window_size = 3):
    # pos_history = [[id, [x, y, z, t]], [id, [x, y, z, t]], ...]
    # vel_history = [[id, [[x, y, z], [x, y, z], [x, y, z]]], [id, [[x, y, z], [x, y, z], [x, y, z]]], ...]
    
    # Clean position history of any ids that are no longer present
    clean_history = []
    removed_labels = []
    for i in range(len(pos_history)):
        id = pos_history[i][0]
        if id in labels:
            clean_history.append(pos_history[i])
        else:
            removed_labels.append(id)
    pos_history = clean_history
    
    # Clean velocity history of any ids that are no longer present (for all of window size)
    i = 0
    while i < len(vel_history):
        remove = False
        if len(vel_history[i][1]) == window_size:
            remove = all(vel == [0, 0, 0] for vel in vel_history[i][1])
        if remove:
            vel_history.pop(i)
        else:
            i += 1
    
    # Add zero velocities to velocity history for ids that are no longer present
    for entry in vel_history:
        id = entry[0]
        if id not in labels:
            entry[1].append([0, 0, 0])
            if len(entry[1]) > window_size:
                entry[1].pop(0)
    
    return pos_history, vel_history, removed_labels

# Find the index of the person in the position and velocity histories
def get_indices(id, pos_history, vel_history):
    pos_index = -1
    vel_index = -1
    for i in range(len(pos_history)):
        if id == pos_history[i][0]:
            pos_index = i
            break
    for i in range(len(vel_history)):
        if id == vel_history[i][0]:
            vel_index = i
            break
    return pos_index, vel_index

# Update the position and velocity histories & find average velocity
def update_pos_vel(pos_history, vel_history, obj_msg, pos_index, vel_index, id):
    # - replace the old position with the new position in position history
    # - calculate the current velocity of the person
    # - save the current velocity in the velocity history
    # - average the velocity over the num_vels most recent frames (num_vels <= window_size)
    
    # Format of pos_history and vel_history (assuming window_size = 3):
    # pos_history = [[id, [x, y, z, t]], [id, [x, y, z, t]], ...]
    # vel_history = [[id, [[x, y, z], [x, y, z], [x, y, z]]], [id, [[x, y, z], [x, y, z], [x, y, z]]], ...]

    # Prepare variables for velocity calculation & position history update
    # Save the time in seconds for velocity calculation
    old_pos = pos_history[pos_index][1]
    position = obj_msg.position
    new_pos = [position[0], position[1], position[2], rospy.Time.now().to_sec()]
    delta_time = new_pos[3] - old_pos[3]
    
    # Replace the old position with the new position in the position history
    pos_history[pos_index] = [id, new_pos]
    
    # Calculate velocity & save the current velocity in the velocity history
    current_velocity = [(new_pos[0] - old_pos[0])/delta_time, (new_pos[1] - old_pos[1])/delta_time, (new_pos[2] - old_pos[2])/delta_time]
    vel_history[vel_index][1].append(current_velocity)

    # Remove the oldest velocity if the window size is exceeded
    if len(vel_history[vel_index][1]) > window_size:
        vel_history[vel_index][1].pop(0) 

    # Find the average the velocity over the num_vels most recent frames to filter out noise
    # (num_vels = number of frames in the velocity history, which is <= window_size)
    recent_velocities = vel_history[vel_index][1]
    averaged_velocity = [0, 0, 0]
    num_vels = len(recent_velocities)
    for i in range(num_vels):
        averaged_velocity[0] += recent_velocities[i][0]
        averaged_velocity[1] += recent_velocities[i][1]
        averaged_velocity[2] += recent_velocities[i][2]
    averaged_velocity[0] = averaged_velocity[0]/num_vels
    averaged_velocity[1] = averaged_velocity[1]/num_vels
    averaged_velocity[2] = averaged_velocity[2]/num_vels
    
    return pos_history, vel_history, averaged_velocity

# Wrap skeleton data into ROS PointCloud2 message for RViz Visualisation
# TODO:
# - fix frame issue
def point_cloud_wrapper(ros_msg):
    global kp_conf_thresh, zed_location

    # Create a header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = ros_msg.header.frame_id
    
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

     # Set the color of the position to white (arbitrary choice)
    r = 255
    g = 255
    b = 255
    white = (r << 16) | (g << 8) | b

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
                # Transform the keypoint into the map frame
                point = [keypoint.kp[0], keypoint.kp[1], keypoint.kp[2]]
                point.append(blue)
                points.append(point)

        # Display the position of the skeleton
        pos = obj_msg.position
        if pos[0] != 0:
            point = [pos[0], pos[1], pos[2], white]
            points.append(point)
    
    point_cloud_msg = pc2.create_cloud(header, fields, points)
    return point_cloud_msg

# Populate a ROS MarkerArray message for each person's velocity using velocity_wrapper()
def bone_array_wrapper(ros_msg, removed_ids):
    global kp_conf_thresh
    frame = ros_msg.header.frame_id
    connections = [ 
        (0,1), (1,3), 
        (0,2), (2,4),
        (0,5), (5,7), (7,9), 
        (0,6), (6,8), (8,10), 
        (5,11), (11,13), (13,15),
        (6,12), (12,14), (14,16),
        (5,6), (11,12)
    ] # 18 connections here
    bone_array = MarkerArray()
    bone_array.markers = []
    if ros_msg.objects == 0:
        bone = bone_wrapper(0, 0, [0,0,0], [0,0,0], [0,0,0,1], frame, Marker.DELETEALL)
        bone_array.markers.append(bone)
    else:
        for obj_msg in ros_msg.objects:
            person_id = obj_msg.label_id
            skeleton = obj_msg.skeleton_3d
            for connection_idx, (start_idx, end_idx) in enumerate(connections):
                start = skeleton.keypoints[start_idx].kp
                end = skeleton.keypoints[end_idx].kp
                if start[3] > kp_conf_thresh and end[3] > kp_conf_thresh and start[0] != 0 and end[0] != 0 and math.isfinite(start[0]) and math.isfinite(end[0]):
                    start_point = start
                    end_point = end 
                    bone = bone_wrapper(person_id, connection_idx, start_point, end_point, [0,0,1,1], frame, Marker.ADD)
                    bone_array.markers.append(bone)
                else:
                    bone = bone_wrapper(person_id, connection_idx, [0,0,0], [0,0,0], [1,0,0,1], frame, Marker.DELETE)
                    bone_array.markers.append(bone)
        for person_id in removed_ids:
            for connection_idx, _ in enumerate(connections):
                bone = bone_wrapper(person_id, connection_idx, [0,0,0], [0,0,0], [1,1,1,1], frame, Marker.DELETE)
                bone_array.markers.append(bone)
    return bone_array

def bone_wrapper(person_id, connection_idx, start_point, end_point, color, frame, action):
    bone = Marker()
    bone.header.frame_id = frame
    bone.header.stamp = rospy.Time.now()
    bone.ns = "bones_" + str(person_id)
    bone.id = connection_idx
    bone.type = Marker.LINE_LIST
    bone.action = action

    # Set the scale of the bone
    bone.scale.x = 0.03  # line width

    # Set the color (red, green, blue, alpha)
    bone.color.r = color[0]
    bone.color.g = color[1]
    bone.color.b = color[2]
    bone.color.a = color[3]

    # Set the start and end points of the bone
    start = Point()
    start.x = start_point[0]
    start.y = start_point[1]
    start.z = start_point[2]

    end = Point()
    end.x = end_point[0]
    end.y = end_point[1]
    end.z = end_point[2]

    bone.points.append(start)
    bone.points.append(end)
    
    # Set the orientation of the bone
    bone.pose.orientation = Quaternion()
    bone.pose.orientation.x = 0.0
    bone.pose.orientation.y = 0.0
    bone.pose.orientation.z = 0.0
    bone.pose.orientation.w = 1.0

    return bone

# Populate a ROS MarkerArray message for each person's velocity using velocity_wrapper()
def velocity_array_wrapper(ros_msg, removed_ids):
    frame = ros_msg.header.frame_id
    velocity_array = MarkerArray()
    velocity_array.markers = []
    if ros_msg.objects == []:
        vel = velocity_wrapper(0, [0,0,0], [0,0,0], [0,0,0,1], frame, Marker.DELETEALL)
        velocity_array.markers.append(vel)
    else:
        for obj_msg in ros_msg.objects:
            person_id = obj_msg.label_id
            person_vel = obj_msg.velocity
            person_pos = obj_msg.position
            vel = velocity_wrapper(person_id, person_vel, person_pos, [0,0,1,1], frame, Marker.ADD) 
            velocity_array.markers.append(vel)
        for person_id in removed_ids:
            vel = velocity_wrapper(person_id, [0,0,0], [0,0,0], [1,1,1,1], frame, Marker.DELETE)
            velocity_array.markers.append(vel)
    return velocity_array
    
# Wrap velocity data for each person into a ROS Marker message
def velocity_wrapper(person_id, person_vel, person_pos, color, frame, action):
    velocity = Marker()
    velocity.header.frame_id = frame
    velocity.header.stamp = rospy.Time.now()
    velocity.ns = "vel_" + str(person_id)
    velocity.id = person_id
    velocity.type = Marker.ARROW
    velocity.action = action

    # Set the scale of the arrow
    velocity.scale.x = 0.1  # shaft diameter
    velocity.scale.y = 0.1  # head diameter
    velocity.scale.z = 0.1  # head length

    # Set the color (red, green, blue, alpha)
    velocity.color.r = color[0]
    velocity.color.g = color[1]
    velocity.color.b = color[2]
    velocity.color.a = color[3]

    # Set the start and end points of the arrow
    vel = person_vel
    pos = person_pos
    end = [pos[0] + vel[0], pos[1] + vel[1], pos[2] + vel[2]]
    
    start_point = Point()
    start_point.x = pos[0]
    start_point.y = pos[1]
    start_point.z = pos[2]

    end_point = Point()
    end_point.x = end[0]
    end_point.y = end[1]
    end_point.z = end[2] 

    velocity.points.append(start_point)
    velocity.points.append(end_point)
    
    # Set the orientation of the arrow
    velocity.pose.orientation = Quaternion()
    velocity.pose.orientation.x = 0.0
    velocity.pose.orientation.y = 0.0
    velocity.pose.orientation.z = 0.0
    velocity.pose.orientation.w = 1.0

    return velocity

 # populate a ROS MarkerArray message for each person's velocity using velocity_wrapper()
def path_array_wrapper(ros_msg, unique_id):
    frame = ros_msg.header.frame_id
    path_array = MarkerArray()
    path_array.markers = []
    for obj_msg in ros_msg.objects:
        person_id = obj_msg.label_id
        person_vel = obj_msg.velocity
        tol = 0.05
        vel_mag = np.linalg.norm(person_vel)
        person_pos = obj_msg.position
        if vel_mag <= tol or not math.isfinite(person_vel[0]) or not math.isfinite(person_pos[0]):
            continue
        rainbow_colors = [
            (255, 0, 0),    # Red
            (255, 127, 0),  # Orange
            (255, 255, 0),  # Yellow
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (75, 0, 130),   # Indigo
            (148, 0, 211)   # Violet
        ]
        color = rainbow_colors[(person_id - 1) % len(rainbow_colors)]
        person_color = [color[0]/255, color[1]/255, color[2]/255, 1]
        vel = path_wrapper(person_id, unique_id, person_vel, person_pos, person_color, frame, Marker.ADD) 
        path_array.markers.append(vel)
    return path_array
    
# Wrap velocity data for each person into a ROS Marker message
def path_wrapper(person_id, unique_id, person_vel, person_pos, color, frame, action):
    path = Marker()
    path.header.frame_id = frame
    path.header.stamp = rospy.Time.now()
    path.ns = "path_" + str(person_id)
    path.id = unique_id
    path.type = Marker.ARROW
    path.action = action

    # Set the scale of the arrow
    path.scale.x = 0.1  # shaft diameter
    path.scale.y = 0.1  # head diameter
    path.scale.z = 0.1  # head length

    # Set the color (red, green, blue, alpha)
    path.color.r = color[0]
    path.color.g = color[1]
    path.color.b = color[2]
    path.color.a = color[3]

    # Set the start and end points of the arrow
    vel = person_vel
    # scale vel to be 0.1 meters long
    length = np.linalg.norm(vel)/0.1
    vel = [vel[0]/length, vel[1]/length, vel[2]/length]

    pos = person_pos
    end = [pos[0] + vel[0], pos[1] + vel[1], pos[2] + vel[2]]
    
    start_point = Point()
    start_point.x = pos[0]
    start_point.y = pos[1]
    start_point.z = 0.0 

    end_point = Point()
    end_point.x = end[0]
    end_point.y = end[1]
    end_point.z = 0.0 

    path.points.append(start_point)
    path.points.append(end_point)
    
    # Set the orientation of the arrow
    path.pose.orientation = Quaternion()
    path.pose.orientation.x = 0.0
    path.pose.orientation.y = 0.0
    path.pose.orientation.z = 0.0
    path.pose.orientation.w = 1.0

    return path

# Receive data from zed camera, ingest YOLO pose detections, and publish the results
def main():
    global image_np, run_signal, img_size, conf_thres, model_name, zed_location, skeletons, ids

    ######################
    ##### Publishers #####
    ######################
    ###### USEFUL DATA ######
    # Publish the people
    pub_people = rospy.Publisher('/people', People, queue_size=50)
    # Publish the objects
    pub_objects = rospy.Publisher(CAMERA_NAME+'/objects', zed_msgs.ObjectsStamped, queue_size=50)

    ###### VISUALISATION ######
    # Publish the bones
    pub_bones = rospy.Publisher(CAMERA_NAME+'/viz/bones', MarkerArray, queue_size=10)
    # Publish the point clouds of keypoints
    pub_pc = rospy.Publisher(CAMERA_NAME+'/viz/point_cloud', PointCloud2, queue_size=10)
    # Publish the velocity markers
    pub_vels = rospy.Publisher(CAMERA_NAME+'/viz/velocity_markers', MarkerArray, queue_size=10)
    # Publish the path of the person
    pub_path = rospy.Publisher(CAMERA_NAME+'/viz/path', MarkerArray, queue_size=10)

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

    # Initialise image, point cloud, and position history
    image_left_temp = sl.Mat()
    point_cloud = sl.Mat()
    pos_history = []
    vel_history = []
    unique_id = 0

    # Initialize the tf_buffer and TransformListener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)


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

            #########################
            ##### Get Transform #####
            #########################
            if zed_location == 'chairry':
                # Get the latest common time
                common_time = tf_buffer.get_latest_common_time("chairry_base_link", CAMERA_NAME + "_left_camera_frame")

                # Get the transform using the common time
                transform = tf_buffer.lookup_transform(
                    "odom", 
                    CAMERA_NAME + "_left_camera_frame", 
                    common_time,  # Use the latest common time
                    rospy.Duration(1.0)  # Timeout duration
                )
            else:
                transform = None


            #######################
            ##### Ingest Data #####
            #######################
            # Wait for skeleton detections
            lock.acquire()
            
            # Ingest the skeletons and their IDs
            objects, labels, frame = ingest_skeletons(skeletons, ids, point_cloud, transform)
        
            # Release the lock for the next iteration
            lock.release()


            ########################
            ##### Publish Data #####
            ########################

            ###### USEFUL DATA ######
            # Publish skeletons in ROS as a custom ObjectsStamped message for social_navigator
            ros_msg, pos_history, vel_history, removed_ids = objects_wrapper(objects, labels, pos_history, vel_history, frame)
            pub_objects.publish(ros_msg)
            
            # Publish the people for move_base
            people_msg = people_wrapper(ros_msg)
            pub_people.publish(people_msg)

            ###### VISUALISAITON ######
            # Publish a 3-D point cloud with all valid keypoints
            pc_msg = point_cloud_wrapper(ros_msg)
            pub_pc.publish(pc_msg)

            # Publish the bones
            bone_msg = bone_array_wrapper(ros_msg, removed_ids)
            pub_bones.publish(bone_msg)  
                      
            # Publish velocity markers
            markers_msg = velocity_array_wrapper(ros_msg, removed_ids)
            pub_vels.publish(markers_msg)

            # Publish the path of the person
            path_msg = path_array_wrapper(ros_msg, unique_id)
            pub_path.publish(path_msg)
            unique_id += 1

          
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
    conf_thres = rospy.get_param(namespace + 'conf_thres', 0.8)
    zed_location = rospy.get_param(namespace + 'zed_location', 'detached')
    rospy.loginfo(f"Model Name: {model_name}")
    rospy.loginfo(f"Image Size: {img_size}")
    rospy.loginfo(f"Confidence Threshold: {conf_thres}")
    rospy.loginfo(f"ZED Location: {zed_location}")

    with torch.no_grad():
        main()
