#!/usr/bin/env python3

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
from geometry_msgs.msg import Point, Quaternion
from people_msgs.msg import People, Person


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
def ingest_skeletons(skeletons, labels, point_cloud):
    # Check to see if there are any skeletons
    if skeletons is None:
        people = None
        labels = []
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
            
            # Save the 3D point and confidence score, converting to meters
            people[i, j] = [point_3d[0]/1000.0, point_3d[1]/1000.0, point_3d[2]/1000.0, conf]

    # Return the people and their IDs/labels
    return people, labels

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
# TODO:
# - check frame id is ok on chairry
# - add a check for outliers?
def objects_wrapper(objects, labels, pose_history, vel_history):
    global zed_location, kp_conf_thresh, window_size
    
    # Create a ROS message
    ros_msg = zed_msgs.ObjectsStamped()
    ros_msg.header.stamp = rospy.Time.now()
    ros_msg.header.frame_id = CAMERA_NAME + "_left_camera_frame"
    obj_list = []   # List for storing each person's object message

    # Prepare the histories, removing old data
    pose_history, vel_history, removed_ids = prep_histories(labels, pose_history, vel_history) 
    
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
                
                # Store the 3d keypoint data (for RViz visualisation)
                x = keypoint[0]
                y = keypoint[1]
                z = keypoint[2]
                conf = keypoint[3]
                obj_msg.skeleton_3d.keypoints[j].kp = [x, y, z, conf]

                # Find the person's position by first summing the VALID keypoints
                #  '--> Keypoints are valid if:
                #       - the confidence is above the confidence threshold
                #       - the keypoint is finite (not inf, -inf, nor NaN)
                #       - the keypoint is not at the origin (sign of an error)
                #       - it's one of the first 6 keypoints (shoulders & up - more stable)
                # TODO: add a check for outliers?
                if conf > kp_conf_thresh and math.isfinite(keypoint[0]) and keypoint[0] != 0 and j <= 6:
                    position[0] += x
                    position[1] += y
                    position[2] += z
                    num_kps += 1 

            # Calculate the position of the person: average of valid keypoints
            if num_kps > 0:
                obj_msg.position = [position[0]/num_kps, position[1]/num_kps, position[2]/num_kps]
            else:
                # No valid keypoints: set the position to the origin (unlikely)
                obj_msg.position = [0, 0, 0]

            # Get indices of the person in the pose and velocity histories
            id = labels[i]
            pose_index, vel_index = get_indices(id, pose_history, vel_history)
            
            
            # If the person has a valid current position & has been detected before
            # Then update the pose and velocity histories accordingly
            if num_kps > 0 and pose_index != -1:
                # Update the pose and velocity histories & find average velocity
                pose_history, vel_history, averaged_velocity = update_pose_vel(pose_history, vel_history, obj_msg, pose_index, vel_index, id)

                # Set the velocity of the person
                obj_msg.velocity = averaged_velocity
                
            # If the person has a valid current position and has no pose history
            # Then add the person to the pose history
            if num_kps > 0 and pose_index == -1:
                # Add new pose to pose history
                position = obj_msg.position
                pose_history.append([id, [position[0], position[1], position[2], rospy.Time.now().to_sec()]])
                
            # If the person has a valid current position and has no vel history
            # Then add the person to the vel history
            if num_kps > 0 and vel_index == -1:
                # Add new velocity to velocity history
                vel_history.append([id, [[0,0,0]]])
                
            # If the person has no valid position, set the velocity to zero & don't change histories
            if num_kps == 0:
                obj_msg.velocity = [0, 0, 0]

            # Append the object message to the list
            obj_list.append(obj_msg)
    
    # Update the object list in the ROS message    
    ros_msg.objects = obj_list
    
    # Return the ROS message and the updated pose and velocity histories
    return ros_msg, pose_history, vel_history, removed_ids

# TODO: fix frames. This is in cbl, and we want it in odom when on chairry (not when detached)
def people_wrapper(ros_msg, tf_buffer):
    people_msg = People()
    # create header step by step
    people_msg.header.stamp = ros_msg.header.stamp
    people_msg.header.seq = ros_msg.header.seq
    people_msg.header.frame_id = "odom"  # Set the new frame_id to "odom"

    people_msg.people = []

    for obj_msg in ros_msg.objects:
        person = Person()
        person.name = str(obj_msg.label_id)

        # Create a PointStamped for the position
        position_stamped = tf2_geometry_msgs.PointStamped()
        position_stamped.header = ros_msg.header
        position_stamped.point.x = obj_msg.position[0]
        position_stamped.point.y = obj_msg.position[1]
        position_stamped.point.z = obj_msg.position[2]

        # Create a PointStamped for the velocity
        velocity_stamped = tf2_geometry_msgs.PointStamped()
        velocity_stamped.header = ros_msg.header
        velocity_stamped.point.x = obj_msg.velocity[0] + obj_msg.position[0]
        velocity_stamped.point.y = obj_msg.velocity[1] + obj_msg.position[1]
        velocity_stamped.point.z = obj_msg.velocity[2] + obj_msg.position[2]

        # Print original position and velocity
        print("original pos: ", [obj_msg.position[0], obj_msg.position[1], obj_msg.position[2]])
        print("original vel: ", [obj_msg.velocity[0], obj_msg.velocity[1], obj_msg.velocity[2]])

        try:
            # Retrieve the transform once
            transform = tf_buffer.lookup_transform("odom", ros_msg.header.frame_id, ros_msg.header.stamp, rospy.Duration(1.0))

            # Apply the transform to the position
            transformed_position = tf2_geometry_msgs.do_transform_point(position_stamped, transform)
            person.position.x = transformed_position.point.x
            person.position.y = transformed_position.point.y
            person.position.z = transformed_position.point.z

            # Print transformed position
            print("transformed pos: ", [person.position.x, person.position.y, person.position.z])

            # Apply the transform to the velocity
            transformed_velocity = tf2_geometry_msgs.do_transform_point(velocity_stamped, transform)
            person.velocity.x = transformed_velocity.point.x - person.position.x
            person.velocity.y = transformed_velocity.point.y - person.position.y
            person.velocity.z = transformed_velocity.point.z - person.position.z

            # Print transformed velocity
            print("transformed vel: ", [person.velocity.x, person.velocity.y, person.velocity.z])

        except tf2_ros.LookupException as e:
            rospy.logwarn(f"Transform lookup failed: {e}")
            continue
        except tf2_ros.ExtrapolationException as e:
            rospy.logwarn(f"Transform extrapolation failed: {e}")
            continue

        people_msg.people.append(person)

    return people_msg





# Prepare the histories, removing old data
def prep_histories(labels, pose_history, vel_history):
    global window_size
    
    # Format of pose_history and vel_history (assuming window_size = 3):
    # pose_history = [[id, [x, y, z, t]], [id, [x, y, z, t]], ...]
    # vel_history = [[id, [[x, y, z], [x, y, z], [x, y, z]]], [id, [[x, y, z], [x, y, z], [x, y, z]]], ...]
    
    # Clean pose history of any ids that are no longer present
    clean_history = []
    removed_labels = []
    for i in range(len(pose_history)):
        id = pose_history[i][0]
        if id in labels:
            clean_history.append(pose_history[i])
        else:
            removed_labels.append(id)
    pose_history = clean_history
    
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
    
    return pose_history, vel_history, removed_labels

# Find the index of the person in the pose and velocity histories
def get_indices(id, pose_history, vel_history):
    pose_index = -1
    vel_index = -1
    for i in range(len(pose_history)):
        if id == pose_history[i][0]:
            pose_index = i
            break
    for i in range(len(vel_history)):
        if id == vel_history[i][0]:
            vel_index = i
            break
    return pose_index, vel_index

# Update the pose and velocity histories & find average velocity
def update_pose_vel(pose_history, vel_history, obj_msg, pose_index, vel_index, id):
    # - replace the old position with the new position in pose history
    # - calculate the current velocity of the person
    # - save the current velocity in the velocity history
    # - average the velocity over the n most recent frames (n = window_size)
    
    # Format of pose_history and vel_history (assuming window_size = 3):
    # pose_history = [[id, [x, y, z, t]], [id, [x, y, z, t]], ...]
    # vel_history = [[id, [[x, y, z], [x, y, z], [x, y, z]]], [id, [[x, y, z], [x, y, z], [x, y, z]]], ...]

    # Prepare variables for velocity calculation & pose history update
    # Save the time in seconds for velocity calculation
    old_pose = pose_history[pose_index][1]
    position = obj_msg.position
    new_pose = [position[0], position[1], position[2], rospy.Time.now().to_sec()]
    delta_time = new_pose[3] - old_pose[3]
    
    # Replace the old pose with the new pose in the pose history
    pose_history[pose_index] = [id, new_pose]
    
    # Calculate velocity
    current_velocity = [(new_pose[0] - old_pose[0])/delta_time, (new_pose[1] - old_pose[1])/delta_time, (new_pose[2] - old_pose[2])/delta_time]
    vel_history[vel_index][1].append(current_velocity)
    if len(vel_history[vel_index][1]) > window_size:
        vel_history[vel_index][1].pop(0)

    # Find the average the velocity over the n most recent frames (n = window_size)
    recent_velocities = vel_history[vel_index][1]
    averaged_velocity = [0, 0, 0]
    for i in range(len(recent_velocities)):
        averaged_velocity[0] += recent_velocities[i][0]
        averaged_velocity[1] += recent_velocities[i][1]
        averaged_velocity[2] += recent_velocities[i][2]
    averaged_velocity[0] = averaged_velocity[0]/len(recent_velocities)
    averaged_velocity[1] = averaged_velocity[1]/len(recent_velocities)
    averaged_velocity[2] = averaged_velocity[2]/len(recent_velocities)
    
    return pose_history, vel_history, averaged_velocity

# Wrap skeleton data into ROS PointCloud2 message for RViz Visualisation
# TODO:
# - fix frame issue
def point_cloud_wrapper(ros_msg, frame):
    global kp_conf_thresh

    # Create a header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame
    
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
                # Convert point coordinates from millimeters to meters
                point = [keypoint.kp[0], keypoint.kp[1], keypoint.kp[2], blue]
                points.append(point)

        # Display the position of the skeleton
        pose = obj_msg.position
        if pose[0] != 0:
            point = [pose[0], pose[1], pose[2], white]
            points.append(point)
    
    point_cloud_msg = pc2.create_cloud(header, fields, points)
    return point_cloud_msg

# Populate a ROS MarkerArray message for each person's velocity using velocity_wrapper()
# TODO: obj_msg.label_id won't be in removed_ids, so this won't work
def velocity_array_wrapper(ros_msg, frame, removed_ids):
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

# Populate a ROS MarkerArray message for each person's velocity using velocity_wrapper()
# TODO: if distance between any two connected keypoints is too large (> 1 m), FLAG THESE POINTS AS outliers somehow
def bone_array_wrapper(ros_msg, frame, removed_ids):
    global kp_conf_thresh
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
    
    # Get velocity and pose of object
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
    
# Transform skeleton data from local to map frame
def local_to_map_transform(ros_msg):
    try:
        # Create a transform from chairry_base_link to zed2i_left_camera_frame
        # Hard-coded because tfBuffer wasn't working
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "chairry_base_link"
        transform.child_frame_id = CAMERA_NAME + "_left_camera_frame"
        transform.transform.translation.x = -0.25
        transform.transform.translation.y = 0
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

            # Velocity is relative to the position, so no need to transform it
            
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
    
    # Initialize the tf_buffer and TransformListener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    ######################
    ##### Publishers #####
    ######################
    # Publish the people
    pub_people = rospy.Publisher('/people', People, queue_size=50)
    pub_people = rospy.Publisher('/people', People, queue_size=50)
    # Publish the bones in the zed2i & chairry_base_link frames
    pub_bones_z = rospy.Publisher(CAMERA_NAME+'/skeletons/bones/z', MarkerArray, queue_size=10)
    pub_bones_c = rospy.Publisher(CAMERA_NAME+'/skeletons/bones/c', MarkerArray, queue_size=10)
    # Publish the objects in the zed2i & chairry_base_link frames
    pub_z = rospy.Publisher(CAMERA_NAME+'/skeletons/objects/z', zed_msgs.ObjectsStamped, queue_size=50)   # zed2i frame
    pub_c = rospy.Publisher(CAMERA_NAME+'/skeletons/objects/c', zed_msgs.ObjectsStamped, queue_size=50)     # chairry_base_link frame
    # Publish the point clouds in the zed2i & chairry_base_link frames
    pub_pc_z = rospy.Publisher(CAMERA_NAME+'/skeletons/point_cloud/z', PointCloud2, queue_size=10)
    pub_pc_c = rospy.Publisher(CAMERA_NAME+'/skeletons/point_cloud/c', PointCloud2, queue_size=10)
    # Publish the velocity markers in the zed2i_left_camera_frame frame
    pub_marker_z = rospy.Publisher(CAMERA_NAME+'/skeletons/velocity_markers/z', MarkerArray, queue_size=10)
    pub_marker_c = rospy.Publisher(CAMERA_NAME+'/skeletons/velocity_markers/c', MarkerArray, queue_size=10)

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
    vel_history = []

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
            ros_msg, pose_history, vel_history, removed_ids = objects_wrapper(objects, labels, pose_history, vel_history)
            pub_z.publish(ros_msg)
            
            # Publish the people for move_base
            people_msg = people_wrapper(ros_msg, tf_buffer)
            pub_people.publish(people_msg)
            
            # Publish the bones
            bone_msg = bone_array_wrapper(ros_msg, CAMERA_NAME + "_left_camera_frame", removed_ids)
            pub_bones_z.publish(bone_msg)  
                      
            # Publish a point cloud with all keypoints for visualisation
            pc_msg = point_cloud_wrapper(ros_msg, CAMERA_NAME + "_left_camera_frame")
            pub_pc_z.publish(pc_msg)
            
            # Publish a marker for the velocity of the skeletons
            markers_msg = velocity_array_wrapper(ros_msg, CAMERA_NAME + "_left_camera_frame", removed_ids)
            pub_marker_z.publish(markers_msg)
            
            

            # Transform the skeletons to the chairry_base_link frame
            if zed_location == 'chairry':
                # Publish skeletons in ROS as a custom ObjectsStamped message
                transform_msg = local_to_map_transform(ros_msg)
                pub_c.publish(transform_msg)
                
                # Publish the bones
                bone_msg = bone_array_wrapper(transform_msg, "chairry_base_link", removed_ids)
                pub_bones_c.publish(bone_msg)  

                # Publish a point cloud with all keypoints for visualisation
                pc_msg = point_cloud_wrapper(transform_msg, "chairry_base_link")
                pub_pc_c.publish(pc_msg)
                
                # Publish a marker for the velocity of the skeletons
                markers_msg = velocity_array_wrapper(transform_msg, "chairry_base_link", removed_ids)
                pub_marker_c.publish(markers_msg)
                
            
    
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
