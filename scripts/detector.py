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

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer


lock = Lock()
run_signal = False
svo = None
img_size = 416
conf_thres = 0.4
model_name = 'yolov8m-ch'
zed_location = 'detached'
skeletons = None

CAMERA_NAME = "zed2i"

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
    print("Model loading...") 
     
    # Load the model
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


# Wrap data into ROS ObjectStamped message
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

def point_cloud_wrapper(ros_msg):
    
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = CAMERA_NAME
    
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]
    
    r = 0
    g = 0
    b = 255
    color = (r << 16) | (g << 8) | b
    
    points = []
    for obj_msg in ros_msg.objects:
        skeleton = obj_msg.skeleton_3d
        for keypoint in skeleton.keypoints:
            confidence = keypoint.kp[3]
            kp_conf_thresh = 0.9
            if confidence > kp_conf_thresh and math.isfinite(keypoint.kp[0]) and keypoint.kp[0] != 0:
                point = [keypoint.kp[0]/1000.0, keypoint.kp[1]/1000.0, keypoint.kp[2]/1000.0, color]
                points.append(point)
    
    point_cloud_msg = pc2.create_cloud(header, fields, points)
    return point_cloud_msg
    

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


def main():
    global image_np, run_signal, svo, img_size, conf_thres, model_name, zed_location, skeletons, ids, display

    # Define ROS publishers
    pub_l = rospy.Publisher(CAMERA_NAME+'/od_yolo_zed2i', zed_msgs.ObjectsStamped, queue_size=50)   # zed2i frame
    pub_c = rospy.Publisher(CAMERA_NAME+'/od_yolo_cbl', zed_msgs.ObjectsStamped, queue_size=50)     # chairry frame
    pub_pc = rospy.Publisher(CAMERA_NAME+'/skeleton_point_cloud', PointCloud2, queue_size=10)
    # tfBuffer = tf2_ros.Buffer()

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
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA        # Use depth ultra mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER   # Use millimeter units (for depth measurements)
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance = 10000           # Set the maximum depth distance to 10 meters

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Initialise image and point cloud
    image_left_temp = sl.Mat()
    point_cloud = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    if zed_location == 'detached':
        # improve static performance and have boxes stuck to the ground
        positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.filtering_mode = sl.OBJECT_FILTERING_MODE.NONE
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    if display:
        # Create OpenGL viewer
        viewer = gl.GLViewer()
        point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
        point_cloud_render = sl.Mat()
        viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
        point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        image_left = sl.Mat()
        # Utilities for 2D display
        display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
        image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

        # Utilities for tracks view
        camera_config = camera_infos.camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
        # Camera pose
        cam_w_pose = sl.Pose()

    while rospy.is_shutdown() is False:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_temp, sl.VIEW.LEFT)       # image_left_temp is a Mat object
            image_np = image_left_temp.get_data()                   # image_np is a numpy array
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for skeleton detections
            lock.acquire()
            
            # -- Ingest skeletons
            objects, labels = ingest_skeletons(skeletons, ids, point_cloud)
        
            lock.release()

            # Publish in ROS as an ObjectsStamped message
            ros_msg = objects_wrapper(objects, labels)
            pub_l.publish(ros_msg)
            # pub_c.publish(local_to_map_transform(ros_msg, tfBuffer, "chairry_base_link"))
            
            # Publish a point cloud with all keypoints
            pc_msg = point_cloud_wrapper(ros_msg)
            pub_pc.publish(pc_msg)

            if display:
                # -- Display
                # Retrieve display data
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
                point_cloud.copy_to(point_cloud_render)
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

                # 3D rendering
                viewer.updateData(point_cloud_render, objects)
                # 2D rendering
                np.copyto(image_left_ocv, image_left.get_data())
                cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
                global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                # Tracking view
                track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

                cv2.imshow("ZED | 2D View and Birds View", global_image)
                key = cv2.waitKey(10)
                if key == 27:
                    break
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
    zed_location = rospy.get_param(namespace + 'zed_location', 'detached')

    rospy.loginfo(f"Model Name: {model_name}")
    rospy.loginfo(f"SVO File: {svo}")
    rospy.loginfo(f"Image Size: {img_size}")
    rospy.loginfo(f"Confidence Threshold: {conf_thres}")
    rospy.loginfo(f"ZED Location: {zed_location}")

    display = False

    with torch.no_grad():
        main()
