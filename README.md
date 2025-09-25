# ZED YOLO ROS

A simple ROS package for 3D object detection using a [ZED stereo camera](https://www.stereolabs.com/zed-2i/) and any YOLO model.
The package runs the ZED camera through the [ZED Python SDK](https://github.com/stereolabs/zed-python-api) and uses [Ultralytics YOLO](https://docs.ultralytics.com/models) for 2D detections, which are then combined with ZED’s depth estimation to produce 3D bounding boxes.

Detections are wrapped as [`zed_interfaces/ObjectsStamped`](https://github.com/stereolabs/zed-ros-interfaces/blob/master/msg/ObjectsStamped.msg) messages and published on ROS topics, making them straightforward to integrate into robotics applications.

Compared to the [official ZED ROS Wrapper](https://github.com/stereolabs/zed-ros-wrapper), this package gives you **full control over the YOLO model**: you can easily switch between available pretrained models or plug in your own custom-trained YOLO network (something not supported by the default wrapper).

---

## Features

* Plug-and-play **3D object detection in ROS**.
* Works with **any YOLO model** (pretrained or custom).
* Publishes detections as `ObjectsStamped` ROS messages:

  * Local frame: `/zed2i/od_yolo`
  * Global (map) frame (TF-transformed): `/zed2i/od_yolo_map_frame`

---

## Installation

Clone into your ROS workspace and build:

```bash
cd ~/catkin_ws/src
git clone https://github.com/casado-fdo/zed-yolo-ros.git
cd ..
catkin_make
```

Dependencies:

* ROS (tested on Noetic)
* [ZED SDK](https://www.stereolabs.com/docs/installation/)
* [ZED Python API](https://github.com/stereolabs/zed-python-api)
* [Ultralytics YOLO](https://docs.ultralytics.com/) (`pip install ultralytics`)
* PyTorch, OpenCV, TF2

---

## Usage

Run the detector node:

```bash
rosrun zed_yolo_ros detector.py --model_name yolov8l
```

Options:

* `--model_name`: YOLO model name (default: `yolov8l-oiv7`).
* `--svo`: Path to an SVO file for playback instead of live camera.
* `--img_size`: Inference resolution (default: 416).
* `--conf_thres`: Detection confidence threshold (default: 0.4).

Full list of available YOLO models: [Ultralytics Models](https://docs.ultralytics.com/models).
You can also provide your **own trained YOLO weights** by placing them under the `models/` directory.

---

## ROS Topics

* `/zed2i/od_yolo` → Detected objects in the **camera frame**.
* `/zed2i/od_yolo_map_frame` → Detected objects transformed into the **map frame** (requires TF).

Messages follow the [`zed_interfaces/ObjectsStamped`](https://github.com/stereolabs/zed-ros-interfaces/blob/master/msg/ObjectsStamped.msg) definition.

![Demo in Rviz](assets/demo_rviz.gif)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

### Acknowledgments
Developed at the Personal Robotics Laboratory, Imperial College London (2024).
