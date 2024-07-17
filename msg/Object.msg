# Object label ID
int16 label_id

# Object closest position
float32[3] position

# Object velocity
float32[3] velocity

# Tracking available
bool tracking_available

# Tracking state
# 0 -> OFF (object not valid)
# 1 -> OK
# 2 -> SEARCHING (occlusion occurred, trajectory is estimated)
int8 tracking_state

# Is skeleton available?
bool skeleton_available

# 3D Person skeleton in world frame
zed_yolo_ros/Skeleton3D skeleton_3d
