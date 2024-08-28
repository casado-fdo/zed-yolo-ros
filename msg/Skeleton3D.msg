# Skeleton joints indices
# As if we are behind the person
# (their left is our left)
# Dotted lines represent connections that we can use to validate Skeleton
# 0: nose
# 1,2: eyes
# 3,4: ears
# 5,6: shoulders
# 7,8: elbows
# 9,10: wrists
# 11,12: hips
# 13,14: knees
# 15,16: ankles
#
#          3-1   2-4
#             \ /
#              0
#           /  |  \
#         /    |    \
#       5-------------6
#       |    |   |    |
#	    |    |   |    |
#       7    |   |    8
#       |    |   |    |
#       |    |   |    |
#       9   11---12   10
#            |   |
#            |   |
#            |   |
#           13   14
#            |   |
#            |   |
#            |   |
#           15   16
zed_yolo_ros/Keypoint3D[17] keypoints
