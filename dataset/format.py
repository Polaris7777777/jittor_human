import os

HARD = os.getenv("HARD", "0")
HARD = HARD.lower() in ["1", "true", "yes", "on"]

if HARD == False:
  print("Difficulty: EASY")
  id_to_name = {
    0: 'hips',
    1: 'spine',
    2: 'chest',
    3: 'upper_chest',
    4: 'neck',
    5: 'head',
    6: 'l_shoulder',
    7: 'l_upper_arm',
    8: 'l_lower_arm',
    9: 'l_hand',
    10: 'r_shoulder',
    11: 'r_upper_arm',
    12: 'r_lower_arm',
    13: 'r_hand',
    14: 'l_upper_leg',
    15: 'l_lower_leg',
    16: 'l_foot',
    17: 'l_toe_base',
    18: 'r_upper_leg',
    19: 'r_lower_leg',
    20: 'r_foot',
    21: 'r_toe_base',
  }
  symmetric_joint_pairs = [
    (6, 10),   # l_shoulder ↔ r_shoulder
    (7, 11),   # l_upper_arm ↔ r_upper_arm
    (8, 12),   # l_lower_arm ↔ r_lower_arm
    (9, 13),   # l_hand ↔ r_hand
    (14, 18),  # l_upper_leg ↔ r_upper_leg
    (15, 19),  # l_lower_leg ↔ r_lower_leg
    (16, 20),  # l_foot ↔ r_foot
    (17, 21),  # l_toe_base ↔ r_toe_base
  ]

  symmetric_bones = [
      # ((左侧骨骼起点, 左侧骨骼终点), (右侧骨骼起点, 右侧骨骼终点))
      ((6, 7), (10, 11)),    # 肩→上臂
      ((7, 8), (11, 12)),    # 上臂→下臂
      ((8, 9), (12, 13)),    # 下臂→手
      ((14, 15), (18, 19)),  # 上腿→下腿
      ((15, 16), (19, 20)),  # 下腿→脚
      ((16, 17), (20, 21)),  # 脚→脚趾
  ]

  parents = [None, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20,]

else:
  print("Difficulty: HARD")
  id_to_name = {
    0: 'hips',
    1: 'spine',
    2: 'chest',
    3: 'upper_chest',
    4: 'neck',
    5: 'head',
    6: 'l_shoulder',
    7: 'l_upper_arm',
    8: 'l_lower_arm',
    9: 'l_hand',
    10: 'r_shoulder',
    11: 'r_upper_arm',
    12: 'r_lower_arm',
    13: 'r_hand',
    14: 'l_upper_leg',
    15: 'l_lower_leg',
    16: 'l_foot',
    17: 'l_toe_base',
    18: 'r_upper_leg',
    19: 'r_lower_leg',
    20: 'r_foot',
    21: 'r_toe_base',
    22: 'l_hand_thumb_1',
    23: 'l_hand_thumb_2',
    24: 'l_hand_thumb_3',
    25: 'l_hand_index_1',
    26: 'l_hand_index_2',
    27: 'l_hand_index_3',
    28: 'l_hand_middle_1',
    29: 'l_hand_middle_2',
    30: 'l_hand_middle_3',
    31: 'l_hand_ring_1',
    32: 'l_hand_ring_2',
    33: 'l_hand_ring_3',
    34: 'l_hand_pinky_1',
    35: 'l_hand_pinky_2',
    36: 'l_hand_pinky_3',
    37: 'r_hand_thumb_1',
    38: 'r_hand_thumb_2',
    39: 'r_hand_thumb_3',
    40: 'r_hand_index_1',
    41: 'r_hand_index_2',
    42: 'r_hand_index_3',
    43: 'r_hand_middle_1',
    44: 'r_hand_middle_2',
    45: 'r_hand_middle_3',
    46: 'r_hand_ring_1',
    47: 'r_hand_ring_2',
    48: 'r_hand_ring_3',
    49: 'r_hand_pinky_1',
    50: 'r_hand_pinky_2',
    51: 'r_hand_pinky_3',
  }

  symmetric_joint_pairs = [
    # 主要身体部位的对称关节
    (6, 10),   # l_shoulder ↔ r_shoulder
    (7, 11),   # l_upper_arm ↔ r_upper_arm
    (8, 12),   # l_lower_arm ↔ r_lower_arm
    (9, 13),   # l_hand ↔ r_hand
    (14, 18),  # l_upper_leg ↔ r_upper_leg
    (15, 19),  # l_lower_leg ↔ r_lower_leg
    (16, 20),  # l_foot ↔ r_foot
    (17, 21),  # l_toe_base ↔ r_toe_base
    
    # 左右手拇指的对称关节
    (22, 37),  # l_hand_thumb_1 ↔ r_hand_thumb_1
    (23, 38),  # l_hand_thumb_2 ↔ r_hand_thumb_2
    (24, 39),  # l_hand_thumb_3 ↔ r_hand_thumb_3
    
    # 左右手食指的对称关节
    (25, 40),  # l_hand_index_1 ↔ r_hand_index_1
    (26, 41),  # l_hand_index_2 ↔ r_hand_index_2
    (27, 42),  # l_hand_index_3 ↔ r_hand_index_3
    
    # 左右手中指的对称关节
    (28, 43),  # l_hand_middle_1 ↔ r_hand_middle_1
    (29, 44),  # l_hand_middle_2 ↔ r_hand_middle_2
    (30, 45),  # l_hand_middle_3 ↔ r_hand_middle_3
    
    # 左右手无名指的对称关节
    (31, 46),  # l_hand_ring_1 ↔ r_hand_ring_1
    (32, 47),  # l_hand_ring_2 ↔ r_hand_ring_2
    (33, 48),  # l_hand_ring_3 ↔ r_hand_ring_3
    
    # 左右手小指的对称关节
    (34, 49),  # l_hand_pinky_1 ↔ r_hand_pinky_1
    (35, 50),  # l_hand_pinky_2 ↔ r_hand_pinky_2
    (36, 51),  # l_hand_pinky_3 ↔ r_hand_pinky_3
  ]

  symmetric_bones = [
    # 主要身体部位的对称骨骼
    # ((左侧骨骼起点, 左侧骨骼终点), (右侧骨骼起点, 右侧骨骼终点))
    ((6, 7), (10, 11)),    # 肩→上臂
    ((7, 8), (11, 12)),    # 上臂→下臂
    ((8, 9), (12, 13)),    # 下臂→手
    ((14, 15), (18, 19)),  # 上腿→下腿
    ((15, 16), (19, 20)),  # 下腿→脚
    ((16, 17), (20, 21)),  # 脚→脚趾
    
    # 左右手拇指的对称骨骼
    ((9, 22), (13, 37)),   # 手→拇指1
    ((22, 23), (37, 38)),  # 拇指1→拇指2
    ((23, 24), (38, 39)),  # 拇指2→拇指3
    
    # 左右手食指的对称骨骼
    ((9, 25), (13, 40)),   # 手→食指1
    ((25, 26), (40, 41)),  # 食指1→食指2
    ((26, 27), (41, 42)),  # 食指2→食指3
    
    # 左右手中指的对称骨骼
    ((9, 28), (13, 43)),   # 手→中指1
    ((28, 29), (43, 44)),  # 中指1→中指2
    ((29, 30), (44, 45)),  # 中指2→中指3
    
    # 左右手无名指的对称骨骼
    ((9, 31), (13, 46)),   # 手→无名指1
    ((31, 32), (46, 47)),  # 无名指1→无名指2
    ((32, 33), (47, 48)),  # 无名指2→无名指3
    
    # 左右手小指的对称骨骼
    ((9, 34), (13, 49)),   # 手→小指1
    ((34, 35), (49, 50)),  # 小指1→小指2
    ((35, 36), (50, 51)),  # 小指2→小指3
  ]

  parents = [None, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20, 9, 22, 23, 9, 25, 26, 9, 28, 29, 9, 31, 32, 9, 34, 35, 13, 37, 38, 13, 40, 41, 13, 43, 44, 13, 46, 47, 13, 49, 50,]

num_joints = len(parents)
body_mask = [True] * 22 + [False] * 30  # 前22个关节为身体，后30个为手部关节

retarget_mapping = {
  'mixamorig:Hips': 'hips',
  'mixamorig:Spine': 'spine',
  'mixamorig:Spine1': 'chest',
  'mixamorig:Spine2': 'upper_chest',
  'mixamorig:Neck': 'neck',
  'mixamorig:Head': 'head',
  'mixamorig:LeftShoulder': 'l_shoulder',
  'mixamorig:LeftArm': 'l_upper_arm',
  'mixamorig:LeftForeArm': 'l_lower_arm',
  'mixamorig:LeftHand': 'l_hand',
  'mixamorig:RightShoulder': 'r_shoulder',
  'mixamorig:RightArm': 'r_upper_arm',
  'mixamorig:RightForeArm': 'r_lower_arm',
  'mixamorig:RightHand': 'r_hand',
  'mixamorig:LeftUpLeg': 'l_upper_leg',
  'mixamorig:LeftLeg': 'l_lower_leg',
  'mixamorig:LeftFoot': 'l_foot',
  'mixamorig:LeftToeBase': 'l_toe_base',
  'mixamorig:RightUpLeg': 'r_upper_leg',
  'mixamorig:RightLeg': 'r_lower_leg',
  'mixamorig:RightFoot': 'r_foot',
  'mixamorig:RightToeBase': 'r_toe_base',
  'mixamorig:LeftHandThumb1': 'l_hand_thumb_1',
  'mixamorig:LeftHandThumb2': 'l_hand_thumb_2',
  'mixamorig:LeftHandThumb3': 'l_hand_thumb_3',
  'mixamorig:LeftHandIndex1': 'l_hand_index_1',
  'mixamorig:LeftHandIndex2': 'l_hand_index_2',
  'mixamorig:LeftHandIndex3': 'l_hand_index_3',
  'mixamorig:LeftHandMiddle1': 'l_hand_middle_1',
  'mixamorig:LeftHandMiddle2': 'l_hand_middle_2',
  'mixamorig:LeftHandMiddle3': 'l_hand_middle_3',
  'mixamorig:LeftHandRing1': 'l_hand_ring_1',
  'mixamorig:LeftHandRing2': 'l_hand_ring_2',
  'mixamorig:LeftHandRing3': 'l_hand_ring_3',
  'mixamorig:LeftHandPinky1': 'l_hand_pinky_1',
  'mixamorig:LeftHandPinky2': 'l_hand_pinky_2',
  'mixamorig:LeftHandPinky3': 'l_hand_pinky_3',
  'mixamorig:RightHandThumb1': 'r_hand_thumb_1',
  'mixamorig:RightHandThumb2': 'r_hand_thumb_2',
  'mixamorig:RightHandThumb3': 'r_hand_thumb_3',
  'mixamorig:RightHandIndex1': 'r_hand_index_1',
  'mixamorig:RightHandIndex2': 'r_hand_index_2',
  'mixamorig:RightHandIndex3': 'r_hand_index_3',
  'mixamorig:RightHandMiddle1': 'r_hand_middle_1',
  'mixamorig:RightHandMiddle2': 'r_hand_middle_2',
  'mixamorig:RightHandMiddle3': 'r_hand_middle_3',
  'mixamorig:RightHandRing1': 'r_hand_ring_1',
  'mixamorig:RightHandRing2': 'r_hand_ring_2',
  'mixamorig:RightHandRing3': 'r_hand_ring_3',
  'mixamorig:RightHandPinky1': 'r_hand_pinky_1',
  'mixamorig:RightHandPinky2': 'r_hand_pinky_2',
  'mixamorig:RightHandPinky3': 'r_hand_pinky_3',
}



# =======================SMPL skeleton=======================
smpl_id_to_name = {
  0: 'Left_Hip', 
  1: 'Right_Hip', 
  2: 'Waist', 
  3: 'Left_Knee', 
  4: 'Right_Knee',
  5: 'Upper_Waist', 
  6: 'Left_Ankle', 
  7: 'Right_Ankle', 
  8: 'Chest',
  9: 'Left_Toe', 
  10: 'Right_Toe', 
  11: 'Base_Neck', 
  12: 'Left_Shoulder',
  13: 'Right_Shoulder', 
  14: 'Upper_Neck', 
  15: 'Left_Arm', 
  16: 'Right_Arm',
  17: 'Left_Elbow', 
  18: 'Right_Elbow', 
  19: 'Left_Wrist', 
  20: 'Right_Wrist',
  21: 'Left_Finger', 
  22: 'Right_Finger'
}