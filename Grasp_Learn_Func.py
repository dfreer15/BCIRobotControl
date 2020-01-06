import vrep_files.vrep as vrep
import time
import random
import numpy as np
import math
import csv
import os
import cv2
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import tensorflow as tf

use_BCI = True
if use_BCI:
    import real_time_BCI_train as real_time_BCI
import eeg_io_pp_2


def sim_setup(gripper_type):
    vrep.simxFinish(-1)  # just in case, close all open connections

    global clientID
    global joint_handles, cam_Handle, aug_Handle
    global resolution, EE_turn
    global success_data

    success_data = []

    clientID = vrep.simxStart("127.0.0.1", 19999, True, True, 2000, 5)
    if clientID != -1:
        print('Connected to remote API server')
        connected = True
        # pix_prop_thresh1, pix_prop_thresh2, vel_1, vel_2, center_res1, center_res2 = 0.25, 0.68, 0.02, 0.02, 5, 4
        # cent_weight1, cent_weight2 = 0.005, 0.0025
        errorCode, cam_Handle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor', vrep.simx_opmode_blocking)
        errorCode, aug_Handle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor_aug', vrep.simx_opmode_blocking)
        check, Joint1_Handle = vrep.simxGetObjectHandle(clientID, 'ActiveArm_joint1', vrep.simx_opmode_blocking)
        check, Joint2_Handle = vrep.simxGetObjectHandle(clientID, 'ActiveArm_joint2', vrep.simx_opmode_blocking)
        check, Joint3_Handle = vrep.simxGetObjectHandle(clientID, 'ActiveArm_joint3', vrep.simx_opmode_blocking)
        check, Joint4_Handle = vrep.simxGetObjectHandle(clientID, 'ActiveArm_joint4', vrep.simx_opmode_blocking)
        check, Joint5_Handle = vrep.simxGetObjectHandle(clientID, 'ActiveArm_joint5', vrep.simx_opmode_blocking)
        check, Joint6_Handle = vrep.simxGetObjectHandle(clientID, 'ActiveArm_joint6', vrep.simx_opmode_blocking)

        joint_handles = [Joint1_Handle, Joint2_Handle, Joint3_Handle, Joint4_Handle, Joint5_Handle, Joint6_Handle]

        errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                      vrep.simx_opmode_streaming)
        errorCode2, resolution, image_d = vrep.simxGetVisionSensorDepthBuffer(clientID, cam_Handle,
                                                                              vrep.simx_opmode_streaming)

        while len(resolution) < 1:
            errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                          vrep.simx_opmode_streaming)

        gripper_handles, FS_handles = setup_gripper(gripper_type)

        for i in range(0, 6):
            vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_ctrl_enabled, 1,
                                           vrep.simx_opmode_oneshot)
            vrep.simxGetJointPosition(clientID, joint_handles[i], vrep.simx_opmode_streaming)

        return [connected, clientID, joint_handles, cam_Handle]
    else:
        return [False, 0, 0, 0]


def kuka_sim_setup(gripper_type):
    vrep.simxFinish(-1)  # just in case, close all open connections

    global clientID
    global joint_handles, cam_Handle, aug_Handle, FS_handles
    global resolution, EE_turn
    global success_data

    success_data = []

    EE_turn = 0

    clientID = vrep.simxStart("127.0.0.1", 19999, True, True, 2000, 5)
    if clientID != -1:
        print('Connected to remote API server')
        connected = True

        errorCode, cam_Handle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor', vrep.simx_opmode_blocking)
        errorCode, aug_Handle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor_aug', vrep.simx_opmode_blocking)

        print(cam_Handle)
        print(aug_Handle)

        # KUKA Handles
        check, Joint1_Handle = vrep.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint1', vrep.simx_opmode_blocking)
        check, Joint2_Handle = vrep.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint2', vrep.simx_opmode_blocking)
        check, Joint3_Handle = vrep.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint3', vrep.simx_opmode_blocking)
        check, Joint4_Handle = vrep.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint4', vrep.simx_opmode_blocking)
        check, Joint5_Handle = vrep.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint5', vrep.simx_opmode_blocking)
        check, Joint6_Handle = vrep.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint6', vrep.simx_opmode_blocking)
        check, Joint7_Handle = vrep.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint7', vrep.simx_opmode_blocking)

        joint_handles = [Joint1_Handle, Joint2_Handle, Joint3_Handle,
                         Joint4_Handle, Joint6_Handle, Joint7_Handle]

        for i in range(0, len(joint_handles)):
            vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_ctrl_enabled, 1,
                                           vrep.simx_opmode_oneshot)
            returncode, angle = vrep.simxGetJointPosition(clientID, joint_handles[i], vrep.simx_opmode_streaming)
            # print (i, joint_handles[i], angle)

        errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                      vrep.simx_opmode_streaming)
        errorCode2, resolution, image_d = vrep.simxGetVisionSensorDepthBuffer(clientID, cam_Handle,
                                                                              vrep.simx_opmode_streaming)

        while len(resolution) < 1:
            errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                          vrep.simx_opmode_streaming)

        gripper_handles, FS_handles = setup_gripper(gripper_type)

        for i in range(2, 4):
            returncode, angle = vrep.simxGetJointPosition(clientID, gripper_handles[i], vrep.simx_opmode_streaming)

        return [connected, clientID, joint_handles, gripper_handles, cam_Handle]

    else:
        return [False, 0, 0, 0, 0]


def ABB_sim_setup(gripper_type):
    vrep.simxFinish(-1)  # just in case, close all open connections

    global clientID
    global joint_handles, cam_Handle, aug_Handle, FS_handles
    global resolution, EE_turn
    global success_data

    success_data = []

    EE_turn = 0

    clientID = vrep.simxStart("127.0.0.1", 19999, True, True, 2000, 5)
    if clientID != -1:
        print('Connected to remote API server')
        connected = True

        errorCode, cam_Handle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor', vrep.simx_opmode_blocking)
        errorCode, aug_Handle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor_aug', vrep.simx_opmode_blocking)

        print(cam_Handle)
        print(aug_Handle)

        # KUKA Handles
        check, Joint1_Handle = vrep.simxGetObjectHandle(clientID, 'IRB140_joint1', vrep.simx_opmode_blocking)
        check, Joint2_Handle = vrep.simxGetObjectHandle(clientID, 'IRB140_joint2', vrep.simx_opmode_blocking)
        check, Joint3_Handle = vrep.simxGetObjectHandle(clientID, 'IRB140_joint3', vrep.simx_opmode_blocking)
        check, Joint4_Handle = vrep.simxGetObjectHandle(clientID, 'IRB140_joint4', vrep.simx_opmode_blocking)
        check, Joint5_Handle = vrep.simxGetObjectHandle(clientID, 'IRB140_joint5', vrep.simx_opmode_blocking)
        check, Joint6_Handle = vrep.simxGetObjectHandle(clientID, 'IRB140_joint6', vrep.simx_opmode_blocking)

        joint_handles = [Joint1_Handle, Joint2_Handle, Joint3_Handle,
                         Joint4_Handle, Joint5_Handle, Joint6_Handle]

        for i in range(0, len(joint_handles)):
            vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_ctrl_enabled, 1,
                                           vrep.simx_opmode_oneshot)
            returncode, angle = vrep.simxGetJointPosition(clientID, joint_handles[i], vrep.simx_opmode_streaming)
            # print (i, joint_handles[i], angle)

        errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                      vrep.simx_opmode_streaming)
        errorCode2, resolution, image_d = vrep.simxGetVisionSensorDepthBuffer(clientID, cam_Handle,
                                                                              vrep.simx_opmode_streaming)

        while len(resolution) < 1:
            errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                          vrep.simx_opmode_streaming)

        gripper_handles, FS_handles = setup_gripper(gripper_type)

        for i in range(0, len(gripper_handles)):
            returncode, angle = vrep.simxGetJointPosition(clientID, gripper_handles[i], vrep.simx_opmode_streaming)

        return [connected, clientID, joint_handles, gripper_handles, cam_Handle]

    else:
        return [False, 0, 0, 0, 0]


def KR10_sim_setup(gripper_type):
    vrep.simxFinish(-1)  # just in case, close all open connections

    global clientID
    global joint_handles, gripper_handles, cam_Handle, aug_Handle, FS_handles
    global resolution, EE_turn
    global success_data

    success_data = []

    EE_turn = 0

    clientID = vrep.simxStart("127.0.0.1", 19999, True, True, 2000, 5)
    if clientID != -1:
        print('Connected to remote API server')
        connected = True

        errorCode, cam_Handle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor', vrep.simx_opmode_blocking)
        # errorCode, aug_Handle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor_aug', vrep.simx_opmode_blocking)

        # KUKA Handles
        check, Joint1_Handle = vrep.simxGetObjectHandle(clientID, 'KR10_joint1', vrep.simx_opmode_blocking)
        check, Joint2_Handle = vrep.simxGetObjectHandle(clientID, 'KR10_joint2', vrep.simx_opmode_blocking)
        check, Joint3_Handle = vrep.simxGetObjectHandle(clientID, 'KR10_joint3', vrep.simx_opmode_blocking)
        check, Joint4_Handle = vrep.simxGetObjectHandle(clientID, 'KR10_joint4', vrep.simx_opmode_blocking)
        check, Joint5_Handle = vrep.simxGetObjectHandle(clientID, 'KR10_joint5', vrep.simx_opmode_blocking)

        joint_handles = [Joint1_Handle, Joint2_Handle, Joint3_Handle,
                         Joint4_Handle, Joint5_Handle]

        for i in range(0, len(joint_handles)):
            vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_ctrl_enabled, 1,
                                           vrep.simx_opmode_oneshot)
            returncode, angle = vrep.simxGetJointPosition(clientID, joint_handles[i], vrep.simx_opmode_streaming)
            # print (i, joint_handles[i], angle)

        errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                      vrep.simx_opmode_streaming)
        errorCode2, resolution, image_d = vrep.simxGetVisionSensorDepthBuffer(clientID, cam_Handle,
                                                                              vrep.simx_opmode_streaming)

        while len(resolution) < 1:
            errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                          vrep.simx_opmode_streaming)

        gripper_handles, FS_handles = setup_gripper(gripper_type)

        for gripper_handle in gripper_handles:
            returncode, angle = vrep.simxGetJointPosition(clientID, gripper_handle, vrep.simx_opmode_streaming)

        return [connected, clientID, joint_handles, gripper_handles, cam_Handle]

    else:
        return [False, 0, 0, 0, 0]


def setup_gripper(gripper_type):
    global full_gripper_handle, full_ghost_gripper_handle, gripper_distance
    gripper_distance = []
    if gripper_type == 1:
        # Jaco Hand
        check, full_gripper_handle = vrep.simxGetObjectHandle(clientID, 'JacoHand', vrep.simx_opmode_blocking)
        check, full_ghost_gripper_handle = vrep.simxGetObjectHandle(clientID, 'JacoHand_g', vrep.simx_opmode_blocking)
        gripper_handles = [0]
        FS_handles = [0]
    elif gripper_type == 2:
        # Schunk 5-fingered hand
        index_handles = [0, 0, 0]
        middle_handles = [0, 0, 0]
        thumb_handles = [0, 0, 0]
        ring_handles = [0, 0, 0]
        pinky_handles = [0, 0, 0]
        for i in range(3):
            check, index_handles[i] = vrep.simxGetObjectHandle(clientID, 'index' + str(i)+'_j', vrep.simx_opmode_blocking)
            check, middle_handles[i] = vrep.simxGetObjectHandle(clientID, 'middle' + str(i)+'_j', vrep.simx_opmode_blocking)
            check, thumb_handles[i] = vrep.simxGetObjectHandle(clientID, 'thumb' + str(i) + '_j', vrep.simx_opmode_blocking)
            check, ring_handles[i] = vrep.simxGetObjectHandle(clientID, 'ring' + str(i) + '_j', vrep.simx_opmode_blocking)
            check, pinky_handles[i] = vrep.simxGetObjectHandle(clientID, 'pinky' + str(i) + '_j', vrep.simx_opmode_blocking)

        carpal_handle = vrep.simxGetObjectHandle(clientID, 'carpal_j', vrep.simx_opmode_blocking)

        gripper_handles = np.concatenate((index_handles, middle_handles, thumb_handles, ring_handles, pinky_handles, carpal_handle))
        FS_handles = [0]

    elif gripper_type == 3:
        # Barrett Hand
        check, GripJointHandleA0 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_jointA_0', vrep.simx_opmode_blocking)
        check, GripJointHandleA2 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_jointA_2', vrep.simx_opmode_blocking)
        check, GripJointHandleB0 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_jointB_0', vrep.simx_opmode_blocking)
        check, GripJointHandleB1 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_jointB_1', vrep.simx_opmode_blocking)
        check, GripJointHandleB2 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_jointB_2', vrep.simx_opmode_blocking)
        check, GripJointHandleC0 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_jointC_0', vrep.simx_opmode_blocking)
        check, GripJointHandleC1 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_jointC_1', vrep.simx_opmode_blocking)
        check, GripJointHandleC2 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_jointC_2', vrep.simx_opmode_blocking)

        check, FSHandle0 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_fingerTipSensor0', vrep.simx_opmode_blocking)
        check, FSHandle1 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_fingerTipSensor1', vrep.simx_opmode_blocking)
        check, FSHandle2 = vrep.simxGetObjectHandle(clientID, 'BarrettHand_fingerTipSensor2', vrep.simx_opmode_blocking)

        gripper_handles = [GripJointHandleA0, GripJointHandleA2,
                           GripJointHandleB0, GripJointHandleB1, GripJointHandleB2,
                           GripJointHandleC0, GripJointHandleC1, GripJointHandleC2]
        FS_handles = [FSHandle0, FSHandle1, FSHandle2]

        returnCode, state, forceVector0, torqueVector0 = vrep.simxReadForceSensor(clientID, FS_handles[0],
                                                                                  vrep.simx_opmode_streaming)
        returnCode, state, forceVector1, torqueVector1 = vrep.simxReadForceSensor(clientID, FS_handles[1],
                                                                                  vrep.simx_opmode_streaming)
        returnCode, state, forceVector2, torqueVector2 = vrep.simxReadForceSensor(clientID, FS_handles[2], vrep.simx_opmode_streaming)

    elif gripper_type == 4:
        # Salford Hand
        i = 0
        base_handles = [0, 0, 0]
        flex_handles = [0, 0, 0]
        flex2_handles = [0, 0, 0]
        FS_handles = [0, 0, 0]

        for finger in ['l0', 'r0', 't0']:
            check, base_handles[i] = vrep.simxGetObjectHandle(clientID, 'Base_joint_' + finger, vrep.simx_opmode_blocking)
            check, flex_handles[i] = vrep.simxGetObjectHandle(clientID, 'Flex_joint_' + finger, vrep.simx_opmode_blocking)
            check, flex2_handles[i] = vrep.simxGetObjectHandle(clientID, 'Flex_joint_2_' + finger, vrep.simx_opmode_blocking)
            check, FS_handles[i] = vrep.simxGetObjectHandle(clientID, 'Force_sensor_' + finger, vrep.simx_opmode_blocking)
            i += 1

        gripper_handles = np.concatenate((base_handles[:2], flex_handles, flex2_handles))

    return [gripper_handles, FS_handles]


def select_object(low_thresh, up_thresh):

    # Assign new object position (simulation)
    OP_x = (random.random() - 0.5) * 1.2
    OP_y = (random.random() - 0.5) * 1.2
    while math.sqrt(OP_x ** 2 + OP_y ** 2) < low_thresh or math.sqrt(OP_x ** 2 + OP_y ** 2) > up_thresh:
        OP_x = (random.random() - 0.5) * 1.2
        OP_y = (random.random() - 0.5) * 1.2

    # Choose which shape will be selected
    ob_sel = random.random()
    if ob_sel < .333:
        print('Cube!')
        obj_type = 'Cube'
    elif ob_sel < .667:
        print('Cylinder!')
        obj_type = 'Cylinder'
    else:
        print('Sphere!')
        obj_type = 'Sphere'

    print('Object Placement x: ', OP_x)
    print('Object Placement y: ', OP_y)

    return [obj_type, OP_x, OP_y]


def place_object(OP_x, OP_y, obj_type):
    global objHandle

    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y, vrep.simx_opmode_oneshot)

    init_param = []

    if obj_type == 'Cube':
        vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cube', vrep.simx_opmode_oneshot)
        errorCode, objHandle = vrep.simxGetObjectHandle(clientID, 'Cuboid', vrep.simx_opmode_blocking)
        paramreadfile = open('ParamFileCube.csv', 'rb+')
    if obj_type == 'Cylinder':
        vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cylinder', vrep.simx_opmode_oneshot)
        errorCode, objHandle = vrep.simxGetObjectHandle(clientID, 'Cylinder', vrep.simx_opmode_blocking)
        paramreadfile = open('ParamFileCyl.csv', 'rb+')
    if obj_type == 'Sphere':
        vrep.simxSetStringSignal(clientID, 'Obj_type', 'Sphere', vrep.simx_opmode_oneshot)
        errorCode, objHandle = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_blocking)
        paramreadfile = open('ParamFileSphere.csv', 'rb+')

    param_reader = csv.reader(paramreadfile, delimiter=',', quotechar='|')
    init_param_t = [list(map(float, rec)) for rec in param_reader]
    if len(init_param_t) > 0:
        init_param = init_param_t[0]
        # print(init_param)
        paramreadfile.close()
    return [objHandle, init_param]


def place_all_3_obj(low_thresh, up_thresh):
    global objHandles
    objHandles = np.zeros(3)

    for i in range(0, 3):
        # Assign new object position (simulation)
        OP_x = (random.random() - 0.5) * 1.2
        OP_y = abs(random.random() - 0.5) * 1.2
        while math.sqrt(OP_x ** 2 + OP_y ** 2) < low_thresh or math.sqrt(OP_x ** 2 + OP_y ** 2) > up_thresh:
            OP_x = (random.random() - 0.5) * 1.2
            OP_y = abs(random.random() - 0.5) * 1.2

        vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x, vrep.simx_opmode_oneshot)
        vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y, vrep.simx_opmode_oneshot)

        if i == 0:
            vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cube', vrep.simx_opmode_oneshot)
            errorCode, objHandles[0] = vrep.simxGetObjectHandle(clientID, 'Cuboid', vrep.simx_opmode_blocking)
        elif i == 1:
            vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cylinder', vrep.simx_opmode_oneshot)
            errorCode, objHandles[1] = vrep.simxGetObjectHandle(clientID, 'Cylinder', vrep.simx_opmode_blocking)
        elif i == 2:
            vrep.simxSetStringSignal(clientID, 'Obj_type', 'Sphere', vrep.simx_opmode_oneshot)
            errorCode, objHandles[2] = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_blocking)

    vrep.simxSetStringSignal(clientID, 'Obj_type', 'All', vrep.simx_opmode_oneshot)

    # get_20()

    return objHandles


def place_all_3_obj_setpos():
    global objHandles
    objHandles = np.zeros(3)
    theta = 45*math.pi/180
    distance = 0.5

    # print('Place All 3 Object Set Positions!!!!!')

    OP_x_cube, OP_y_cube = -0.1035, distance
    OP_x_cyl, OP_y_cyl = distance*math.sin(theta)-0.1035, distance*math.cos(theta)
    OP_x_sphere, OP_y_sphere = -distance*math.sin(theta)-0.1035, distance*math.cos(theta)
    # OP_x_sphere, OP_y_sphere = -0.1035, distance
    # OP_x_cyl, OP_y_cyl = distance * math.sin(theta) - 0.1035, distance * math.cos(theta)
    # OP_x_cube, OP_y_cube = -distance * math.sin(theta) - 0.1035, distance * math.cos(theta)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cube, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cube, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cube', vrep.simx_opmode_oneshot)
    errorCode, objHandles[0] = vrep.simxGetObjectHandle(clientID, 'Cuboid', vrep.simx_opmode_blocking)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cyl, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cyl, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cylinder', vrep.simx_opmode_oneshot)
    errorCode, objHandles[1] = vrep.simxGetObjectHandle(clientID, 'Cylinder', vrep.simx_opmode_blocking)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_sphere, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_sphere, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Sphere', vrep.simx_opmode_oneshot)
    errorCode, objHandles[2] = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_blocking)

    vrep.simxSetStringSignal(clientID, 'Obj_type', 'All', vrep.simx_opmode_oneshot)

    return objHandles


def place_all_9_obj_setpos():
    global objHandles, obj_names
    objHandles = np.zeros(9)
    obj_names = []
    theta = 40*math.pi/180
    distance = 0.5

    # OP_x_cube, OP_y_cube = -0.1035, distance
    # OP_x_cyl, OP_y_cyl = distance*math.sin(theta)-0.1035, distance*math.cos(theta)
    # OP_x_sphere, OP_y_sphere = -distance*math.sin(theta)-0.1035, distance*math.cos(theta)
    # OP_x_cube_y, OP_y_cube_y = distance*math.sin(2*theta)-0.1035, distance*math.cos(2*theta)
    # OP_x_cyl_r, OP_y_cyl_r = distance*math.sin(3*theta)-0.1035, distance*math.cos(3*theta)
    # OP_x_sphere_b, OP_y_sphere_b = distance * math.sin(4 * theta) - 0.1035, distance * math.cos(4 * theta)
    # OP_x_cube_b, OP_y_cube_b = distance * math.sin(-2 * theta) - 0.1035, distance * math.cos(-2 * theta)
    # OP_x_sphere_r, OP_y_sphere_r = distance * math.sin(-3 * theta) - 0.1035, distance * math.cos(-3 * theta)
    # OP_x_cyl_y, OP_y_cyl_y = distance * math.sin(-4 * theta) - 0.1035, distance * math.cos(-4 * theta)

    OP_x_cube, OP_y_cube = -0.0, distance
    OP_x_cyl, OP_y_cyl = distance * math.sin(theta), distance * math.cos(theta)
    OP_x_sphere, OP_y_sphere = -distance * math.sin(theta), distance * math.cos(theta)
    # OP_x_sphere, OP_y_sphere = distance * math.sin(-4 * theta), distance * math.cos(-4 * theta)
    OP_x_cube_y, OP_y_cube_y = distance * math.sin(2 * theta), distance * math.cos(2 * theta)
    OP_x_cyl_r, OP_y_cyl_r = distance * math.sin(3 * theta), distance * math.cos(3 * theta)
    OP_x_sphere_b, OP_y_sphere_b = distance * math.sin(4 * theta), distance * math.cos(4 * theta)
    OP_x_cube_b, OP_y_cube_b = distance * math.sin(-2 * theta), distance * math.cos(-2 * theta)
    OP_x_sphere_r, OP_y_sphere_r = distance * math.sin(-3 * theta), distance * math.cos(-3 * theta)
    OP_x_cyl_y, OP_y_cyl_y = distance * math.sin(-4 * theta), distance * math.cos(-4 * theta)
    # OP_x_cyl_y, OP_y_cyl_y = -distance * math.sin(theta), distance * math.cos(theta)

    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cube, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cube, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cube', vrep.simx_opmode_oneshot)
    errorCode, objHandles[0] = vrep.simxGetObjectHandle(clientID, 'Cuboid', vrep.simx_opmode_blocking)
    obj_names.append('Cube_r')
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cyl, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cyl, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cylinder', vrep.simx_opmode_oneshot)
    errorCode, objHandles[1] = vrep.simxGetObjectHandle(clientID, 'Cylinder', vrep.simx_opmode_blocking)
    obj_names.append('Cylinder_b')
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_sphere, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_sphere, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Sphere', vrep.simx_opmode_oneshot)
    errorCode, objHandles[2] = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_blocking)
    obj_names.append('Sphere_y')
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cube_y, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cube_y, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cube_y', vrep.simx_opmode_oneshot)
    errorCode, objHandles[3] = vrep.simxGetObjectHandle(clientID, 'Cuboid_y', vrep.simx_opmode_blocking)
    obj_names.append('Cube_y')
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cyl_r, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cyl_r, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cylinder_r', vrep.simx_opmode_oneshot)
    errorCode, objHandles[4] = vrep.simxGetObjectHandle(clientID, 'Cylinder_r', vrep.simx_opmode_blocking)
    obj_names.append('Cylinder_r')
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_sphere_b, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_sphere_b, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Sphere_b', vrep.simx_opmode_oneshot)
    errorCode, objHandles[5] = vrep.simxGetObjectHandle(clientID, 'Sphere_b', vrep.simx_opmode_blocking)
    obj_names.append('Sphere_b')
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cube_b, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cube_b, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cube_b', vrep.simx_opmode_oneshot)
    errorCode, objHandles[6] = vrep.simxGetObjectHandle(clientID, 'Cuboid_b', vrep.simx_opmode_blocking)
    obj_names.append('Cube_b')
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_sphere_r, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_sphere_r, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Sphere_r', vrep.simx_opmode_oneshot)
    errorCode, objHandles[7] = vrep.simxGetObjectHandle(clientID, 'Sphere_r', vrep.simx_opmode_blocking)
    obj_names.append('Sphere_r')
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cyl_y, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cyl_y, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cylinder_y', vrep.simx_opmode_oneshot)
    errorCode, objHandles[8] = vrep.simxGetObjectHandle(clientID, 'Cylinder_y', vrep.simx_opmode_blocking)
    obj_names.append('Cylinder_y')

    print(obj_names)
    print(objHandles)

    vrep.simxSetStringSignal(clientID, 'Obj_type', 'All', vrep.simx_opmode_oneshot)

    return objHandles


def choose_desired_object():
    # print('Choosing object')
    global obj_choice

    errorCode, obj_disp_Handle = vrep.simxGetObjectHandle(clientID, 'Object_display', vrep.simx_opmode_blocking)
    obj_pic_list = ['Cube_r.PNG', 'Cylinder_b.PNG', 'Sphere_y.PNG', 'Cube_y.PNG', 'Cylinder_r.PNG', 'Sphere_b.PNG',
                    'Cube_b.PNG', 'Sphere_r.PNG', 'Cylinder_y.PNG']

    obj_choice_num = int(random.random()*9)
    obj_choice = obj_names[obj_choice_num]
    print('Object Choice: ')
    img_path = 'vrep_files/ObjectPics/' + obj_pic_list[obj_choice_num]
    print(img_path)
    try:
        # obj_img = Image.open(img_path)
        obj_img = cv2.imread(img_path)
        img_shape = obj_img.shape
        obj_img = cv2.resize(obj_img, (128, 128), interpolation=cv2.INTER_LINEAR)
        obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)
        obj_img = cv2.flip(obj_img, 0)
        obj_img = obj_img.ravel()
        returncode = vrep.simxSetVisionSensorImage(clientID, obj_disp_Handle, obj_img, 0, vrep.simx_opmode_blocking)
    except IOError:
        print('Could not find file ' + img_path)


def image_process(im, resolution):
    global obj_x, obj_y, obj_pix_prop, objHandle
    global im2
    min_i, min_j, max_i, max_j = 100, 100, 0, 0
    grasp_type_ret, obj_pix = 0, 0
    up_thresh = 0.7 * 255
    low_thresh = 0.3 * 255
    im2 = im

    for i in range(0, resolution[0]):
        for j in range(0, resolution[1]):
            grasp_type_t = 0
            if im[i, j, 0] > up_thresh and im[i, j, 1] < low_thresh and im[i, j, 2] < up_thresh:  # red
                grasp_type_t = 1  # cube (power + orientation)
            elif im[i, j, 0] < low_thresh and im[i, j, 1] < low_thresh and im[i, j, 2] > up_thresh:
                grasp_type_t = 2  # cylinder (grab from side)
            elif im[i, j, 0] > up_thresh and im[i, j, 1] > up_thresh and im[i, j, 2] < low_thresh:
                grasp_type_t = 3  # sphere

            if grasp_type_t > 0:
                obj_pix = obj_pix + 1
                grasp_type_ret = grasp_type_t
                if i < min_i:
                    min_i = i
                if j < min_j:
                    min_j = j
                if i > max_i:
                    max_i = i
                if j > max_j:
                    max_j = j

    i_pos = (max_i + min_i) / 2.0
    j_pos = (max_j + min_j) / 2.0

    obj_pix_prop = obj_pix/(resolution[0]*resolution[1]*1.00)

    obj_x, obj_y = int(j_pos), int(i_pos)

    # visual_feedback()

    return [grasp_type_ret, j_pos, i_pos, obj_pix_prop]


def image_process_2(im, resolution, bci_class):

    global obj_x, obj_y, obj_pix_prop, objHandle
    global im2
    min_i, min_j, max_i, max_j = 100, 100, 0, 0
    grasp_type_ret, obj_pix = 0, 0
    up_thresh = 0.7 * 255
    low_thresh = 0.3 * 255
    im2 = im

    cube_x, cube_y, cyl_x, cyl_y, sphere_x, sphere_y = [], [], [], [], [], []
    obj_list, cost_list = [], []

    for i in range(0, resolution[0]):
        for j in range(0, resolution[1]):
            grasp_type_t = 0
            # print(i, j)
            if im[i, j, 0] > up_thresh and im[i, j, 1] < low_thresh and im[i, j, 2] < low_thresh:  # red
                cube_x.append(j); cube_y.append(i)
                grasp_type_t = 1  # cube (power + orientation)
            elif im[i, j, 0] < low_thresh and im[i, j, 1] < low_thresh and im[i, j, 2] > up_thresh:
                cyl_x.append(j); cyl_y.append(i)
                grasp_type_t = 2  # cylinder (grab from side)
            elif im[i, j, 0] > up_thresh and im[i, j, 1] > up_thresh and im[i, j, 2] < low_thresh:
                sphere_x.append(j); sphere_y.append(i)
                grasp_type_t = 3  # sphere

    obj_list = obj_process(cube_x, cube_y, obj_list, 'Cube')
    obj_list = obj_process(cyl_x, cyl_y, obj_list, 'Cyl')
    obj_list = obj_process(sphere_x, sphere_y, obj_list, 'Sphere')

    # print(obj_list)

    if bci_class > 0:
        obj_list = obj_process_bci(obj_list, bci_class)

    for k in range(len(obj_list)):
        cost_list.append(obj_list[k][4])
        # print obj_list[k][0], obj_list[k][4]

    if len(cost_list) > 0:
        a = cost_list.index(max(cost_list))

        if 'objHandles' in globals():
            obj_x, obj_y, obj_pix_prop, objHandle = int(obj_list[a][1]), int(obj_list[a][2]), obj_list[a][3], int(objHandles[a])
        else:
            obj_x, obj_y, obj_pix_prop, objHandle = int(obj_list[a][1]), int(obj_list[a][2]), obj_list[a][3], objHandle

    visual_feedback()

    return obj_list


def obj_process(x_obj, y_obj, obj_list, type_obj):
    if len(x_obj) > 0:
        x_cent_cam, y_cent_cam = resolution[0]/2, resolution[1]/2
        max_x, max_y, min_x, min_y = max(x_obj), max(y_obj), min(x_obj), min(y_obj)
        x_pos = (max_x + min_x) / 2.0
        y_pos = (max_y + min_y) / 2.0
        obj_pix_prop = len(x_obj) / (resolution[0] * resolution[1] * 1.00)

        pix_cost = 5 * obj_pix_prop
        dist_cost = math.sqrt((x_pos - x_cent_cam) ** 2 + (y_pos-y_cent_cam) ** 2) / resolution[0]
        tot_cost = pix_cost - dist_cost
        obj_list.append([type_obj, x_pos, y_pos, obj_pix_prop, tot_cost])  # Cube, Cyl, Sphere = 1, 2, 3

    return obj_list


def obj_process_bci(obj_list, bci_class):
    x_cent_cam, y_cent_cam = resolution[0] / 2, resolution[1] / 2

    for k in range(len(obj_list)):
        ox, oy, opp = obj_list[k][1], obj_list[k][2], obj_list[k][3]
        pix_cost = 5 * opp
        if bci_class == 1:      # Right
            x_cent_cam = x_cent_cam + resolution[0]/4
        elif bci_class == 2:    # Left
            x_cent_cam = x_cent_cam - resolution[0]/4

        dist_cost = math.sqrt((ox-x_cent_cam) ** 2 + (oy-y_cent_cam) ** 2) / resolution[0]
        bci_cost = opp - dist_cost
        obj_list[k][4] = bci_cost

    return obj_list


def get_bci_class_perfect(bci_iter):
    # 0: None     1: Left     2: Right    3: Both
    bci_freq = 160
    class_freq = 4.0
    bci_task_len = 5

    bci_class = [1, 3, 0, 0,  # Right object - check
                 2, 3, 0, 0,  # Left object - check (sloppy)
                 0, 3, 0, 0,  # Center object - check
                 1, 2, 1, 3,  # Right object - check
                 2, 1, 2, 3,  # Left object - check
                 1, 3, 2, 0,  # Center object - check
                 2, 3, 1, 0,  # Center object - check
                 2, 3, 4, 0,  # None - check
                 1, 3, 4, 0,  # None - check
                 0, 3, 4, 0,  # None - check
                 0, 3, 1, 0,  # Right object - check
                 0, 3, 2, 0,  # Left object - FAIL - leaves image too quickly
                 3, 1, 4, 0,  # None - FAIL - gets to approach2 too quickly
                 3, 2, 4, 0,  # None - FAIL -
                 1, 2, 1, 3,  # Right object - check
                 2, 1, 2, 3,  # Left object - check (sloppy)
                 3, 4, 2, 3,  # Left object - check (sloppy)
                 3, 4, 1, 3,  # Right object - check
                 3, 4, 1, 2,  # None - check
                 3, 4, 2, 1]  # None - check

    bci_class = np.asarray(bci_class)
    bci_class_exp = np.repeat(bci_class, class_freq*bci_task_len)
    # time_vec = np.arange(1, 1 + int(bci_class_exp.shape[0] * 1/class_freq), 1/class_freq)

    print(bci_iter, bci_class_exp[bci_iter])

    return bci_class_exp[bci_iter]


def init_marker_receiver_vrep():
    global final_markers, markers_ts
    # markers_receiver = LSLa.lslReceiver('markers', True, True)
    returncode, signal_val = vrep.simxGetIntegerSignal(clientID, 'BCI_class', vrep.simx_opmode_streaming)
    final_markers, markers_ts = [], []
    return


def read_bci_markers_vrep(robot_action):
    returncode, signal_val = vrep.simxGetIntegerSignal(clientID, 'BCI_class', vrep.simx_opmode_buffer)
    if signal_val > 0:
        vrep.simxClearIntegerSignal(clientID, 'BCI_class', vrep.simx_opmode_oneshot)
        if signal_val == 100:
            exit()
        elif signal_val == 10:
            print('Returning to starting position')
            return 6
        signal_val = signal_val - 1
        if len(final_markers) > 0:
            final_markers.append(signal_val)
            # markers_ts.append(time.process_time())
            markers_ts.append(time.perf_counter())
        else:
            # init_ts_m = time.process_time()
            init_ts_m = time.perf_counter()
            final_markers.append(signal_val)
            markers_ts.append(init_ts_m)

            real_time_BCI.sync_streams(init_ts_m)

        print(signal_val)

    return robot_action
    # if signal_val != nil:


def read_success_data_train():
    # print("reading training success data")
    # Get position of real robot end-effector
    checkO, gripper_pos = vrep.simxGetObjectPosition(clientID, full_gripper_handle, -1, vrep.simx_opmode_oneshot)
    # Get position of ghost end-effector
    checkO, gripper_pos_g = vrep.simxGetObjectPosition(clientID, full_ghost_gripper_handle, -1, vrep.simx_opmode_oneshot)
    # Find distance between them
    dist = math.sqrt((gripper_pos[0] - gripper_pos_g[0])**2 + (gripper_pos[1] - gripper_pos_g[1])**2 + (gripper_pos[2] - gripper_pos_g[2])**2)
    # Add distance to array
    gripper_distance.append(dist)


def read_success_data_test(grasped_object_handle, start_time, success):
    print("reading testing success data")
    # After grasp, add information about which object has been grasped, which was supposed to be, if they match,
    # how long the user took to complete the task, and so on

    time_taken = time.perf_counter() - start_time

    print(obj_choice)
    print(grasped_object)
    success_data.append([obj_choice, grasped_object, success, time_taken])


def save_data(feedback=True, vrep=True, markers=True, mode='train'):
    real_time_BCI.update_markers(final_markers, markers_ts)
    real_time_BCI.save_data(feedback=feedback, vrep=vrep, markers=markers, mode=mode)


def save_success_data(subject_name="unworn", mode='test'):
    if mode == 'train':
        np.savetxt(subject_name + "_dist_data.csv", gripper_distance, delimiter=",")
    if mode == 'test':
        print(np.asarray(success_data))
        np.savetxt(subject_name + "_test_success.csv", np.asarray(success_data), delimiter=",", fmt='%s')


def depth_process(im_d):
    im_max = np.amax(im_d)
    im_min = np.amin(im_d)

    im_d = (im_d - im_min)/(im_max-im_min)

    im_d_out = np.asarray(im_d).reshape(resolution[0], resolution[1])
    im_d_out *= 256

    return im_d_out


def center_object(weight, im_x, im_y, resolution):
    # print('Center!')
    vrep.simxSetObjectIntParameter(clientID, joint_handles[0], vrep.sim_jointintparam_ctrl_enabled, 0,
                                   vrep.simx_opmode_oneshot)
    vrep.simxSetObjectIntParameter(clientID, joint_handles[4], vrep.sim_jointintparam_ctrl_enabled, 0,
                                   vrep.simx_opmode_oneshot)

    # if EE_turn == 1:
    #     im_x = im_y
    #     im_y = im_x

    vel_x = weight * (resolution[0] / 2 - im_x)
    if abs(vel_x) > weight * 5:
        vel_x = np.sign(vel_x) * weight * 5
    # vel_x = -vel_x # if KUKA
    returnCode = vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], vel_x, vrep.simx_opmode_oneshot)
    # print returnCode

    vel_y = weight * (resolution[1] / 2 - im_y)
    if abs(vel_y) > weight * 8:
        vel_y = np.sign(vel_y) * weight * 10
    # vel_y = -vel_y # if KUKA
    returnCode = vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], vel_y, vrep.simx_opmode_oneshot)

    return


def full_center(im_x, im_y, center_res, resolution):
    print('Full Center!')

    # tic_fc = time.process_time()
    tic_fc = time.perf_counter()
    while not (((resolution[1]) / 2 - center_res < im_x < (resolution[0]) / 2 + center_res) and
               ((resolution[1]) / 2 - center_res < im_y < (resolution[1]) / 2 + center_res)):

        center_object(.02, im_x, im_y, resolution)

        errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                      vrep.simx_opmode_buffer)
        im = np.array(image, dtype=np.uint8)
        if len(resolution) > 0:
            im.resize([resolution[0], resolution[1], 3])
            # [grasp_type, im_x, im_y, obj_pix_prop] = image_process(im, resolution)
            obj_list = image_process_2(im, resolution, 0)
            if len(obj_list) > 0:
                im_x, im_y = obj_list[0][1], obj_list[0][2]

        # toc_fc = time.process_time()
        toc_fc = time.perf_counter()
        if toc_fc - tic_fc > 10:
            return 6

    else:
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], 0, vrep.simx_opmode_oneshot)
        print('Centered!')
        return 0


def search_object():
    print('Searching...')
    vrep.simxSetStringSignal(clientID, 'IKCommands', 'LookPos', vrep.simx_opmode_oneshot)  # "Looking" position
    center_res = 10
    vrep.simxSetObjectIntParameter(clientID, joint_handles[0], vrep.sim_jointintparam_ctrl_enabled, 0,
                                   vrep.simx_opmode_oneshot)
    vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0.5, vrep.simx_opmode_streaming)

    # tic_so = time.process_time()
    tic_so = time.perf_counter()
    objectFound = False
    while objectFound is False:
        # toc_so = time.process_time()
        toc_so = time.perf_counter()

        errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0, vrep.simx_opmode_buffer)
        im = np.array(image, dtype=np.uint8)
        if len(resolution) > 0:
            im.resize([resolution[0], resolution[1], 3])
            [grasp_type, im_x, im_y, obj_pix_prop] = image_process(im, resolution)

            if grasp_type > 0:
                print('Object Found!')
                print('Grasp Type: ', grasp_type)
                full_center(im_x, im_y, center_res, resolution)
                # lock_all_joints()
                # center_object(0.0025, im_x, im_y, resolution)
                return 1
            elif toc_so-tic_so > 20:
                return 6


def search_object_2():
    print('Searching...')
    vrep.simxSetStringSignal(clientID, 'IKCommands', 'LookPos', vrep.simx_opmode_oneshot)  # "Looking" position
    center_res = 10
    # tic_so = time.process_time()
    tic_so = time.perf_counter()
    objectFound = False

    while objectFound is False:
        # toc_so = time.process_time()
        toc_so = time.perf_counter()

        errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0, vrep.simx_opmode_buffer)
        im = np.array(image, dtype=np.uint8)
        if len(resolution) > 0:
            im.resize([resolution[0], resolution[1], 3])
            obj_list = image_process_2(im, resolution)

            if len(obj_list) > 0:
                print(obj_list[0])
                if len(obj_list[0]) > 1:
                    im_x = obj_list[0][1]
                    im_y = obj_list[0][2]
                    print('Object Found!')
                    full_center(im_x, im_y, center_res, resolution)
                    return 1
        errorCode, joint_ang_0 = vrep.simxGetJointPosition(clientID, joint_handles[0], vrep.simx_opmode_buffer)
        errorCode, joint_ang_4 = vrep.simxGetJointPosition(clientID, joint_handles[4], vrep.simx_opmode_buffer)
        if joint_ang_0 > 70 * math.pi / 180:
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], -0.2, vrep.simx_opmode_streaming)
        if joint_ang_4 < 100 * math.pi / 180:
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], -0.2, vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], 0, vrep.simx_opmode_streaming)
        if joint_ang_0 < -70 * math.pi / 180:
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], 0.2, vrep.simx_opmode_streaming)
        if joint_ang_4 > 118 * math.pi / 180:
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0.2, vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], 0, vrep.simx_opmode_streaming)

        if toc_so-tic_so > 20:
            return 6


def search_object_bci(bci_iter, this_bci_iter, next_bci_iter, clf, num_channels=22, mode='train'):
    time_limit = False

    vrep.simxSetStringSignal(clientID, 'IKCommands', 'LookPos', vrep.simx_opmode_oneshot)  # "Looking" position
    vrep.simxSetObjectIntParameter(clientID, joint_handles[0], vrep.sim_jointintparam_velocity_lock, 0,
                                   vrep.simx_opmode_oneshot)
    center_res = 10
    bci_update = 0.01

    # tic_so = time.process_time()
    tic_so = time.perf_counter()
    tic_bci = tic_so
    objectFound = False
    five_class_control = False
    four_class_control = True
    print('num_channels: ', num_channels)
    next_state = read_bci_markers_vrep(0)
    bci_class, cert = real_time_BCI.get_bci_class(bci_iter, clf, num_channels=num_channels)
    # get_bci_class_lsl
    bci_iter = bci_iter + 1

    vrep.simxSetStringSignal(clientID, 'robot_state', 'searching', vrep.simx_opmode_blocking)

    while objectFound is False:
        # toc_so = time.process_time()
        toc_so = time.perf_counter()
        # print('time_diff: ', toc_so - tic_bci)

        if mode == 'train':
            read_success_data_train()

        # print bci_iter, next_bci_iter
        if bci_iter > next_bci_iter - 1:
            return bci_iter, 6

        if toc_so - tic_bci > bci_update:
            next_state = read_bci_markers_vrep(0)
            bci_class, cert = real_time_BCI.get_bci_class(bci_iter, clf)
            # tic_bci = time.process_time()
            tic_bci = time.perf_counter()
            bci_iter = bci_iter + 1
            if np.isnan(cert):
                # cert = 0.2
                cert = 0.02
            # print bci_time, tic_bci

        if five_class_control:
            if bci_class == 1:
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0.065, vrep.simx_opmode_oneshot)
            elif bci_class == 2:
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], -0.065, vrep.simx_opmode_oneshot)
            elif bci_class == 0:
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
            elif bci_class == 4:
                vrep.simxSetObjectIntParameter(clientID, joint_handles[0], vrep.sim_jointintparam_velocity_lock, 1,
                                               vrep.simx_opmode_oneshot)
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
            elif bci_class == 3:
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
                objectFound = True
        elif four_class_control:
            if bci_class == 0:          # Turns left
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0.065 * cert, vrep.simx_opmode_oneshot)
            elif bci_class == 1:        # Turns right
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], -0.065 * cert, vrep.simx_opmode_oneshot)
            elif bci_class == 2:        # Starts approach
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
                objectFound = True
            elif bci_class == 3:        # Stops movement
                vrep.simxSetObjectIntParameter(clientID, joint_handles[0], vrep.sim_jointintparam_velocity_lock, 1,
                                               vrep.simx_opmode_oneshot)
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)

        errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0, vrep.simx_opmode_buffer)
        im = np.array(image, dtype=np.uint8)
        if len(resolution) > 0:
            im.resize([resolution[0], resolution[1], 3])
            image_process_2(im, resolution, bci_class)

        if next_state == 6:
            return bci_iter, 6

    return bci_iter, 1


def start_pos(joint_start, start_time, mode='train'):
    # print('StartPos!')
    global objHandle
    success, max_height, i = 0, 0, 0
    grasped_object_found = False

    joint_pos, joint_success = np.zeros(len(joint_start)), np.zeros(len(joint_start))

    for i in range(0, len(joint_start), 1):
        returncode, joint_pos[i] = vrep.simxGetJointPosition(clientID, joint_handles[i], vrep.simx_opmode_buffer)
        vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_ctrl_enabled, 0,
                                       vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_velocity_lock, 1,
                                       vrep.simx_opmode_oneshot)

    vrep.simxSetJointTargetVelocity(clientID, joint_handles[1], 0.5, vrep.simx_opmode_oneshot)
    tic_sp = time.time()
    while sum(joint_success) < len(joint_start):
        if not grasped_object_found:
            for handle in objHandles:
                checkO, obj_pos = vrep.simxGetObjectPosition(clientID, int(handle), -1, vrep.simx_opmode_blocking)
                if obj_pos[2] > 0.07:
                    print('Object is higher than 0.07. Height: ', obj_pos[2])
                    objHandle = int(handle)
                    grasped_object_found = True

        try:
            checkO, obj_pos = vrep.simxGetObjectPosition(clientID, objHandle, -1, vrep.simx_opmode_blocking)
        except NameError:
            obj_pos = [0, 0, 0]
        toc_sp = time.time()
        if obj_pos[2] > max_height:
            max_height = obj_pos[2]

        for i in range(0, len(joint_start)):
            returncode, joint_pos[i] = vrep.simxGetJointPosition(clientID, joint_handles[i], vrep.simx_opmode_buffer)

            if joint_pos[i] < joint_start[i] - 0.2 or joint_pos[i] > joint_start[i] + 0.2:
                joint_vel = 0.2*(joint_start[i] - joint_pos[i])
                # if i == 3 and joint_vel > 0:
                #     joint_vel = -1 * joint_vel

                vrep.simxSetJointTargetVelocity(clientID, joint_handles[i], joint_vel, vrep.simx_opmode_oneshot)

                if toc_sp - tic_sp > 10:
                    vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_ctrl_enabled, 1,
                                                   vrep.simx_opmode_oneshot)
                    vrep.simxSetJointTargetVelocity(clientID, joint_handles[i], 0, vrep.simx_opmode_oneshot)
                    joint_success[i] = 1
            else:
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[i], 0, vrep.simx_opmode_oneshot)
                joint_success[i] = 1

        # print(obj_pos[2])
        if sum(joint_success) == 6 or max_height > 0.22:
            vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_ctrl_enabled, 1,
                                           vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[i], 0, vrep.simx_opmode_oneshot)

            if max_height > 0.22:
                print('success!')
                success = 1

            if mode == 'test':
                read_success_data_test(objHandle, start_time, success)

            return [success, max_height]

    return [success, max_height]


def start_pos_grip(grip_joint_start):
    # print('Gripper Start Pos!')
    success, max_height, i = 0, 0, 0

    grip_joint_pos, grip_joint_success = np.zeros(len(grip_joint_start)), np.zeros(len(grip_joint_start))

    for i in range(0, len(grip_joint_start), 1):
        returncode, grip_joint_pos[i] = vrep.simxGetJointPosition(clientID, gripper_handles[i], vrep.simx_opmode_buffer)
        vrep.simxSetObjectIntParameter(clientID, gripper_handles[i], vrep.sim_jointintparam_ctrl_enabled, 0,
                                       vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, gripper_handles[i], vrep.sim_jointintparam_velocity_lock, 1,
                                       vrep.simx_opmode_oneshot)

    # tic_sp = time.process_time()
    tic_sp = time.perf_counter()
    while sum(grip_joint_success) < len(grip_joint_start) - 2:

        try:
            checkO, obj_pos = vrep.simxGetObjectPosition(clientID, objHandle, -1, vrep.simx_opmode_blocking)
        except NameError:
            obj_pos = [0, 0, 0]
        # toc_sp = time.process_time()
        toc_sp = time.perf_counter()
        if obj_pos[2] > max_height:
            max_height = obj_pos[2]

        for i in range(0, len(grip_joint_start)):
            returncode, grip_joint_pos[i] = vrep.simxGetJointPosition(clientID, gripper_handles[i], vrep.simx_opmode_buffer)

            if grip_joint_pos[i] < grip_joint_start[i] - 0.2 or grip_joint_pos[i] > grip_joint_start[i] + 0.2:
                grip_joint_vel = 0.2 * (grip_joint_start[i] - grip_joint_pos[i])
                vrep.simxSetJointTargetVelocity(clientID, gripper_handles[i], grip_joint_vel, vrep.simx_opmode_oneshot)

                if toc_sp - tic_sp > 1:
                    vrep.simxSetObjectIntParameter(clientID, gripper_handles[i], vrep.sim_jointintparam_ctrl_enabled, 1,
                                                   vrep.simx_opmode_oneshot)
                    vrep.simxSetJointTargetVelocity(clientID, gripper_handles[i], 0, vrep.simx_opmode_oneshot)
                    grip_joint_success[i] = 1
            else:
                vrep.simxSetJointTargetVelocity(clientID, gripper_handles[i], 0, vrep.simx_opmode_oneshot)
                grip_joint_success[i] = 1

        # print(obj_pos[2])
        if sum(grip_joint_success) == 8 or max_height > 0.2:
            if obj_pos[2] > 0.25:
                # print('success!')
                success = 1

    return [success, max_height]


def approach1(prop_thresh, vel_1, center_res, cent_weight, jps_thresh):
    # print('Approach1')

    vrep.simxSetJointTargetVelocity(clientID, joint_handles[1], vel_1, vrep.simx_opmode_oneshot)
    vrep.simxSetObjectIntParameter(clientID, joint_handles[1], vrep.sim_jointintparam_ctrl_enabled, 0,
                                   vrep.simx_opmode_oneshot)
    vrep.simxSetObjectIntParameter(clientID, joint_handles[1], vrep.sim_jointintparam_velocity_lock, 1,
                                   vrep.simx_opmode_oneshot)

    if obj_x > resolution[0]/2 + center_res or obj_x < resolution[0]/2 - center_res or \
       obj_y > resolution[1]/2 + center_res or obj_y < resolution[1]/2 - center_res:

        center_object(cent_weight, obj_x, obj_y, resolution)

    returncode, joint2_pos = vrep.simxGetJointPosition(clientID, joint_handles[1], vrep.simx_opmode_buffer)
    returncode, joint5_pos = vrep.simxGetJointPosition(clientID, joint_handles[4], vrep.simx_opmode_buffer)

    jointpossum = joint2_pos + joint5_pos
    # print(joint2_pos, joint5_pos)
    # print(jointpossum)

    if obj_pix_prop > prop_thresh or jointpossum > jps_thresh:
        print('Finished Approach1!')
        print('Object Pixel Proportion: ', obj_pix_prop)
        print('Joint Position Sum: ', jointpossum)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[1], 0, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], 0, vrep.simx_opmode_oneshot)
        return 2
    # elif obj_pix_prop < 0.001:
    #     return 6
    else:
        return 1


def approach1_bci(prop_thresh, vel_1, center_res, cent_weight, jps_thresh, bci_class, cert):
    # print('Approach1')
    # global bci_class, cert
    four_class_control = True

    # bci_class, cert = real_time_BCI.get_bci_class(bci_iter, clf)

    if cert > 0.5:
        if four_class_control:
            if bci_class == 0:  # Turns left
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[1], 0, vrep.simx_opmode_oneshot)
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0.065 * cert, vrep.simx_opmode_oneshot)
            elif bci_class == 1:  # Turns right
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[1], 0, vrep.simx_opmode_oneshot)
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], -0.065 * cert, vrep.simx_opmode_oneshot)
            elif bci_class == 3:  # Moves backward
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[1], -vel_1 * cert, vrep.simx_opmode_oneshot)
            else:
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[1], cert * vel_1, vrep.simx_opmode_oneshot)
                vrep.simxSetObjectIntParameter(clientID, joint_handles[1], vrep.sim_jointintparam_ctrl_enabled, 0,
                                               vrep.simx_opmode_oneshot)
                vrep.simxSetObjectIntParameter(clientID, joint_handles[1], vrep.sim_jointintparam_velocity_lock, 1,
                                               vrep.simx_opmode_oneshot)

                if obj_x > resolution[0]/2 + center_res or obj_x < resolution[0]/2 - center_res or \
                   obj_y > resolution[1]/2 + center_res or obj_y < resolution[1]/2 - center_res:

                    center_object(cent_weight, obj_x, obj_y, resolution)
    else:
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[1], cert * vel_1 / 2, vrep.simx_opmode_oneshot)
        # vrep.simxSetJointTargetVelocity(clientID, joint_handles[1], vel_1, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, joint_handles[1], vrep.sim_jointintparam_ctrl_enabled, 0,
                                       vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, joint_handles[1], vrep.sim_jointintparam_velocity_lock, 1,
                                       vrep.simx_opmode_oneshot)

        if obj_x > resolution[0] / 2 + center_res or obj_x < resolution[0] / 2 - center_res or \
                obj_y > resolution[1] / 2 + center_res or obj_y < resolution[1] / 2 - center_res:
            center_object(cent_weight, obj_x, obj_y, resolution)

    returncode, joint2_pos = vrep.simxGetJointPosition(clientID, joint_handles[1], vrep.simx_opmode_buffer)
    returncode, joint5_pos = vrep.simxGetJointPosition(clientID, joint_handles[4], vrep.simx_opmode_buffer)

    jointpossum = joint2_pos + joint5_pos
    # print(joint2_pos, joint5_pos)
    # print(jointpossum)

    if obj_pix_prop > prop_thresh or jointpossum > jps_thresh:
        print('Finished Approach1!')
        print('Object Pixel Proportion: ', obj_pix_prop)
        print('Joint Position Sum: ', jointpossum)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[1], 0, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], 0, vrep.simx_opmode_oneshot)
        return 2
    # elif obj_pix_prop < 0.001:
    #     return 6
    else:
        return 1


def approach2(prop_thresh, vel_2, center_res, cent_weight):
    # print("Approach2!")
    vrep.simxSetStringSignal(clientID, 'robot_state', 'grasping', vrep.simx_opmode_oneshot)

    vrep.simxSetJointTargetVelocity(clientID, joint_handles[3], vel_2, vrep.simx_opmode_oneshot)
    # vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], -vel_2/2, vrep.simx_opmode_oneshot)
    vrep.simxSetObjectIntParameter(clientID, joint_handles[3], vrep.sim_jointintparam_ctrl_enabled, 0,
                                   vrep.simx_opmode_oneshot)
    vrep.simxSetObjectIntParameter(clientID, joint_handles[3], vrep.sim_jointintparam_velocity_lock, 1,
                                   vrep.simx_opmode_oneshot)

    if obj_x > resolution[0]/2 + center_res or obj_x < resolution[0]/2 - center_res or \
       obj_y > resolution[1]/2 + center_res or obj_y < resolution[1]/2 - center_res:
        center_object(cent_weight, obj_x, obj_y, resolution)

    returncode, joint4_pos = vrep.simxGetJointPosition(clientID, joint_handles[3], vrep.simx_opmode_buffer)
    returncode, joint5_pos = vrep.simxGetJointPosition(clientID, joint_handles[4], vrep.simx_opmode_buffer)

    jointposdiff = joint4_pos - joint5_pos

    # print(jointposdiff)

    if obj_pix_prop > prop_thresh or jointposdiff > 3:
        print('Finished Approach2!')
        print('Object Pixel Proportion: ', obj_pix_prop)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[3], 0, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], 0, vrep.simx_opmode_oneshot)

        return 3
    elif obj_pix_prop < 0.001:
        return 6
    else:
        return 2


def final_grasp():
    print('Final Grasp!')
    gripper_type = 1
    if gripper_type == 2:
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], 0.04, vrep.simx_opmode_oneshot)
        time.sleep(2)
    time.sleep(1)
    vrep.simxSetStringSignal(clientID, 'Hand', 'true', vrep.simx_opmode_oneshot)
    time.sleep(5)
    return 6


def move_back(vel_mb):
    vrep.simxSetObjectIntParameter(clientID, joint_handles[3], vrep.sim_jointintparam_ctrl_enabled, 0,
                                   vrep.simx_opmode_oneshot)
    print('Move back')
    # move_back_count += 1
    # tic = time.clock()
    vrep.simxSetJointTargetVelocity(clientID, joint_handles[3], -vel_mb * 2, vrep.simx_opmode_oneshot)
    time.sleep(3)
    vrep.simxSetJointTargetVelocity(clientID, joint_handles[3], 0, vrep.simx_opmode_oneshot)
    time.sleep(3)
    return 2


def turn_EE():
    global EE_turn
    center_res = 10
    vrep.simxSetObjectIntParameter(clientID, joint_handles[5], vrep.sim_jointintparam_ctrl_enabled, 0,
                                   vrep.simx_opmode_oneshot)
    print('Turn EE!')
    if EE_turn == 0:
        EE_turn = 1
        returncode, joint6_pos = vrep.simxGetJointPosition(clientID, joint_handles[5], vrep.simx_opmode_buffer)
        targetpos6 = joint6_pos + math.pi/2
        print('Target Angle: ', targetpos6)
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[5], 0.1, vrep.simx_opmode_oneshot)

        while joint6_pos < targetpos6:
            returncode, joint6_pos = vrep.simxGetJointPosition(clientID, joint_handles[5], vrep.simx_opmode_buffer)

            if joint6_pos >= targetpos6:
                vrep.simxSetJointTargetVelocity(clientID, joint_handles[5], 0, vrep.simx_opmode_oneshot)
                return 1

            errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                          vrep.simx_opmode_buffer)
            im = np.array(image, dtype=np.uint8)
            if len(resolution) > 0:
                im.resize([resolution[0], resolution[1], 3])
                [grasp_type, im_x, im_y, obj_pix_prop] = image_process(im, resolution)

                if not (((resolution[0]) / 2 - center_res < im_x < (resolution[0]) / 2 + center_res) and
                        ((resolution[1]) / 2 - center_res < im_y < (resolution[1]) / 2 + center_res)):

                    center_object(.02, im_x, im_y, resolution)
    #elif EE_turn == 1:
    #    returncode, joint6_pos = vrep.simxGetJointPosition(clientID, Joint6_Handle, vrep.simx_opmode_buffer)
    #    targetpos6 = joint6_pos - math.pi/2
    #    while joint6_pos > joint6_pos - math.pi / 2:
    #        vrep.simxSetJointTargetVelocity(clientID, Joint6_Handle, -0.1, vrep.simx_opmode_oneshot)
    #        returncode, joint6_pos = vrep.simxGetJointPosition(clientID, Joint6_Handle, vrep.simx_opmode_buffer)
    #        # print('Joint 6: ', joint6_pos)
    #        if joint6_pos <= targetpos6:
    #            vrep.simxSetJointTargetVelocity(clientID, Joint6_Handle, 0, vrep.simx_opmode_oneshot)
    #            EE_turn = 0
    #            return


def get_joints_pos():
    joint_pos = np.zeros(len(joint_handles))
    for i in range(0, len(joint_handles), 1):
        returncode, joint_pos[i] = vrep.simxGetJointPosition(clientID, joint_handles[i], vrep.simx_opmode_buffer)
    return joint_pos

def get_joints_vel():
    joint_vel = np.zeros(len(joint_handles))
    for i in range(0, len(joint_handles), 1):
        returncode, joint_vel[i] = vrep.simxGetObjectFloatParameter(clientID, joint_handles[i], 2012, vrep.simx_opmode_buffer)
    return joint_vel

def get_gripper_joints_pos():
    gripper_pos = np.zeros(len(gripper_handles))
    for i in range(0, len(gripper_handles), 1):
        returncode, gripper_pos[i] = vrep.simxGetJointPosition(clientID, gripper_handles[i], vrep.simx_opmode_buffer)
    return gripper_pos

def get_gripper_vel():
    gripper_vel = np.zeros(len(gripper_handles))
    for i in range(0, len(gripper_handles), 1):
        returncode, gripper_vel[i] = vrep.simxGetObjectFloatParameter(clientID, gripper_handles[i], 2012, vrep.simx_opmode_buffer)
    return gripper_vel

def set_joint_vels(joint_act):
    for i in range(0, len(joint_act)):
        # print('joint act shape: {}'.format(joint_act.shape))
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[i], joint_act[i], vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_ctrl_enabled, 0,
                                       vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_velocity_lock, 1,
                                       vrep.simx_opmode_oneshot)

def set_gripper_vels(gripper_act):
    for i in range(0, len(gripper_act)):
        vrep.simxSetJointTargetVelocity(clientID, gripper_handles[i], gripper_act[i], vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, gripper_handles[i], vrep.sim_jointintparam_ctrl_enabled, 0,
                                       vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, gripper_handles[i], vrep.sim_jointintparam_velocity_lock, 1,
                                       vrep.simx_opmode_oneshot)


def lock_all_joints():
    for i in range(0, len(joint_handles)):
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[i], 0, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, joint_handles[i],
                                       vrep.sim_jointintparam_velocity_lock, 1,
                                       vrep.simx_opmode_oneshot)
    time.sleep(2)


def read_FS():
    global gripper_handles

    penalty = 0

    returnCode, state, forceVector0, torqueVector0 = vrep.simxReadForceSensor(clientID, FS_handles[0], vrep.simx_opmode_buffer)
    returnCode, state, forceVector1, torqueVector1 = vrep.simxReadForceSensor(clientID, FS_handles[1], vrep.simx_opmode_buffer)
    returnCode, state, forceVector2, torqueVector2 = vrep.simxReadForceSensor(clientID, FS_handles[2], vrep.simx_opmode_buffer)

    # print forceVector0, torqueVector0
    # print forceVector1, torqueVector1
    # print forceVector2, torqueVector2

    force0 = (forceVector0[0] ** 2 + forceVector0[1] ** 2 + forceVector0[2] ** 2) ** (0.5)
    force1 = (forceVector1[0] ** 2 + forceVector1[1] ** 2 + forceVector1[2] ** 2) ** (0.5)
    force2 = (forceVector2[0] ** 2 + forceVector2[1] ** 2 + forceVector2[2] ** 2) ** (0.5)

    # print force0, force1, force2

    av_force = np.mean([force0, force1, force2])
    force_dist = (abs(force0 - force1) + abs(force1 - force2) + abs(force2 - force0)) / 3
    force_dist_cost = 3 * (1 - np.exp(force_dist - 3))

    if force_dist_cost < -20:
        force_dist_cost = -20

    print(av_force, force_dist, force_dist_cost)

    returnCode, hand_joint_0 = vrep.simxGetJointPosition(clientID, gripper_handles[2], vrep.simx_opmode_buffer)
    returnCode, hand_joint_1 = vrep.simxGetJointPosition(clientID, gripper_handles[3], vrep.simx_opmode_buffer)
    returnCode, hand_joint_2 = vrep.simxGetJointPosition(clientID, gripper_handles[4], vrep.simx_opmode_buffer)

    if hand_joint_0 > 100 * math.pi / 180 or hand_joint_0 < 40 * math.pi / 180:
        penalty = penalty + 10
    if hand_joint_1 > 100 * math.pi / 180 or hand_joint_1 < 40 * math.pi / 180:
        penalty = penalty + 10
    if hand_joint_2 > 100 * math.pi / 180 or hand_joint_2 < 40 * math.pi / 180:
        penalty = penalty + 10

    return [av_force, force_dist_cost, penalty]


def visual_feedback():

    img2 = im2
    try:
        if 0 < obj_x < 126:
            # print('obj x: ', obj_x)
            obj_diff = int((obj_pix_prop**0.5)*100)
            if obj_diff < 15:
                obj_diff = 15
            cv2.rectangle(img2, (obj_x-obj_diff, obj_y-obj_diff), (obj_x+obj_diff, obj_y+obj_diff), (0, 255, 0), 1)
    except NameError:
        print('no object in view')

    img2 = img2.ravel()
    # print(img2.shape)

    returncode = vrep.simxSetVisionSensorImage(clientID, aug_Handle, img2, 0, vrep.simx_opmode_oneshot)


    return


def send_obj_sig(obj_type):
    if obj_type == 'Cube':
        vrep.simxSetIntegerSignal(clientID, 'sig', 4, vrep.simx_opmode_oneshot)
    elif obj_type == 'Cyl':
        vrep.simxSetIntegerSignal(clientID, 'sig', 1, vrep.simx_opmode_oneshot)
    elif obj_type == 'Sphere':
        vrep.simxSetIntegerSignal(clientID, 'sig', 4, vrep.simx_opmode_oneshot)
    return


def train_cv_network():
    global model
    # Try to find pre-trained model

    # If can't find, then build and train from scratch
    print('Training CV network!')
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    print(model.summary())

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    im_path = "D:/BCIRobotControl/training_images"

    train_images = []
    train_labels = []
    for im_name in os.listdir(im_path):
        # print(im_name)
        im = cv2.imread(im_path + "/" + im_name)
        im_d = im[:, :, 0]
        train_images.append(im_d)
        train_labels.append(int(im_name[0]))

    unique, counts = np.unique(train_labels, return_counts=True)
    print("Image labels (all): ", unique, counts)

    # shuffle training images and labels
    images_tr, labels_tr, images_val, labels_val = shuffle_split_ims(np.asarray(train_images), np.asarray(train_labels))

    unique, counts = np.unique(labels_tr, return_counts=True)
    print("Image labels (train): ", unique, counts)
    unique, counts = np.unique(labels_val, return_counts=True)
    print("Image labels (val): ", unique, counts)

    model.fit(images_tr, labels_tr, epochs=10, batch_size=16)
    model.evaluate(images_val, labels_val, batch_size=16)
    predictions = model.predict(images_val)
    pred_out = np.argmax(predictions, axis=1)
    confusion = confusion_matrix(labels_val, pred_out)
    print(confusion)


def shuffle_split_ims(ims, labels):
    ones, twos, threes = [], [], []
    labels = labels-1
    ims = ims / 256.0
    # one_count, two_count, three_count = 0, 0, 0
    for i in range(len(ims)):
        if labels[i] == 1:
            ones.append(ims)
        elif labels[i] == 2:
            if len(twos) < len(ones):
                twos.append(ims)
        elif labels[i] == 3:
            if len(threes) < len(twos) and len(threes) < len(ones):
                threes.append(ims)

    test_split = 0.8

    ims_out, labels_out = shuffle(ims, labels)
    im_train = ims_out[:int(test_split * len(ims_out))]
    im_train = im_train.reshape((im_train.shape[0], im_train.shape[1], im_train.shape[2], 1))
    labels_train = labels_out[:int(test_split * len(labels_out))]
    im_val = ims_out[int(test_split * len(ims_out)):]
    im_val = im_val.reshape((im_val.shape[0], im_val.shape[1], im_val.shape[2], 1))
    labels_val = labels_out[int(test_split * len(labels_out)):]

    return im_train, labels_train, im_val, labels_val


def classify_object(iter_num):
    print('Classifying object')
    save_object_im = False

    errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                  vrep.simx_opmode_buffer)
    errorCode2, resolution, image_d = vrep.simxGetVisionSensorDepthBuffer(clientID, cam_Handle,
                                                                          vrep.simx_opmode_buffer)

    image_d_out = depth_process(image_d)

    if save_object_im:

        # cv2.imshow('depth', image_d_out)
        im_save_name = 'training_images/' + str(iter_num) + '.png'
        cv2.imwrite(im_save_name, image_d_out)
        # im_num += 1

    classify_shape(image_d_out)
    im = np.array(image, dtype=np.uint8)
    classify_color(im)


def classify_shape(image_d):

    global grasped_object

    print('Predicted Shape: ')
    image_d = image_d/256.0
    image_d = image_d.reshape((1, image_d.shape[0], image_d.shape[1], 1))
    pred = model.predict(image_d)
    pred_out = np.argmax(pred)

    print(pred_out)
    if pred_out == 0:
        grasped_object = 'cube'
        print("Cube!")
    elif pred_out == 1:
        grasped_object = 'cylinder'
        print("Cylinder!")
    elif pred_out == 2:
        grasped_object = 'sphere'
        print("Sphere!")


def classify_color(image):
    global grasped_object

    if len(resolution) > 0:
        image.resize([resolution[0], resolution[1], 3])
        ret_obj = image_process(image, resolution)

        if ret_obj[0] == 1:
            grasped_object += '_r'
            print("Red!")
        elif ret_obj[0] == 2:
            grasped_object += '_b'
            print("Blue!")
        elif ret_obj[0] == 3:
            grasped_object += '_y'
            print("Yellow!")



