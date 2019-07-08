import vrep_files.vrep as vrep
import time
import random
import numpy as np
import math
import csv

import tty, sys, termios

# import real_time_BCI
use_BCI = True
if use_BCI:
    import real_time_BCI_train as real_time_BCI
import eeg_io_pp

import cv2
import array
from PIL import Image as I


def sim_setup():
    vrep.simxFinish(-1)  # just in case, close all open connections

    global clientID
    global joint_handles, cam_Handle, aug_Handle
    global resolution, EE_turn
    # global obj_x, obj_y, obj_pix_prop

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
    if gripper_type == 2:           # Schunk 5-fingered hand
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
        # Barrett Hand Handles
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
    global objHandles
    objHandles = np.zeros(9)
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
    OP_x_cube_y, OP_y_cube_y = distance * math.sin(2 * theta), distance * math.cos(2 * theta)
    OP_x_cyl_r, OP_y_cyl_r = distance * math.sin(3 * theta), distance * math.cos(3 * theta)
    OP_x_sphere_b, OP_y_sphere_b = distance * math.sin(4 * theta), distance * math.cos(4 * theta)
    OP_x_cube_b, OP_y_cube_b = distance * math.sin(-2 * theta), distance * math.cos(-2 * theta)
    OP_x_sphere_r, OP_y_sphere_r = distance * math.sin(-3 * theta), distance * math.cos(-3 * theta)
    OP_x_cyl_y, OP_y_cyl_y = distance * math.sin(-4 * theta), distance * math.cos(-4 * theta)

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
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cube_y, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cube_y, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cube_y', vrep.simx_opmode_oneshot)
    errorCode, objHandles[3] = vrep.simxGetObjectHandle(clientID, 'Cuboid_y', vrep.simx_opmode_blocking)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cyl_r, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cyl_r, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cylinder_r', vrep.simx_opmode_oneshot)
    errorCode, objHandles[4] = vrep.simxGetObjectHandle(clientID, 'Cylinder_r', vrep.simx_opmode_blocking)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_sphere_b, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_sphere_b, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Sphere_b', vrep.simx_opmode_oneshot)
    errorCode, objHandles[5] = vrep.simxGetObjectHandle(clientID, 'Sphere_b', vrep.simx_opmode_blocking)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cube_b, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cube_b, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cube_b', vrep.simx_opmode_oneshot)
    errorCode, objHandles[5] = vrep.simxGetObjectHandle(clientID, 'Cube_b', vrep.simx_opmode_blocking)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_sphere_r, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_sphere_r, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Sphere_r', vrep.simx_opmode_oneshot)
    errorCode, objHandles[5] = vrep.simxGetObjectHandle(clientID, 'Sphere_r', vrep.simx_opmode_blocking)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_x', OP_x_cyl_y, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'ObjPos_y', OP_y_cyl_y, vrep.simx_opmode_oneshot)
    vrep.simxSetStringSignal(clientID, 'Obj_type', 'Cylinder_y', vrep.simx_opmode_oneshot)
    errorCode, objHandles[5] = vrep.simxGetObjectHandle(clientID, 'Cylinder_y', vrep.simx_opmode_blocking)

    vrep.simxSetStringSignal(clientID, 'Obj_type', 'All', vrep.simx_opmode_oneshot)

    return objHandles


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


def get_bci_class(bci_iter, clf, bci_freq=250, window_size=0.5, num_channels=22):
    buffer_size = int(bci_freq*window_size)
    label = [0]
    if bci_iter == 0:
        global buffer_data
        buffer_data = np.zeros((buffer_size, num_channels))
        print('num Channels: ', num_channels)
    buffer_data = real_time_BCI.iter_bci_buffer(buffer_data, bci_iter)
    x1 = eeg_io_pp.norm_dataset(buffer_data)
    x1 = x1.reshape(1, x1.shape[0], x1.shape[1])

    try:
        a, cert = real_time_BCI.predict(clf, x1, label)
    except ValueError:
        a, cert = 0, 0

    print(bci_iter, a, cert)

    return a, cert


def depth_process(im_d, resolution, obj_x, obj_y):
    # print ('max ', np.amax(im_d))
    # print np.amin(im_d)
    return


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
    tic_fc = time.clock()
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

        toc_fc = time.clock()
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

    tic_so = time.clock()
    objectFound = False
    while objectFound is False:
        toc_so = time.clock()

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
    tic_so = time.clock()
    objectFound = False

    while objectFound is False:
        toc_so = time.clock()

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


def search_object_bci(bci_iter, this_bci_iter, next_bci_iter, clf, num_channels=22):
    time_limit = False

    vrep.simxSetStringSignal(clientID, 'IKCommands', 'LookPos', vrep.simx_opmode_oneshot)  # "Looking" position
    vrep.simxSetObjectIntParameter(clientID, joint_handles[0], vrep.sim_jointintparam_velocity_lock, 0,
                                   vrep.simx_opmode_oneshot)
    center_res = 10
    bci_update = 0.25

    tic_so = time.clock()
    tic_bci = tic_so
    objectFound = False
    print('num_channels: ', num_channels)
    bci_class, cert = real_time_BCI.get_bci_class(bci_iter, clf, num_channels=num_channels)
    bci_iter = bci_iter + 1

    if time_limit:
        # while bci_time + 3 < tic_bci:
        while bci_iter < this_bci_iter:
            bci_class, cert = real_time_BCI.get_bci_class(bci_iter, clf)
            bci_iter = bci_iter + 1
            tic_bci = time.clock()

    while objectFound is False:
        toc_so = time.clock()
        # print('time_diff: ', toc_so - tic_bci)

        # print bci_iter, next_bci_iter
        if bci_iter > next_bci_iter - 1:
            return bci_iter, 6

        if toc_so - tic_bci > bci_update:
            bci_class, cert = real_time_BCI.get_bci_class(bci_iter, clf)
            tic_bci = time.clock()
            bci_iter = bci_iter + 1
            # print bci_time, tic_bci

        if bci_class == 1:
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], -0.065, vrep.simx_opmode_oneshot)
        elif bci_class == 2:
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0.065, vrep.simx_opmode_oneshot)
        elif bci_class == 0:
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
        elif bci_class == 4:
            vrep.simxSetObjectIntParameter(clientID, joint_handles[0], vrep.sim_jointintparam_velocity_lock, 1,
                                           vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
        elif bci_class == 3:
            vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0, vrep.simx_opmode_oneshot)
            objectFound = True

        errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0, vrep.simx_opmode_buffer)
        im = np.array(image, dtype=np.uint8)
        if len(resolution) > 0:
            im.resize([resolution[0], resolution[1], 3])
            image_process_2(im, resolution, bci_class)

    return bci_iter, 1


def start_pos(joint_start):
    # print('StartPos!')
    success, max_height, i = 0, 0, 0

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

    tic_sp = time.clock()
    while sum(grip_joint_success) < len(grip_joint_start) - 2:

        try:
            checkO, obj_pos = vrep.simxGetObjectPosition(clientID, objHandle, -1, vrep.simx_opmode_blocking)
        except NameError:
            obj_pos = [0, 0, 0]
        toc_sp = time.clock()
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


def approach2(prop_thresh, vel_2, center_res, cent_weight):
    # print("Approach2!")
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
    tic = time.clock()
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

    returncode = vrep.simxSetVisionSensorImage(clientID, aug_Handle, img2, 0, vrep.simx_opmode_oneshot)


    return


def send_obj_sig(obj_type):
    if obj_type=='Cube':
        vrep.simxSetIntegerSignal(clientID, 'sig', 4, vrep.simx_opmode_oneshot)
    elif obj_type == 'Cyl':
        vrep.simxSetIntegerSignal(clientID, 'sig', 1, vrep.simx_opmode_oneshot)
    elif obj_type == 'Sphere':
        vrep.simxSetIntegerSignal(clientID, 'sig', 4, vrep.simx_opmode_oneshot)
    return
