import vrep
import time
import random
import numpy as np
import math
import csv

# ROSpy does not support Python 3
import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import PointCloud2 as msg_PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import _Empty, Float32, Int8
# from rs2_listener import CWaitForMessage

# import eeg_io_pp, real_time_BCI

import cv2
import array
from PIL import Image as I


def sim_setup():
    vrep.simxFinish(-1)  # just in case, close all open connections

    global resolution, EE_turn
    global robot_joint_pub, robot_joint_sub, aligned_image_sub
    global robot_joints
    # global obj_x, obj_y, obj_pix_prop

    robot_joints = JointState()
    robot_joints.position = np.zeros(7)
    robot_joints.velocity = np.zeros(7)

    robot_joint_pub = rospy.Publisher('cmdJoints', JointState)
    # Augmented reality publisher???
    rospy.init_node('robot_experiment')
    robot_joint_sub = rospy.Subscriber('simJoints', JointState, simjoints_callback)
    aligned_image_sub = rospy.Subscriber('image_raw', msg_Image, rgb_image_callback)
    # msg_retriever = CWaitForMessage(msg_params) # look at rs2_test.py

    return True


def kuka_sim_setup(gripper_type):
    vrep.simxFinish(-1)  # just in case, close all open connections

    global resolution, EE_turn
    global robot_joint_pub, robot_joint_sub, aligned_image_sub
    global clientID, robot_joints, rgb_image, simjoints
    clientID = 0

    robot_joints = JointState()
    robot_joints.position = np.zeros(7)
    robot_joints.velocity = np.zeros(7)

    resolution = np.array([64, 64])
    rgb_image = np.zeros(resolution[0], resolution[1],3)

    simjoints = JointState()
    simjoints.position = np.zeros(7)
    simjoints.velocity = np.zeros(7)

    robot_joint_pub = rospy.Publisher('cmdJoints', JointState, queue_size=10)
    # Augmented reality publisher???
    rospy.init_node('robot_experiment')
    robot_joint_sub = rospy.Subscriber('simJoints', JointState, simjoints_callback)
    aligned_image_sub = rospy.Subscriber('image_raw', msg_Image, rgb_image_callback)
    # msg_retriever = CWaitForMessage(msg_params) # look at rs2_test.py

    setup_gripper(gripper_type)

    return True


def ABB_sim_setup(gripper_type):
    vrep.simxFinish(-1)  # just in case, close all open connections

    global resolution, EE_turn
    global robot_joint_pub, robot_joint_sub, aligned_image_sub
    global clientID, robot_joints, rgb_image, simjoints
    clientID = 0

    robot_joints = JointState()
    robot_joints.position = np.zeros(6)
    robot_joints.velocity = np.zeros(6)

    resolution = np.array([64, 64])
    rgb_image = np.zeros(resolution[0], resolution[1],3)

    simjoints = JointState()
    simjoints.position = np.zeros(6)
    simjoints.velocity = np.zeros(6)

    robot_joint_pub = rospy.Publisher('cmdJoints', JointState, queue_size=10)
    # Augmented reality publisher???
    rospy.init_node('robot_experiment')
    robot_joint_sub = rospy.Subscriber('simJoints', JointState, simjoints_callback)
    aligned_image_sub = rospy.Subscriber('image_raw', msg_Image, rgb_image_callback)
    # msg_retriever = CWaitForMessage(msg_params) # look at rs2_test.py

    setup_gripper(gripper_type)

    return True


def rgb_image_callback(data):
    global rgb_image
    rgb_image = np.array(map(ord, data.data))
    return


def simjoints_callback(data):
    global simjoints
    simjoints = data
    # print(simjoints)
    return

def setup_gripper(gripper_type):
    if gripper_type == 2:
        global run_sub, speed_sub, loop_sub, grasp_class_sub

        # run_sub = rospy.Subscriber('/svh_controller/toggle_run', _Empty)
        speed_sub = rospy.Subscriber('/svh_controller/speed', Float32)
        loop_sub = rospy.Subscriber('/svh_controller/loop_sub', Float32)
        grasp_class_sub = rospy.Subscriber('/svh_controller/grasp_class', Int8)

    # elif gripper_type == 3:
        # Barrett Hand

    return


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


def image_process_2():

    global obj_x, obj_y, obj_pix_prop, objHandle
    global im2
    min_i, min_j, max_i, max_j = 100, 100, 0, 0
    grasp_type_ret, obj_pix = 0, 0
    up_thresh = 0.7 * 255
    low_thresh = 0.3 * 255
    im = rgb_image
    im2 = im

    cube_x, cube_y, cyl_x, cyl_y, sphere_x, sphere_y = [], [], [], [], [], []
    obj_list, cost_list = [], []

    for i in range(0, resolution[0]):
        for j in range(0, resolution[1]):
            grasp_type_t = 0
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

    # if bci_class > 0:
    #     obj_list = obj_process_bci(obj_list, bci_class)

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

    # if EE_turn == 1:
    #     im_x = im_y
    #     im_y = im_x

    vel_x = weight * (resolution[0] / 2 - im_x)
    if abs(vel_x) > weight * 5:
        vel_x = np.sign(vel_x) * weight * 5
    vel_x = -vel_x # if KUKA
    returnCode = vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], vel_x, vrep.simx_opmode_oneshot)
    # robot_joints.velocity
    # print returnCode

    vel_y = weight * (resolution[1] / 2 - im_y)
    if abs(vel_y) > weight * 8:
        vel_y = np.sign(vel_y) * weight * 8
    vel_y = -vel_y # if KUKA
    returnCode = vrep.simxSetJointTargetVelocity(clientID, joint_handles[4], vel_y, vrep.simx_opmode_oneshot)
    # print returnCode
    # print joint_handles[0]
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

    robot_joints.velocity[0] = 0.5
    robot_joint_pub.publish(robot_joints)
    # vrep.simxSetJointTargetVelocity(clientID, joint_handles[0], 0.5, vrep.simx_opmode_streaming)

    tic_so = time.clock()
    objectFound = False
    while objectFound is False:
        toc_so = time.clock()
        robot_joints.velocity[0] = 0.5
        robot_joint_pub.publish(robot_joints)

        rgb_image.resize([resolution[0], resolution[1], 3])
        [grasp_type, im_x, im_y, obj_pix_prop] = image_process(rgb_image, resolution)
#
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
    # vrep.simxSetObjectIntParameter(clientID, joint_handles[0], vrep.sim_jointintparam_velocity_lock, 0,
    #                                vrep.simx_opmode_oneshot)

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

            robot_joint_pub.publish()
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
    print('StartPos!')
    success, max_height, i = 0, 0, 0
    joint_pos, joint_success = [0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0]
    # joint_start = [0, -100 * math.pi / 180, 30 * math.pi / 180, 85 * math.pi / 180, 118 * math.pi / 180, 0] # Active Arm
    # joint_start = [0, -20 * math.pi / 180, 0, -55 * math.pi / 180, 120 * math.pi / 180, 90 * math.pi / 180]  # KUKA

    joint_pos = simjoints.position

    robot_joints.velocity[1] = 0.2
    robot_joint_pub.publish(robot_joints)

    tic_sp = time.clock()
    while sum(joint_success) < 7:

        toc_sp = time.clock()
        joint_pos = simjoints.position

        for i in range(0, 7):
            if joint_pos[i] < joint_start[i] - 0.1 or joint_pos[i] > joint_start[i] + 0.1:
                joint_vel = 0.2*(joint_start[i] - joint_pos[i])
                # if i == 3 and joint_vel > 0:
                #     joint_vel = -1 * joint_vel

                # if i==6:
                #     print(joint_vel)

                robot_joints.velocity[i] = joint_vel
                robot_joint_pub.publish(robot_joints)

                # if toc_sp - tic_sp > 30:
                #     joint_success[i] = 1
            else:
                robot_joints.velocity[i] = 0
                robot_joint_pub.publish(robot_joints)
                joint_success[i] = 1

    for i in range(0,7):
        robot_joints.velocity[i] = 0

    time.sleep(2)

    robot_joint_pub.publish(robot_joints)

    return [success, max_height]


def approach1(prop_thresh, vel_1, center_res, cent_weight, jps_thresh):
    print('Approach1')
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
    print(joint2_pos, joint5_pos)
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
    # print('Final Grasp!')
    gripper_type = 2
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





def set_joint_vels(joint_act):
    for i in range(0, len(joint_act)):
        vrep.simxSetJointTargetVelocity(clientID, joint_handles[i], joint_act[i], vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_ctrl_enabled, 0,
                                       vrep.simx_opmode_oneshot)
        vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_velocity_lock, 1,
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
    if 0 < obj_x < 126:
        # print('obj x: ', obj_x)
        obj_diff = int((obj_pix_prop**0.5)*100)
        if obj_diff < 15:
            obj_diff = 15
        cv2.rectangle(img2, (obj_x-obj_diff, obj_y-obj_diff), (obj_x+obj_diff, obj_y+obj_diff), (0, 255, 0), 1)

    img2 = img2.ravel()
    # returncode = vrep.simxSetVisionSensorImage(clientID, aug_Handle, img2, 0, vrep.simx_opmode_oneshot)

    # print(returncode)

    return


def send_obj_sig(obj_type):
    if obj_type=='Cube':
        vrep.simxSetIntegerSignal(clientID, 'sig', 4, vrep.simx_opmode_oneshot)
    elif obj_type == 'Cyl':
        vrep.simxSetIntegerSignal(clientID, 'sig', 1, vrep.simx_opmode_oneshot)
    elif obj_type == 'Sphere':
        vrep.simxSetIntegerSignal(clientID, 'sig', 4, vrep.simx_opmode_oneshot)
    return
