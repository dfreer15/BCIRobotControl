import vrep_files.vrep as vrep
import time
import numpy as np
import math
import csv
import Grasp_Learn_Func

print('Program started')
global EE_turn, joint_start
robot_type = 1      # 1 = Hamlyn Active Arm; 2 = KUKA; 3 = ABB IRB 140; 4 = KR10
gripper_type = 1    # 1 = Jaco; 2 = Schunk SVH; 3 = Barrett Hand; 4 = Salford Hand
use_BCI = True
learning_rl = False
set_pos = True
num_obj = 9
bci_update = 0.25
bci_iter = 0
mode = 'train'
# mode = 'test'

if learning_rl:
    import Learn_Final_Grasp
if use_BCI:
    if mode == 'train':
        import real_time_BCI_train as real_time_BCI
    elif mode == 'test':
        import real_time_BCI_test as real_time_BCI
    else:
        import real_time_BCI

    # clf_method = "Riemann"
    # clf_method = "Braindecode"
    clf_method = "LSTM"

    n_classes, epochs = 5, 20

    # dataset = "bci_comp"
    dataset = "gtec"

    if dataset == "bci_comp":
        num_channels = 22
    elif dataset == "gtec":
        num_channels = 32

if robot_type == 1:
    [connected, clientID, joint_handles, cam_Handle] = Grasp_Learn_Func.sim_setup()
    if gripper_type == 2:
        joint_start = [0, -100 * math.pi / 180, 30 * math.pi / 180, 85 * math.pi / 180, 30 * math.pi / 180, 0]
        pix_prop_thresh1, pix_prop_thresh2, jps_thresh, vel_1, vel_2, center_res1, center_res2 = 0.10, 0.55, -0.7, 0.02, 0.02, 5, 4
    else:
        joint_start = [0, -100 * math.pi / 180, 30 * math.pi / 180, 85 * math.pi / 180, 100 * math.pi / 180, 0]
        pix_prop_thresh1, pix_prop_thresh2, jps_thresh, vel_1, vel_2, center_res1, center_res2 = 0.15, 0.65, 0.65, 0.02, 0.02, 5, 4

    obj_dist_min, obj_dist_max = 0.35, 1
    cent_weight1, cent_weight2 = 0.01, 0.01
elif robot_type == 2:
    [connected, clientID, joint_handles, hand_handles, cam_Handle] = Grasp_Learn_Func.kuka_sim_setup(gripper_type)
    pix_prop_thresh1, pix_prop_thresh2, jps_thresh, vel_1, vel_2, center_res1, center_res2 = 0.15, 0.30, 2.5, 0.03, -0.03, 5, 4
    if gripper_type == 2:
        joint_start = [0, -20 * math.pi / 180, 0, -55 * math.pi / 180, 70 * math.pi / 180, 90 * math.pi / 180]
    else:
        joint_start = [0, -20 * math.pi / 180, 0, -55 * math.pi / 180, 120 * math.pi / 180, 90 * math.pi / 180]
    obj_dist_min, obj_dist_max = 0.25, 0.7
    cent_weight1, cent_weight2 = 0.01, 0.05
elif robot_type == 3:
    [connected, clientID, joint_handles, hand_handles, cam_Handle] = Grasp_Learn_Func.ABB_sim_setup(gripper_type)
    pix_prop_thresh1, pix_prop_thresh2, jps_thresh, vel_1, vel_2, center_res1, center_res2 = 0.15, 0.30, 2.5, 0.03, -0.03, 5, 4
    if gripper_type == 2:
        joint_start = [0, -20 * math.pi / 180, 0, -55 * math.pi / 180, 70 * math.pi / 180, 90 * math.pi / 180]
    else:
        joint_start = [0, 0, 0, 0, -90 * math.pi / 180, 0]
    obj_dist_min, obj_dist_max = 0.25, 0.7
    cent_weight1, cent_weight2 = 0.01, 0.05


if connected:

    resetNum, grasp_type, EE_turn, bci_class = 0, 0, 0, 0
    step1, sig, robot_moving = False, False, False
    obj_pos_c = [0, 0, 0]
    bci_iter_len = 100000
    trial_timeout = 600

    expInfo = {'participant': 'daniel', 'session': '001'}

    if use_BCI:
        clf = real_time_BCI.train_network(expInfo)
        all_bci_data = real_time_BCI.get_test_data()
        # real_time_BCI.init_receiver()

    with open('vrep_data_record.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # spamwriter.writerow(['ObjectType', 'Object_x', 'Object_y', 'Max_Height', 'Time_of_Completion', 'Success?'])

        total_start = time.clock()
        while resetNum < 10:
            print('Iteration Number: ', resetNum)
            trial_incomplete = True
            success, EE_turn = 0, 0
            this_bci_iter = resetNum * bci_iter_len
            next_bci_iter = this_bci_iter + bci_iter_len

            vrep.simxSetStringSignal(clientID, 'Hand', 'false', vrep.simx_opmode_oneshot)
            vrep.simxSetIntegerSignal(clientID, 'sig', 0, vrep.simx_opmode_oneshot)
            # vrep.simxSetStringSignal(clientID, 'IKCommands', 'LookPos', vrep.simx_opmode_oneshot)  # "Looking" position
            # [success, max_height] = Grasp_Learn_Func.start_pos(joint_start)

            if num_obj == 1:
                [obj_type, OP_x, OP_y] = Grasp_Learn_Func.select_object(obj_dist_min, obj_dist_max)
                [objHandle, init_param] = Grasp_Learn_Func.place_object(OP_x, OP_y, obj_type)
            elif num_obj == 3:
                if set_pos:
                    objHandles = Grasp_Learn_Func.place_all_3_obj_setpos()
                else:
                    objHandles = Grasp_Learn_Func.place_all_3_obj(obj_dist_min, obj_dist_max)
            elif num_obj == 9:
                if set_pos:
                    objHandles = Grasp_Learn_Func.place_all_9_obj_setpos()


            if use_BCI:
                bci_iter, robot_action = Grasp_Learn_Func.search_object_bci(bci_iter, this_bci_iter, next_bci_iter, clf, num_channels=num_channels)
                # robot_action = 1
                new_robot_action = robot_action
            else:
                Grasp_Learn_Func.search_object()
                robot_action = 1
                new_robot_action = 1

            robot_moving = True
            print('NRA: ', robot_action)

            tic = time.clock()
            tic_start_trial = tic
            tic_bci = tic
            while trial_incomplete:

                toc = time.clock()
                if toc - tic_start_trial > trial_timeout:
                    # print('Timeout!')
                    robot_action = 3
                    tic = time.clock()

                errorCode2, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_Handle, 0,
                                                                              vrep.simx_opmode_buffer)
                errorCode2, resolution, image_d = vrep.simxGetVisionSensorDepthBuffer(clientID, cam_Handle,
                                                                                      vrep.simx_opmode_buffer)
                if robot_action > 1 or bci_iter > next_bci_iter - 1:
                    bci_class = 0
                elif use_BCI and toc - tic_bci > bci_update:
                    bci_class, cert = real_time_BCI.get_bci_class(bci_iter, clf)
                    tic_bci = time.clock()
                    bci_iter = bci_iter + 1

                if bci_class == 4:
                    Grasp_Learn_Func.lock_all_joints()
                    robot_moving = False

                # print(robot_action)

                if robot_moving:
                    im = np.array(image, dtype=np.uint8)
                    if len(resolution) > 0:
                        im.resize([resolution[0], resolution[1], 3])
                        if num_obj > 1:
                            obj_list = Grasp_Learn_Func.image_process_2(im, resolution, bci_class)
                            if len(obj_list) > 0:
                                if len(obj_list[0]) > 1:
                                    obj_x, obj_y, obj_pix_prop = obj_list[0][1], obj_list[0][2], obj_list[0][3]
                                    type_obj = obj_list[0][0]
                                    Grasp_Learn_Func.send_obj_sig(type_obj)
                        else:
                            [grasp_type, obj_x, obj_y, obj_pix_prop] = Grasp_Learn_Func.image_process(im, resolution)

                        if robot_action == 1:
                            new_robot_action = Grasp_Learn_Func.approach1(pix_prop_thresh1, vel_1, center_res1, cent_weight1, jps_thresh)
                        elif robot_action == 2:
                            new_robot_action = Grasp_Learn_Func.approach2(pix_prop_thresh2, vel_2, center_res2, cent_weight2)
                        elif robot_action == 3:
                            if learning_rl:
                                new_robot_action = Learn_Final_Grasp.start(clientID, joint_handles, obj_type, OP_x, OP_y, cam_Handle)
                            else:
                                new_robot_action = Grasp_Learn_Func.final_grasp()
                        elif robot_action == 4:
                            new_robot_action = Grasp_Learn_Func.move_back()
                        elif robot_action == 5:
                            new_robot_action = Grasp_Learn_Func.turn_EE()
                        elif robot_action == 6:
                            [success, max_height] = Grasp_Learn_Func.start_pos(joint_start)
                            for i in range(0, len(joint_handles)):
                                vrep.simxSetObjectIntParameter(clientID, joint_handles[i], vrep.sim_jointintparam_ctrl_enabled,
                                                               1, vrep.simx_opmode_oneshot)


                            vrep.simxSetStringSignal(clientID, 'Hand', 'false', vrep.simx_opmode_oneshot)
                            vrep.simxSetIntegerSignal(clientID, 'sig', 0, vrep.simx_opmode_oneshot)
                            time.sleep(5)
                            trial_incomplete = False
                            toc = time.clock()
                            time_to_complete = toc - tic_start_trial
                            resetNum = resetNum + 1
                            # spamwriter.writerow([obj_type, OP_x, OP_y, max_height, time_to_complete, success])

                        if new_robot_action is not robot_action:
                            print('NRA: ', new_robot_action)
                            robot_action = new_robot_action
                            # if robot_action is not 6:
                            #     fc_robot_action = Grasp_Learn_Func.full_center(obj_x, obj_y, 10, resolution)
                            #     if fc_robot_action == 6:
                            #        robot_action = 6
                            Grasp_Learn_Func.lock_all_joints()
                else:
                    if use_BCI:
                        bci_iter, robot_action = Grasp_Learn_Func.search_object_bci(bci_iter, this_bci_iter, next_bci_iter, clf)
                    print('NRA: ', robot_action)
                    robot_moving = True
                    # time.sleep(3)

        vrep.simxFinish(clientID)
if use_BCI:
    real_time_BCI.save_data()
print('Program ended')