# BCIRobotControl

Hello, this repository houses a framework for EEG Motor Imagery control of an assistive robotic manipulator. 
The framework uses simple computer vision techniques along with EEG processing to control a simulated robot in 3-dimensional space.

To run the MIndGrasp Trainer, download V-REP (now CoppeliaSim) and run the scene: BCIRobotControl/vrep_files/Scenes/HamlynArm_Grasp_train.ttt

To run the MIndGrasp Tester, similarly run the scene:
BCIRobotControl/vrep_files/Scenes/HamlynArm_Grasp.ttt

To connect to V-REP and start controlling the robot, use robot_grasp.py, ensuring that the variable "mode" is set to either 'train' or 'test', depending on which scene you are running.

For any further questions, please contact Daniel Freer via df1215@ic.ac.uk or freerdan00@gmail.com.

The simulation environment we have used is V-REP, so for instructions on how to install this and use the remote API, check out their website: http://www.coppeliarobotics.com/

We have also used LabStreamingLayer to facilitate the synchronisation of EEG data between different programs. For more information about this, their open source code is also available: https://github.com/sccn/labstreaminglayer 

Any use of this code or the simulator should cite the following paper:
Daniel Freer, Guang-Zhong Yang. "MIndGrasp: A New Training and Testing Framework for Motor Imagery Based 3-Dimensional Assistive Robotic Control". https://arxiv.org/pdf/2003.00369.pdf
