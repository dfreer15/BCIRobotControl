# BCIRobotControl

Hello, this repository houses a framework for EEG Motor Imagery control of an assistive robotic manipulator. 
The framework uses simple computer vision techniques along with EEG processing to control a simulated robot in 3-dimensional space.

To run the MIndGrasp Trainer, download V-REP (now CoppeliaSim) and run the scene in:

To run the MIndGrasp Tester, similarly run the scene in:

The simulation environment we have used is V-REP, so for instructions on how to install this and use the remote API, check out their website: http://www.coppeliarobotics.com/

We have also used LabStreamingLayer to facilitate the synchronisation of EEG data between different programs. For more information about this, their open source code is also available: https://github.com/sccn/labstreaminglayer 

Any use of this code or the simulator should cite the following paper:

Daniel Freer, Guang-Zhong Yang. "MIndGrasp: A New Training and Testing Framework for Motor Imagery Based 3-Dimensional Assistive Robotic Control". https://arxiv.org/pdf/2003.00369.pdf
