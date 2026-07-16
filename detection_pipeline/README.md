# CV2X based crack-detection pipeline

Here we show the procedures to run the pipeline on our platform, you are welcomed to test it on your own, but please remember to change the directories accordingly. 
1. Direct to the detection folder, as all the codes for the pipeline are there
   ```bash
      cd detection_pipline
   ```
2. run `docker_run_lincoln.sh` to start the docker container. 
   ```bash
   ./docker_run_lincoln.sh
   ```
   This will start a Docker container with the necessary environment for running the crack-detection application. 

3. Open another terminal, run the following command if you need to run more to check the vehicle status. 
   ```bash
   sh docker_bash.sh
   ```
4. The four nodes are under [node](/detection_pipeline/nodes), for detail description please refer to our [paper](https://www.computer.org/csdl/proceedings-article/most/2026/618200a193/2gNOZvXt4Iw)
5. To use the pipeline, please use launch file [all_launch.py](/detection_pipeline/all_launch.py) using the following command on the vehicle side:
   ```bash
   ros2 launch all_launch.py
   ```
   Please note that you need to change the directory of the nodes folder before using this command. 
6. Please note that there is also RSU side to run to pass the crack GPS location, to run [RSU_receive.py](/detection_pipeline/RSU_code/RSU_receive.py), use the following command on RSU end,
   ```bash
   python3 RSU_receive.py
   ```
7. If you need to calibrate the camera, please use RTK GPS on the vehicle to locate a target spot, put checkerboard on it, and run the pipeline using steps 5-6 to pass the checkerboard to get data logs.
8. Once you have the dataset from 7, select one image frame with the best checkerboard visibility from the Image folder in the data folder, and run the [offline_tunning.py](/detection_pipeline/offline_tunning.py)
   ```bash
   python3 offline_tunning.py --folder "folder_name" --image "frame_name.png" --model "model_path"
   ```
9. Once calibration is done, please remember to change the calibrated parameters in [tunable_params.json](/detection_pipeline/utilities/config/tunable_params.json)