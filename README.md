# Unity-ml-agents-self-driving-car
Construction of 3D digital twin environment for vehicle using Unity ML-Agents and implementation of self-driving technology using deep reinforcement learning(DQN)

## Version
* **Unity:** 2021.3
* **ML-Agents:** 2.2.1 (release 19)
* **Python:** 3.7
* **ML-Agents Python package:** 0.28.0
* **Torch:** 1.7.1+cu110

## Installation and How to playing(In Window)
1. Anaconda3: Download and installation instructions here:
https://www.anaconda.com/products/individual

    <img src="https://github.com/sh02092/unity-ml-agents-self-driving-car/blob/6b52e03eb65ce652ae60d398122be055b4f9c487/README_image/Anaconda3.png"></img>

2. Nvidia driver: Download by setting it to your GPU specifications.
https://www.nvidia.com/Download/index.aspx?lang=kr

    <img src="https://github.com/sh02092/unity-ml-agents-self-driving-car/blob/6b52e03eb65ce652ae60d398122be055b4f9c487/README_image/Nvidia%20driver.png"></img>

3. cuDNN, CUDA, Python version: Check the version here:
https://www.tensorflow.org/install/source_windows

    - CUDA: Download here:

        https://developer.nvidia.com/cuda-toolkit-archive

    - cuDNN: Download here:

        https://developer.nvidia.com/rdp/cudnn-archive
        
        After downloading, extract the file and move it to
        >C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2

4. Create a Virtual Environment in Anaconda3:
    > conda create -n your_virtual_env_name
    
    > conda activate your_virtual_env_name

    > pip install tensorflow-gpu

    > python -m pip install -q mlagents==0.28.0

    > pip3 install torch~=1.7.1 -f https://download.pytorch.org/whl/torch_stable.html 

5. Open Visual Studio Code(Set up virtual environment)
    > Ctrl + Shift + P (Python select interpreter)

6. Open python scripts(Car_gym.py, DQN_220523.py)

7. Run DQN_220523.py script

## Preview
<img width="80%" src="https://github.com/sh02092/unity-ml-agents-self-driving-car/blob/573e31974e8a4fd5c41a95e7e6b37abdc8bc7dc8/README_image/Self-driving-car.gif"/></img>

## Reference
* [Unity-ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/release_19_docs/docs/Readme.md)
* [2021_RLKR_Drone_Delivery_Challenge_with_Unity](https://github.com/reinforcement-learning-kr/2021_RLKR_Drone_Delivery_Challenge_with_Unity/tree/master/baseline/code)

