# Live MoCap

![Ganyu-ji](images/ganyu_ji.gif)

## Requirements

* python>=3.8
    * mediapipe
    * pytorch (cpu version is ok)
* blender >= 3.0 (for reading assets and binding animation)

## How to use

1.  Prepare your character model
    
    Currently this script uses **Blender** to load model skeleton and bind animation. Your model should be saved as .blend file. 
    
    You may edit your model to assure that

    * Model must be in rest pose (clear all bone rotation/translation/scale in pose mode). And the rest pose should be close to T pose. 
    
    * Clear previous bone animation data and constraints.
    
    * Name related bones as below (in lower case). 
    (Picture here)

    * Save the model as `.blend` file somewhere.

2.  Run the script `mocap.py`.

    ```
    python mocap.py --blend your_character_model.blend --video your_video.mp4 [other options] 
    ```

# Future work

* Now working on face capture.