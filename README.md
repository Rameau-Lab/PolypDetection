# Detection and Tracking of Vocal Fold Polyps

This project is for the detection of vocal fold polyps with YOLO12 and temporal tracking.

### What is Temporal Tracking?

Temporal tracking refers to the process of following objects across consecutive video frames by leveraging motion and spatial continuity. 

In this project, temporal tracking is used to extrapolate the YOLO detected polyps to any consecutive undetected polyp frames.

### Files

train.py => Training the YOLO12 model and getting the detections for the test set.

main.py => Applying temporal tracking to the generated detections, to extrapolate the seed YOLO detections to the consecutive frames.

### Inputs required

 1. Patient images from the video stroboscopy

    - Filenames in the format of HF|PF_patientid_framenumber.jpg, for example, HF002_tensor(337).jpg or PF002_tensor(337).jpg.

      - PF or HF => depending on whether the frame contains a polyp or not.
        
    

``` bash
├── images
│   ├── train
│   └── test
│   └── validation
```

2. Polyp annotated label files in yolo format
   
   - Filenames in the format of HF|PF_patientid_framenumber.txt, for example, HF002_tensor(337).txt or PF002_tensor(337).txt.

      - PF or HF => epending on whether the frame contains a polyp or not.
        
      - The HF files should be blank as the corresponding frames do not contain a polyp.

``` bash
├── labels
│   ├── train
│   └── test
│   └── validation
```

3. Yaml configuration file for YOLO.

### Run steps

1. Run train.py, change the hyperparameters according to your data.
2. Save the YOLO generated test detection labels.
3. Run main.py, specify the paths.
   - Tweak the thresholds based on your requirements.
   
   
    





