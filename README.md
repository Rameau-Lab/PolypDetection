### This project is for the detection of vocal fold polyps from the frames generated through a video stroboscopy through YOLO12 and temporal tracking

train.py => File for training the YOLO12 model and getting the detections for the test set

main.py => Taking the YOLO test detections and applying temporal tracking on them to extrapolate the ground YOLO detections to the next frame

**Inputs required**

 1. Patient images from the video stroboscopy

    - Filenames in the format of HF|PF_patientid_framenumber.jpg, for example, HF002_tensor(337).jpg or PF002_tensor(337).jpg

      - PF or HF => depending on whether the frame contains a polyp or not
        
    

``` bash
├── images
│   ├── train
│   └── test
│   └── validation
```

2. Polyp annotated label files in the yolo format
   - Filenames in the format of HF|PF_patientid_framenumber.txt, for example, HF002_tensor(337).txt or PF002_tensor(337).txt

      - PF or HF => specifiying whether the frames contains a polyp or not
      - The HF files should be blank as the corresponding frames do not contain a polyp

``` bash
├── labels
│   ├── train
│   └── test
│   └── validation
```

3. Yaml configuration file for YOLO

**Run steps**

1. Run train.py, change the hyperparmamters according to your data
2. Save the YOLO generated test detection labels in the same format as ground truth labels
3. Run main.py, specifying the ground truth paths, the YOLO prediction outputs and the output path
   - Tweak the thresholds based on your requirements 
   
   
    





