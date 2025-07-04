import torch


from ultralytics import YOLO


model = YOLO('yolo12x.pt') # Specifying the YOLO model


# Training the YOLO model and specifying all the hyperparameters

train_results = model.train(
    data='Path to the YOLO yaml file',
    imgsz=1024,
    epochs=100,
    batch=2,
    project='Path for the outputs',
    name='Name of the output folder',
    verbose=True,
    workers=4,
    save_period=50,
    single_cls=True,
    close_mosaic=1,
    lr0=0.00005,
    lrf=0.01,
    warmup_epochs=10,
    box=7.5,
    cls=8.0,
    dfl=5.0,
    label_smoothing=0.05,
    dropout=0.2,         
    plots=True,
    cos_lr=True,
    patience=40,
    weight_decay=0.0005,
    optimizer='RAdam',
    max_det = 1,
    task = "detect",
    iou=0.5,
    device="cuda",
    half=True,
    amp=True
)   

# Performing YOLO detections on the test set using the trained model and outputting the detection files

test_results = model.val(
    data='Path to the YOLO yaml file',
    split = 'test',
    imgsz=1024,
    iou=0.5,
    device="cuda",
    half=True,
    amp=True,
    save_txt=True,
    save_conf=True,
    save_json=True
)

                            
