import os
import cv2
import numpy as np
import re
from datetime import datetime 
import logging

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def create_orb_detector(nfeatures=2000): # Creating an ORB detector object
    return cv2.ORB_create(nfeatures=nfeatures)

def detect_keypoint_compute_descriptor(image, detector): # Computing feature descriptors for the frame

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp = detector.detect(gray, None)
    kp, des = detector.compute(gray, kp)
    return kp, des

def match_descriptors(des1, des2, ratio=0.75): # Matching unique descriptors between two frames

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
        
    matches_knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches_knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def compute_similarity_score(img_1, img_2, orb_detector=None): # Calculating the frame similarity score between two frames

    if orb_detector is None:
        orb_detector = create_orb_detector(nfeatures=2000)

    
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    
    kp1_full, des1_full = detect_keypoint_compute_descriptor(gray_1, orb_detector)
    kp2_full, des2_full = detect_keypoint_compute_descriptor(gray_2, orb_detector)
    
    h, w = gray_1.shape[:2]
    centerA = gray_1[h//4:3*h//4, w//4:3*w//4]
    
    h, w = gray_2.shape[:2]
    centerB = gray_2[h//4:3*h//4, w//4:3*w//4]
    
    kp1_center, des1_center = detect_keypoint_compute_descriptor(centerA, orb_detector)
    kp2_center, des2_center = detect_keypoint_compute_descriptor(centerB, orb_detector)
    
    score_full = 0
    good_matches_full = match_descriptors(des1_full, des2_full, ratio=0.75)
    score_full = len(good_matches_full)
    
    score_center = 0
    good_matches_center = match_descriptors(des1_center, des2_center, ratio=0.75)
    score_center = len(good_matches_center)
    
    score = score_full + 2.0 * score_center
    
    return score

def detect_object_yolo(frame, detections_folder, frame_file, conf=0.5): # Getting the YOLO detection for that frame

    frame_base = os.path.splitext(os.path.basename(frame_file))[0]
    
    detection_file = os.path.join(detections_folder, f"{frame_base}.txt")

    
    best_conf = 0.0
    best_box = None
    best_class_id = 0  
    

    with open(detection_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            confidence = float(parts[5])
                
            if confidence > best_conf:
                    best_conf = confidence
                    best_class_id = class_id
                    
                    img_height, img_width = frame.shape[:2]
                    
                    x = int((x_center - width/2) * img_width)
                    y = int((y_center - height/2) * img_height)
                    w = int(width * img_width)
                    h = int(height * img_height)
                    
                    best_box = (x, y, w, h)

    
    if best_box is None:
        return None
    if best_conf < conf:
        return None
    
    return (*best_box, best_conf, best_class_id)

def template_matching_tracker(prev_frame, curr_frame, prev_bbox, search_margin=20): # Calulating template matching to track the polyp from one frame to another

    x, y, w, h = map(int, prev_bbox)
    
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w >= prev_frame.shape[1] or y + h >= prev_frame.shape[0]:
        print("Invalid bounding box for the shape of the frame")
        return None
    
    template = prev_frame[y:y+h, x:x+w]
    
    x_min = max(0, x - search_margin)
    y_min = max(0, y - search_margin)
    x_max = min(curr_frame.shape[1], x + w + search_margin)
    y_max = min(curr_frame.shape[0], y + h + search_margin)
    
    if x_max <= x_min + w or y_max <= y_min + h:
        print("Search region too small")
        return None
    
    search_region = curr_frame[y_min:y_max, x_min:x_max]
    
    if template.shape[0] == 0 or template.shape[1] == 0 or search_region.shape[0] <= template.shape[0] or search_region.shape[1] <= template.shape[1]:
        print("Invalid template or search region")
        return None
    
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
        
    if len(search_region.shape) == 3:
        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    else:
        search_gray = search_region
    
    result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    
    if max_val < 0.5:
        print("Template matching confidence score was too low")
        return None
    
    
    new_x = x_min + max_loc[0]
    new_y = y_min + max_loc[1]
    
    
    return (new_x, new_y, w, h, float(max_val))
    

    
def adaptive_search_margin(frame_distance): # Calculating the search margin for the tracking bounding box depending on the frame distance between the YOLO detected frame and current frame

    
    if frame_distance == 1:
        return 20
    elif frame_distance == 2:
        return 30
    else:
        return 40

def compare_bounding_box_contents(img1, img2, bbox1, bbox2, min_size=10): # Calculating the pixel similairty score between the YOLO detected frame and the tracked frame
    
    x1, y1, w1, h1 = map(int, bbox1)
    x2, y2, w2, h2 = map(int, bbox2)
    
    
    if w1 < min_size or h1 < min_size or w2 < min_size or h2 < min_size:
        return 0.0
    
    
    h_img1, w_img1 = img1.shape[:2]
    h_img2, w_img2 = img2.shape[:2]
    
    if (x1 < 0 or y1 < 0 or x1 + w1 > w_img1 or y1 + h1 > h_img1 or
        x2 < 0 or y2 < 0 or x2 + w2 > w_img2 or y2 + h2 > h_img2):
        return 0.0
    
    
    content1 = img1[y1:y1+h1, x1:x1+w1]
    content2 = img2[y2:y2+h2, x2:x2+w2]
    
    
    if content1.shape[:2] != content2.shape[:2]:
        content2 = cv2.resize(content2, (content1.shape[1], content1.shape[0]))
    
        
    if len(content1.shape) < 3:
        content1 = cv2.cvtColor(content1, cv2.COLOR_GRAY2BGR)
    if len(content2.shape) < 3:
        content2 = cv2.cvtColor(content2, cv2.COLOR_GRAY2BGR)
        
    
    channels_err = []
    for channel in range(3):  
        err = np.sum((content1[:,:,channel].astype("float") - content2[:,:,channel].astype("float")) ** 2)
        err /= float(content1.shape[0] * content1.shape[1])
        channels_err.append(err)
    
    
    avg_err = sum(channels_err) / 3
    
    
    similarity = max(0, 1 - (avg_err / 10000))

        
    return similarity
    

def extract_patient_id(filename): # Getting the patient id
    
    match = re.search(r'((?:PF|HF)\d+)', filename)
    if match:
        return match.group(1)
    return None

def get_frames_by_patient(frames_folder): # Getting the frames for each patient

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    frame_files = sorted([
        f for f in os.listdir(frames_folder)
        if f.lower().endswith(valid_exts)
    ])
    
    patient_frames = {}
    for frame_file in frame_files:
        patient_id = extract_patient_id(frame_file)
        if patient_id not in patient_frames:
                patient_frames[patient_id] = []
        patient_frames[patient_id].append(frame_file)
    
    return patient_frames

def bbox_to_yolo_format(bbox, img_shape, class_id=0): # Converting the output tracking detections to yolo format 

    x, y, w, h = bbox[:4]
    confidence = bbox[4] if len(bbox) > 4 else 1.0
    
    img_height, img_width = img_shape[:2]
    
    
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    
    width_norm = w / img_width
    height_norm = h / img_height
    
    
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width_norm = max(0, min(1, width_norm))
    height_norm = max(0, min(1, height_norm))
    
    return f"{class_id} {x_center} {y_center} {width_norm} {height_norm} {confidence}"

def process_patient(
    frames_folder,
    detections_folder,
    predictions_folder,
    patient_id,
    patient_frame_files,
    confidence_thresh=0.5,
    max_tracking_frames=5,
    similarity_threshold=25,
    pixel_similarity_threshold=0.5
):                                         # Pipeline function that calls all the functions performing the temporal tracking for each individual patient


        
        os.makedirs(predictions_folder, exist_ok=True)

        print("Processing Patient")

        
        frames = []
        frame_paths = []
        
        for frame_file in patient_frame_files:
            path = os.path.join(frames_folder, frame_file)
            frame_paths.append(path)
            img = cv2.imread(path)
            if img is None:
                print("Could not read {path}")
                frames.append(None)
            else:
                frames.append(img)

        
        detections = [None] * len(frames)  
        yolo_only_detections = [None] * len(frames)   
        detection_sources = {}  
        
        
        pixel_check_results = {}  
        pixel_similarity_scores = {}  
        
        
        frame_similarity_scores = {}  
        
        
        frame_metrics = {}  
        for i in range(len(frames)):
            if frames[i] is not None:
                frame_file = patient_frame_files[i]
                frame_base = os.path.splitext(os.path.basename(frame_file))[0]
                frame_metrics[i] = {
                    'frame_file': frame_file,
                    'frame_base': frame_base,
                    'has_detection': False,
                    'detection_source': None,
                    'pixel_similarity': None,
                    'frame_similarity': None,
                    'reference_frame': None
                }
        
        
        already_tracked_frames = set()
        
        
        print(f"First Pass: Reading YOLO detections for Patient {patient_id}...")
        yolo_detection_frames = []  

        for i, frame in enumerate(frames):
            if frame is None:
                continue
            
            
            frame_file = patient_frame_files[i]
            detect = detect_object_yolo(frame, detections_folder, frame_file, conf=confidence_thresh)
            
            if detect is not None:
                (x, y, w, h, c, class_id) = detect
                detections[i] = (x, y, w, h, c, class_id)
                yolo_only_detections[i] = (x, y, w, h, c, class_id)
                detection_sources[i] = "yolo"
                yolo_detection_frames.append(i)  
                pixel_check_results[i] = None  
                
                
                frame_metrics[i]['has_detection'] = True
                frame_metrics[i]['detection_source'] = 'yolo'
                

        
        orb_detector = create_orb_detector(nfeatures=2000)

        
        
        
        for seed_idx in yolo_detection_frames:
            if frames[seed_idx] is None:
                continue
                
            
            (x_seed, y_seed, w_seed, h_seed, conf_seed, class_id) = detections[seed_idx]
            init_bbox = (x_seed, y_seed, w_seed, h_seed)
            
            
            end_frame = min(seed_idx + max_tracking_frames, len(frames) - 1)
            
            
            
            for target_idx in range(seed_idx + 1, end_frame + 1):
                
                if frames[target_idx] is None:
                    continue
                    
                
                
                similarity = compute_similarity_score(frames[seed_idx], frames[target_idx], orb_detector)
                
                
                frame_similarity_scores[(seed_idx, target_idx)] = similarity
                
                
                if frame_metrics[target_idx]['frame_similarity'] is None or similarity > frame_metrics[target_idx]['frame_similarity']:
                    frame_metrics[target_idx]['frame_similarity'] = similarity
                    frame_metrics[target_idx]['reference_frame'] = seed_idx
                
                
                
                if detections[target_idx] is not None:
                    continue
                    
                if target_idx in already_tracked_frames:
                    continue
                
                if similarity < similarity_threshold:
                    continue
                
                
                search_margin = adaptive_search_margin(target_idx - seed_idx)
                
                template_result = template_matching_tracker(
                    frames[seed_idx],
                    frames[target_idx],
                    init_bbox,
                    search_margin=search_margin
                )
                
                if template_result is not None:
                    x_t, y_t, w_t, h_t, track_conf = template_result
                    
                    
                    
                    
                    pixel_similarity = compare_bounding_box_contents(
                        frames[seed_idx], 
                        frames[target_idx],
                        init_bbox,
                        (x_t, y_t, w_t, h_t)
                    )
                    

                    pixel_check_passed = pixel_similarity >= pixel_similarity_threshold
                    pixel_check_results[target_idx] = pixel_check_passed
                    pixel_similarity_scores[target_idx] = pixel_similarity
                    
                    frame_metrics[target_idx]['pixel_similarity'] = pixel_similarity

                    if pixel_check_passed:
                        
                        detections[target_idx] = (x_t, y_t, w_t, h_t, track_conf, class_id)  
                        detection_sources[target_idx] = "forward_tracking"  
                        already_tracked_frames.add(target_idx)
                        
                        frame_metrics[target_idx]['has_detection'] = True
                        frame_metrics[target_idx]['detection_source'] = 'forward_tracking'
                        
                    else:
                        print(f"Frame {target_idx}: Pixel check FAILED (similarity: {pixel_similarity:.3f}) => Detection filtered out")
                else:
                    print(f"Template matching failed from frame {seed_idx} to {target_idx}")
                    
        print(f"Saving detection results for patient {patient_id}...")
        for i, frame in enumerate(frames):
            if frame is None:
                continue
            
            frame_file = patient_frame_files[i]
            frame_base = os.path.splitext(os.path.basename(frame_file))[0]
            output_path = os.path.join(predictions_folder, f"{frame_base}.txt")
            
            
            if detections[i] is not None:
                x, y, w, h, conf, class_id = detections[i]
                
                
                yolo_line = bbox_to_yolo_format((x, y, w, h, conf), frame.shape, class_id=class_id)
                
                
                with open(output_path, 'w') as f:
                    f.write(yolo_line + '\n')
                    
            else:
                
                open(output_path, 'w').close()
        
        
    
        print(f"Completed processing patient {patient_id}")
        
        return {
            'patient_id': patient_id,
            'detection_count': sum(1 for d in detections if d is not None),
            'yolo_detection_count': sum(1 for d in yolo_only_detections if d is not None),
            'tracking_detection_count': sum(1 for i, d in enumerate(detections) 
                                          if d is not None and i not in yolo_detection_frames),
        }


def process_all_patients(
    frames_folder,
    detections_folder,
    output_folder="predictions",
    confidence_thresh=0.5,
    max_tracking_frames=5,
    similarity_threshold=25,
    pixel_similarity_threshold=0.5
):                                   # Main function that calls process_patient for all patients and generates the total outputs
    
    
    processing_start_time = datetime.now()

    
    os.makedirs(output_folder, exist_ok=True)

    print("Using pre-computed YOLO detections from files...")
    
    
    print("Getting all patients from frames folder...")
    patient_frames_dict = get_frames_by_patient(frames_folder)
    
    
    patient_count = len(patient_frames_dict)
    
    
    processing_stats = []
    patient_processing_times = {}
    
    for i, (patient_id, frame_files) in enumerate(patient_frames_dict.items()):
        print("Processing patient")
        
        
                
        patient_start_time = datetime.now()
        
            
        stats = process_patient(
            frames_folder=frames_folder,
            detections_folder=detections_folder,
            predictions_folder=output_folder,
            patient_id=patient_id,
            patient_frame_files=frame_files,
            confidence_thresh=confidence_thresh,
            max_tracking_frames=max_tracking_frames,
            similarity_threshold=similarity_threshold,
            pixel_similarity_threshold=pixel_similarity_threshold,
        )
        
        patient_end_time = datetime.now()
        processing_duration = (patient_end_time - patient_start_time).total_seconds()
        patient_processing_times[patient_id] = processing_duration
        
        print(f"Patient {patient_id} processed in {processing_duration:.2f} seconds")
        
        
        if stats:
            stats['processing_time'] = processing_duration
            processing_stats.append(stats)
                                
    
    processing_end_time = datetime.now()
    total_processing_time = (processing_end_time - processing_start_time).total_seconds()
    
    print(f"\nTotal processing time: {total_processing_time:.2f} seconds")
        
    
    total_frames = sum(stats['frame_count'] for stats in processing_stats)
    total_yolo_detections = sum(stats['yolo_detection_count'] for stats in processing_stats)
    total_tracking_detections = sum(stats['tracking_detection_count'] for stats in processing_stats)
    total_detections = total_yolo_detections + total_tracking_detections
        
    
    overall_stats = {
        'total_patients': patient_count,
        'total_frames': total_frames,
        'total_yolo_detections': total_yolo_detections,
        'total_tracking_detections': total_tracking_detections,
        'total_detections': total_detections,
        'processing_time': total_processing_time
    }
    
    
    
    return overall_stats

# Specify the ground truth paths and the thresholds 
# confidence_thresh: Minimum confidence threshold for the YOLO detection to be considered valid
# max_tracking_frames: The maximum number of consective frames the YOLO detection can be tracked to
# similarity_threshold: The threshold for determing whether two frames are similar enough to proceed with tracking
# pixel_similarity_threshold: Threshold to check if the tracked polyp is similar enough to the seed YOLO detected polyp (where the tracking originated from)

def main():
    frames_folder = "Path to the test frames folder"  
    detections_folder = "Path to the yolo detection prediction text files"  
    output_folder = "Path to the detection output text files"  

    
    confidence_thresh = "Confidence Threshold"    
    max_tracking_frames = "Number of tracking frames"  
    similarity_threshold = "Frame Similarity Threshold"   
    pixel_similarity_threshold = "Pixel Similarity Threshold"

    
    stats = process_all_patients(
        frames_folder=frames_folder,
        detections_folder=detections_folder,
        output_folder=output_folder,
        confidence_thresh=confidence_thresh,
        max_tracking_frames=max_tracking_frames,
        similarity_threshold=similarity_threshold,
        pixel_similarity_threshold=pixel_similarity_threshold,
    )

    
    print("Processing complete")
    print(f"Results saved to: {output_folder}")
    print("\nOverall Summary:")
    print(f"Total frames: {stats['total_frames']}")
    print(f"YOLO detections: {stats['total_yolo_detections']}")
    print(f"Tracking detections: {stats['total_tracking_detections']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Processing time: {stats['processing_time']:.2f} seconds")


if __name__ == "__main__":
    main()
