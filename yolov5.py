from ultralytics import YOLO

# Load a COCO-pretrained YOLOv5n model
model = YOLO('yolov5nu.pt')

# Display model information (optional)

# Run inference with the YOLOv5n model on the 'bus.jpg' image

results = model.predict('IMG_2216.JPG', conf=0.5, classes=[0, 15])


for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screenç
    print(result)
    result.save(filename='result.jpg')  # save to disk