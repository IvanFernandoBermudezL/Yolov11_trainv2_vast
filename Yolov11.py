from ultralytics import YOLO

# Load a COCO-pretrained YOLOv11 
model = YOLO("yolo11n.pt")
 
def main():
    # Train 
    model.train(
    data="/Users/ivanbermudez/Downloads/Yolov11_trainv2_vast/DITCH_Yolov11/data.yaml", 
    epochs=120, 
    batch=8, 
    imgsz=640, 
    device="mps",
    save_period=5)

if __name__ == '__main__':
    main()