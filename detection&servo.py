import cv2
from picamera2 import Picamera2
from tflite_support.task import core, processor, vision
from gpiozero import AngularServo
from time import sleep
import utils

# Initialize GPIO and servo motor
servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)

# Load the TFLite model and set up the object detector
model_path = 'detect_metadata.tflite'
num_threads = 4
base_options = core.BaseOptions(file_name=model_path, use_coral=False, num_threads=num_threads)
detection_options = processor.DetectionOptions(max_results=2, score_threshold=0.8)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

def move_servo(angle):
    servo.angle = angle
    sleep(1)

def main():
    try:
        # Use picamera2 to capture frames
        with Picamera2() as picam:
            picam.preview_configuration.main.size = (1280, 720) 
            picam.preview_configuration.main.format = 'RGB888'  # Change to a supported format
            picam.preview_configuration.align()
            picam.start()

            while True:
                # Capture image from camera
                im = picam.capture_array()

                # Detect objects
                im_tensor = vision.TensorImage.create_from_array(im)
                detections = detector.detect(im_tensor)

                for detection in detections.detections:
                    confidence = detection.categories[0].score
                    bbox = detection.bounding_box
                    bbox_center_x = bbox.origin_x + bbox.width // 2

                    # If confidence is high, move the servo based on deviation
                    if confidence > 0.8:
                        frame_center_x = im.shape[1] // 2
                        deviation = frame_center_x - bbox_center_x
                        print(deviation)

                        if deviation > -0.5:
                            move_servo(45)  # Rotate clockwise
                        elif deviation < 0.5:
                            move_servo(-45)  # Rotate anti-clockwise
                        else:
                            sleep(2)  # Centers coincide

                    # Draw bounding box and text on the image
                    bbox_coords = (int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height))
                    cv2.rectangle(im, (bbox_coords[0], bbox_coords[1]), (bbox_coords[0] + bbox_coords[2], bbox_coords[1] + bbox_coords[3]), (0, 255, 0), 2)
                    text = f"Accuracy: {confidence * 100:.2f}%"
                    cv2.putText(im, text, (bbox_coords[0], bbox_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the camera feed with bounding boxes and text
                cv2.imshow('Camera Feed', im)
                if cv2.waitKey(1) == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        # Make sure to stop the servo before exiting
        servo.angle = None

if __name__ == "__main__":
    main()
