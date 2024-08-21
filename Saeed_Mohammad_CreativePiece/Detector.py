import cv2, time, os, tensorflow as tf
import numpy as np

# TensorFlows 'keras' model
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
    def __init__(self):
        # Initialize as an empty dictionary to store object counts
        self.object_count = {}  

    # Reads class labels from  file
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

            # Generates colours for each class
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    # Downloads the pre-trained model     
    def downloadModel(self, modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        
        self.cacheDir = "./pretrained_models"

        os.makedirs(self.cacheDir, exist_ok=True)

        # Downloads and extracts the model checkpoints
        get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    # Loads the pre-trained model
    def loadModel(self):
        print("loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))

        print("Model " + self.modelName + " loaded sucessfully...")

    # Creates bounding boxes around the objects
    def creatBoundingBox(self, image, threshold=0.5):
    # Converts the image to RGB format
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        # Performs object detection
        detections = self.model(inputTensor)

        # Extracts the bounding boxes, class indexes, and scores
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        # Applys non-max suppression to filter out overlapping bounding boxes
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)
        
        # Creates a copy of the original image
        bboxImage = image.copy()  

        # Initialize dictionary to store coordinates
        self.object_count = {}

        # Draw bounding boxes
        if len(bboxs) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]
                classLabelText = self.classesList[classIndex].upper()

                classColor = self.colorList[classIndex]

                # This ensures that the object_count is treated as a dictionary
                if isinstance(self.object_count, dict):
                    # Increment count for this class
                    self.object_count[classLabelText] = self.object_count.get(classLabelText, 0) + 1

                # Extracts the coordinates of the bounding box
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = int(xmin * imW), int(xmax * imW), int(ymin * imH), int(ymax * imH)
                
                # Draws a bounding box rectangle
                cv2.rectangle(bboxImage, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)

                # Displays the class label and confidence score
                displayText = '{}:{}%'.format(classLabelText, classConfidence)
                cv2.putText(bboxImage, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, thickness=2)

                # Draws lines to indicate bounding box edges
                lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))
                cv2.line(bboxImage, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                cv2.line(bboxImage, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)
                cv2.line(bboxImage, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                cv2.line(bboxImage, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                cv2.line(bboxImage, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)
                cv2.line(bboxImage, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                cv2.line(bboxImage, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)
                cv2.line(bboxImage, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)

                 # Stores coordinates in the object_count dictionary
                self.object_count[classLabelText + "_coordinates"] = (xmin, ymin, xmax, ymax)    
                
                # Calculates IoU with bbox1 and print the result
                bbox1 = [0, 0, 100, 100]  # Modify this according to your use case
                iou = self.calculate_iou(bbox, bbox1)
                print(f"IoU with bbox1: {iou:.4f}")

        return bboxImage
    
    # This helps to calculate the Intersecrion over Union value
    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate IoU (Intersection over Union) between two bounding boxes.

        bbox1 and bbox2 should be in the format [xmin, ymin, xmax, ymax].
        """
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])

        # Computes the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Computes the area of both the prediction and ground-truth rectangles
        boxAArea = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        boxBArea = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # Returns the intersection over union value
        return iou

    # This method predicts objects in images
    def predictImage(self, imagePath, threshold=0.5):
        image = cv2.imread(imagePath)
        original_image = image.copy()

        bboxImage = self.creatBoundingBox(original_image, threshold)

        # Display objects counts on image
        y_offset = 40
        for classLabelText, count in self.object_count.items():
            if "_coordinates" in classLabelText:
                # Extract coordinates
                xmin, ymin, xmax, ymax = self.object_count.get(classLabelText, (0, 0, 0, 0))
                # Draw bounding box rectangle
                cv2.rectangle(bboxImage, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
                # Display coordinates
                cv2.putText(bboxImage, f"({xmin}, {ymin})", (xmin, ymin - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), thickness=1)
            else:
                text = "{}: {}".format(classLabelText, count)
                cv2.putText(bboxImage, text, (20, y_offset), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                y_offset += 30

        # Display the result
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)  # Create window with resizable option
        cv2.imshow("Result", bboxImage)
        
        # Wait for a key press or window close event
        key = cv2.waitKey(0)
        if key == 27:  # 27 is the ASCII value of the escape key
            cv2.destroyAllWindows()  # Close all OpenCV windows

    # This method predicts objects in videos
    def predictVideo(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("Error opening file")
            return

        while True:
            (success, image) = cap.read()

            if not success:
                break
            original_image = image.copy()

            # Detect objects and count them
            bboxImage = self.creatBoundingBox(image, threshold)

            # Display object counts on frame
            y_offset = 50  # Initial y offset for displaying text
            for classLabelText, count in self.object_count.items():
                if "_coordinates" in classLabelText:
                    # Extract coordinates
                    xmin, ymin, xmax, ymax = self.object_count.get(classLabelText, (0, 0, 0, 0))
                    # Draw bounding box rectangle
                    cv2.rectangle(bboxImage, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
                    # Display coordinates
                    cv2.putText(bboxImage, f"({xmin}, {ymin})", (xmin, ymin - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), thickness=1)
                else:
                    text = "{}: {}".format(classLabelText, count)
                    cv2.putText(bboxImage, text, (20, y_offset), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    # Increase y offset for the next line of text
                    y_offset += 30  

            # Display the result and creates a window with a resize option
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Result", bboxImage)

            # Wait for a key press or window close event
            key = cv2.waitKey(1)

            # 27 is the ASCII value of the escape key
            if key == 27:  
                break

        cap.release()
        cv2.destroyAllWindows()