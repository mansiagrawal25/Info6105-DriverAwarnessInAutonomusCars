import cv2
from keras.models import load_model
import numpy as np
from statistics import mode
from PIL import Image
from scipy.misc import imresize
import os
import pathlib as p

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def draw_text(coordinates, image_array, text,color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,color, thickness, cv2.LINE_AA)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def get_face_coordinates(face_coordinates):
    x, y, width, height = face_coordinates

    return (x,y,width,height)
    
def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
def predict(image,model):
    img = Image.fromarray(image, 'RGB').convert('L')
    img = imresize(img, (24, 24)).astype('float32')
    img /= 255
    img = img.reshape(1,24,24,1)
    prediction = model.predict(img)
    if prediction < 0.1:
        prediction = 'closed'
    elif prediction > 0.9:
        prediction = 'open'
    else:
        prediction = 'idk'
    return prediction


# parameters for loading data and images
#enter the file names and directories accordingly in the following path variables
path = os.path.dirname(__file__)
detection_model_path = path+'/openCV_helper/' + 'haarcascade_frontalface_default.xml'
emotion_model_path = path+'/emotion_classifier/' + 'trained_models.16.hdf5'
open_eye_model_path = path+'/openCV_helper/' + 'haarcascade_eye_tree_eyeglasses.xml'
left_eye_model_path = path+'/openCV_helper/' + 'haarcascade_lefteye_2splits.xml'
right_eye_model_path = path+'/openCV_helper/' + 'haarcascade_righteye_2splits.xml'
eye_detection_model_path = path+'/eye_classifier_trained_model/' + 'eye_classifier_17.hdf5'

emotion_labels = {0: "angry", 1: "closed", 2: "disgusted", 3: "fearful", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}

#importing models
face_detection = load_detection_model(detection_model_path)
open_eyes_detector = load_detection_model(open_eye_model_path)
left_eye_detector = load_detection_model(left_eye_model_path)
right_eye_detector = load_detection_model(right_eye_model_path)

emotion_classifier = load_model(emotion_model_path, compile=False)
eye_classifier = load_model(eye_detection_model_path,compile=False)

print(emotion_classifier.input_shape)
emotion_target_size = emotion_classifier.input_shape[1:3]
eye_target_size = eye_classifier.input_shape[1:3]
print(emotion_target_size)
print(eye_target_size)
emotion_window = []
emotion_offsets = (10, 40)
frame_window = 1
angry,closed,happy,neutral=0,0,0,0
# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        gray_eye = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except:
            continue
        x,y,width,height = get_face_coordinates(face_coordinates)

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = int(np.argmax(emotion_prediction))


        #################################
        #   Testing
        # if emotion_label_arg == 0:
        #     angry = angry+1
        # elif emotion_label_arg== 1:
        #     closed = closed+ 1
        # elif emotion_label_arg == 4:
        #     happy = happy + 1
        # elif emotion_label_arg == 5:
        #     neutral = neutral + 1
        #################################

        emotion_text = emotion_labels[emotion_label_arg]
        #print(angry,closed,happy,neutral)
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        # define the color for text
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'closed':
            color = emotion_probability * np.asarray((255,0,255))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        ######################################################

        open_eyes_glasses = open_eyes_detector.detectMultiScale(
            gray_eye,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # if open_eyes_glasses detect eyes then they are open
        if len(open_eyes_glasses) == 2:
            print('Eyes open')
        else:
            # otherwise try detecting eyes using left and right_eye_detector
            # which can detect open and closed eyes

            # separate the face into left and right sides
            left_face = bgr_image[y:y + height, x + int(width / 2):x + width]
            left_face_gray = bgr_image[y:y + height, x + int(width / 2):x + width]
            print('closed')
            right_face = bgr_image[y:y + height, x:x + int(width / 2)]
            right_face_gray = bgr_image[y:y + height, x:x + int(width / 2)]

            # Detect the left eye
            left_eye = left_eye_detector.detectMultiScale(
                left_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Detect the right eye
            right_eye = right_eye_detector.detectMultiScale(
                right_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(24, 24),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # we suppose the eyes are open
            eye_status_right = '1'
            eye_status_left = '1'

            # For each eye check whether the eye is closed.
            # If one is closed we conclude the eyes are closed
            for (ex, ey, ew, eh) in right_eye:
                color = (0, 255, 0)
                pred = predict(right_face[ey:ey + eh, ex:ex + ew], eye_classifier)
                if pred == 'closed':
                    eye_status_right = '0'

            for (ex, ey, ew, eh) in left_eye:
                color = (0, 255, 0)
                pred = predict(left_face[ey:ey + eh, ex:ex + ew], eye_classifier)
                if pred == 'closed':
                    eye_status_left = '0'

            # if eye_status_left == '0' and eye_status_right == '0':
            #     print('closed')

        ######################################################


        #draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color,0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

