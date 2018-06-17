import face_recognition
import cv2
import os
import numpy as np
import subprocess as sp
from PIL import ImageFont, ImageDraw, Image
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
if (video_capture.isOpened()):# 判断视频是否打开
    print ("Open camera")
else:
    print ("Fail to open camera!")
# 设置摄像属性
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 2560x1920 2217x2217 2952×1944 1920x1080
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video_capture.set(cv2.CAP_PROP_FPS, 5)
# 视频属性
size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sizeStr = str(size[0]) + 'x' + str(size[1])
fps = video_capture.get(cv2.CAP_PROP_FPS)  # 30p/self
fps = int(fps)
hz = int(1000.0 / fps)
print("size: {} fps: {} hz: {}".format(sizeStr, str(fps), str(hz)))
# ffmpeg 推流 初始化
command = ['ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', sizeStr,
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv',
        'rtmp://127.0.0.1:1935/rtmplive/room']
proc = sp.Popen(command, stdin=sp.PIPE,shell=False)
# Define the codec and create VideoWriter object
# 初始化写入文件
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('./output.avi',fourcc, fps, size)
# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []
# 批量读取特征
for root, dirs, files in os.walk("./face", topdown=False):
    for name in files:
        print(os.path.join(root, name))
        image = face_recognition.load_image_file(os.path.join(root, name))
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name[0:name.find(".")])
print(known_face_names)
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.4)
            name = "Unknown"
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("SimHei.ttf", 15)
        draw.text((left + 4, bottom - 24), name, font=font)
        frame = cv2.cvtColor(np.array(im), 1)
    # Display the resulting image
    cv2.imshow('Video', frame)
    # 写入文件
    out.write(frame)
    # 推流
    proc.stdin.write(frame.tostring())
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
out.release()
cv2.destroyAllWindows()
