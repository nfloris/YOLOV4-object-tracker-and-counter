import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from scipy.spatial import distance
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    class Vehicle:
        def init(self):
            self.current_bbox = dict()
            self.previous_bbox = dict()
            self.is_moving = False
            self.was_outside = False

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (1280, 7220))

    frame_num = 0
    vehicles_bbox = dict()
    previous_bbox = dict()
    is_moving = dict()
    vehicles_inside = set()
    was_outside = set()
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        #print('Frame #: ', frame_num)
        frame = cv2.resize(frame, (1280, 720))
        frame_size = frame.shape[:2]
        full_frame = frame
        roi_minx = 70
        roi_maxx = 400
        roi_miny = 400
        roi_maxy = 1200
        frame = frame[roi_minx:roi_maxx, roi_miny:roi_maxy]

        roi_x = 500  # X-coordinate of the top-left corner of the ROI
        roi_y = 0  # Y-coordinate of the top-left corner of the ROI
        roi_width = 520  # Width of the ROI (in pixels)
        roi_height = 330  # Height of the ROI (in pixels)

        y_line = [205, 225]
        x_line = [665, 860]

        def is_inside_roi(bbox, roi_x, roi_y, roi_width, roi_height):
            x, y, w, h = bbox
            return (roi_x <= x < roi_x + roi_width) and (roi_y <= y < roi_y + roi_height)

        # Modify the following line to resize the frame to the ROI size
        #image_data = cv2.resize(frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width], (input_size, input_size))



        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        if frame_num <= 3:
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                if class_name != 'person':
                    id_ = track.track_id
                    vehicles_bbox[str(id_)] = bbox
                    is_moving[str(id_)] = ''

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            id_ = track.track_id

        # draw bbox on screen

            roi_color = (0, 255, 0)  # Red color for the ROI bounding box (in BGR format)


            if class_name != 'person':
                if str(id_) in vehicles_bbox.keys():
                    previous_bbox[str(id_)] = vehicles_bbox[str(id_)]
                else:
                    previous_bbox[str(id_)] = bbox
                vehicles_bbox[str(id_)] = bbox

                current_centre = [((int(bbox[0]) + int(bbox[2]))/2), ((int(bbox[1]) + int(bbox[3]))/2)]
                previous_centre = [((int(previous_bbox[str(id_)][0]) + int(previous_bbox[str(id_)][2])) / 2),
                                   ((int(previous_bbox[str(id_)][1]) + int(previous_bbox[str(id_)][3])) / 2)]

                dst = distance.euclidean(previous_centre, current_centre)
                if dst >= 5: #treshold
                    is_moving[str(id_)] = 'Moving'
                else: is_moving[str(id_)] = ''

                if is_moving[str(id_)] == 'Moving' \
                        and current_centre[0] > 321 and current_centre[1] > 110 \
                        and current_centre[0] < 650 and current_centre[1] < 240:
                    if class_name + str(id_) not in was_outside:
                        was_outside.add(class_name + str(id_))
                        print("Sta per entrare " + class_name + str(id_))

                #if id_ == 2:
                    #print(str(current_centre[0]) + " " + str(current_centre[1]))

                if is_moving[str(id_)] == 'Moving' \
                        and current_centre[0] < 320 and current_centre[1] < 143\
                        and current_centre[0] > 15 and current_centre[1] > 93:
                    if class_name + str(id_) not in vehicles_inside and class_name + str(id_) in was_outside:
                        vehicles_inside.add(class_name + str(id_))
                        print("Entrata " + class_name + str(id_))

                '''
                print(str(id_), "- Distance = ", dst)
                print('PREVIOUS BBOX: \n', class_name + "-" + str(track.track_id), ' x0y0: ', int(previous_bbox[str(id_)][0]),
                    ' ', int(previous_bbox[str(id_)][1]), ' x1y1: ', int(previous_bbox[str(id_)][2]), ' ', int(previous_bbox[str(id_)][3]))
                print('CUURENT BBOX: \n', class_name + "-" + str(track.track_id), ' x0y0: ', int(vehicles_bbox[str(id_)][0]),
                    ' ', int(vehicles_bbox[str(id_)][1]), ' x1y1: ', int(vehicles_bbox[str(id_)][2]), ' ', int(vehicles_bbox[str(id_)][3]))
                '''

                # Inside the loop where the frame is being processed
                #cv2.rectangle(full_frame, (roi_miny, roi_minx), (roi_maxy, roi_maxx), roi_color, 2)
                # Area d'interesse - parcheggio
                #cv2.rectangle(frame, (320, 143), (15, 93), (245, 66, 230), 2)
                # Area d'interesse - esterno del parcheggio
                #cv2.rectangle(frame, (321, 110), (650, 240), (90, 245, 66), 2)
                #cv2.line(full_frame, (x_line[0], y_line[1]), (x_line[1], y_line[0]), (255, 0, 0), 5)
                # Distanza in X: 191 pixel - Distanza in Y: 20 pixel
                # Prima X minore, prima Y maggiore
                #cv2.line(frame, (270, 153), (461, 133), (0, 255, 255), 5)
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                # print(class_name + "-" + str(track.track_id), ' x0y0: ', int(bbox[0]), ' ', int(bbox[1]), ' x1y1: ', int(bbox[2]), ' ', int(bbox[3]))
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + str(id_) + ' - ' + is_moving[str(id_)], (int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                cv2.putText(full_frame, "Veicoli entrati: " + str(len(vehicles_inside)), (20, 20), 0, 0.75, (255, 255, 255), 2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        #print("FPS: %.2f" % fps)
        result = np.asarray(full_frame)
        result = cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
