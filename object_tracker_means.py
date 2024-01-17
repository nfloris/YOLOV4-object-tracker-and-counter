import os

# Commenta la riga sottostante per disabilitare gli output di logging di TensorFlow
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
    # Definizione dei parametri
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # inizializzazione DeepSort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # inizializzazione del tracker
    tracker = Tracker(metric)

    # caricamento delle impostazione di configurazione per il monitoraggio e conteggio dei veicoli
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    '''
    class Vehicle:
        def init(self):
            self.current_bbox = dict()
            self.previous_bbox = dict()
            self.is_moving = False
            self.was_outside = False
    '''

    # se il flag è impostato carica il modello tensorflow lite (tflite)
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # altrimenti carica il modello standard di tensorflow (saved_model)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # inizio della cattura del video
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # imposta il flag per salvare il video di output in locale
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # bounding box dei veicoli
    vehicles_bbox = dict()
    # numero di bbox per cui eseguire la media (serve per la stabilizzazione dell'algoritmo)
    num_bboxes = 3
    # indica se il veicolo è in movimento
    is_moving = dict()
    # insieme dei veicoli entrati nello stallo di parcheggio
    vehicles_inside = set()
    # insieme di veicoli che sostano o marciano nella zona di ingresso del parcheggio
    was_outside = set()
    # insieme di veicoli che sostano o marciano all'interno del parcheggio, nella zona di uscita
    was_inside = set()
    # insieme di veicoli usciti dal parcheggio
    exited_vehicles = set()
    # Ciclo che processo ogni frame del video
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print("Il video è terminato o l'esecuzione è fallita (prova un diverso formato!)")
            break
        frame_num += 1
        # print('Frame #: ', frame_num)
        # imposta la dimensione in pixel dei frame processati
        frame = cv2.resize(frame, (1280, 720))
        frame_size = frame.shape[:2]
        full_frame = frame
        roi_minx = 70
        roi_maxx = 400
        roi_miny = 5
        roi_maxy = 1200
        frame = frame[roi_minx:roi_maxx, roi_miny:roi_maxy]

        def calc_bboxes_mean(b_boxes):
            mean = [0, 0, 0, 0]
            nboxes = len(b_boxes)
            print(len(b_boxes))
            for i in range(nboxes):
                mean[0] += bboxes[i][0]
                mean[1] += bboxes[i][1]
                mean[2] += bboxes[i][2]
                mean[3] += bboxes[i][3]
            for i in range(len(mean)):
                mean[i] = mean[i] / len(mean)
            return mean

        # Modify the following line to resize the frame to the ROI size
        # image_data = cv2.resize(frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width], (input_size, input_size))

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        # esegue il monitoraggio su tflite se il flag è impostato
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
            # dal batch di predicono i bounding boxes
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

        # converte i dati in array numpy e taglia gli elementi inutilizzati
        # i dati si compongono di bounding boxes, classe rilevata e punteggio assegnato alla classe
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # formato dei bounding boxes dopo la normalizzazione: ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # salva tutte le predizioni in un singolo parametro per semplificare la chiamata di funzioni
        pred_bbox = [bboxes, scores, classes, num_objects]

        # dal file di config vengono rilevate le classi permesse
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # per impostazione di default concedi il permesso a tutte le classi presenti nel file .names
        allowed_classes = list(class_names.values())
        # classi permesse custom
        # allowed_classes = ['person']

        # per ogni oggetto si utilizza l'indice di classe per ottenere il nome della classe associato
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
        # se il flag è a true si stampa a video il numero di oggetti rilevati (sconsigliato!)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # codifica i monitoraggi di yolo per poi passarle al tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # inizializza la color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # richiama il tracker
        tracker.predict()
        tracker.update(detections)

        # si parte dal terzo frame per evitare complicazioni
        if frame_num <= 3:
            # si considera una predizione e si inizializzano i parametri
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                if class_name != 'person':
                    id_ = track.track_id
                    vehicles_bbox[str(id_)] = list()
                    is_moving[str(id_)] = ''

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            id_ = track.track_id

            roi_color = (0, 255, 0)  # colore dell'area di interesse

            if class_name != 'person':
                '''
                if str(id_) in vehicles_bbox.keys():
                    print(vehicles_bbox[str(id_)])
                    previous_bbox[str(id_)] = vehicles_bbox[str(id_)][num_bboxes]
                else:
                    previous_bbox[str(id_)] = bbox
                vehicles_bbox[str(id_)][num_bboxes] = bbox

                if frame_num >= num_bboxes + 3:
                    boundings.pop(0)
                if frame_num >= 3:
                    boundings.append(bbox)
                    vehicles_bbox[str(id_)] = boundings
                    if id_ == 16:
                        print(vehicles_bbox)

                current_centre = [((int(bbox[0]) + int(bbox[2]))/2), ((int(bbox[1]) + int(bbox[3]))/2)]
                previous_centre = [((int(previous_bbox[str(id_)][0]) + int(previous_bbox[str(id_)][2])) / 2),
                                   ((int(previous_bbox[str(id_)][1]) + int(previous_bbox[str(id_)][3])) / 2)]

                bbox[0] = (previous_bbox[str(id_)][0] + bbox[0])/2
                bbox[1] = (previous_bbox[str(id_)][1] + bbox[1])/2

                dst = distance.euclidean(previous_centre, current_centre)
                if dst >= 5: #treshold
                    is_moving[str(id_)] = 'Moving'
                else: is_moving[str(id_)] = ''
                '''
            if class_name != 'person':
                if True:  # is_moving[str(id_)] == 'Moving':
                    if str(id_) not in vehicles_bbox:
                        vehicles_bbox[str(id_)] = list()
                    current_centre = [((int(bbox[0]) + int(bbox[2])) / 2), ((int(bbox[1]) + int(bbox[3])) / 2)]
                    if len(vehicles_bbox[str(id_)]) >= num_bboxes:
                        vehicles_bbox[str(id_)].pop(0)
                    if frame_num >= 3 and len(vehicles_bbox[str(id_)]) < num_bboxes:
                        vehicles_bbox[str(id_)].append(bbox)

                        bbox_to_print = [0, 0, 0, 0]

                        for elem in vehicles_bbox[str(id_)]:
                            bbox_to_print[0] += elem[0]
                            bbox_to_print[1] += elem[1]
                            bbox_to_print[2] += elem[2]
                            bbox_to_print[3] += elem[3]

                        for i in range(4):
                            bbox_to_print[i] = bbox_to_print[i] / len(vehicles_bbox[str(id_)])

                    else:
                        bbox_to_print = [0, 0, 0, 0]

                    if 716 < current_centre[0] < 1045 and 110 < current_centre[1] < 240:
                        if class_name + str(id_) not in was_outside:
                            was_outside.add(class_name + str(id_))
                            print("Sta per entrare " + class_name + str(id_))

                    if 715 > current_centre[0] > 410 and 143 > current_centre[1] > 93:
                        if class_name + str(id_) not in vehicles_inside and class_name + str(id_) in was_outside:
                            vehicles_inside.add(class_name + str(id_))
                            print("Entrata " + class_name + str(id_))

                    if 5 < current_centre[0] < 200 and 93 < current_centre[1] < 185:
                        if class_name + str(id_) not in was_inside:
                            was_inside.add(class_name + str(id_))
                            print("Sta per uscire " + class_name + str(id_))

                    if 65 < current_centre[0] < 450 and 190 < current_centre[1] < 320:
                        if class_name + str(id_) not in exited_vehicles and class_name + str(id_) in was_inside:
                            exited_vehicles.add(class_name + str(id_))
                            print("Uscita " + class_name + str(id_))

                # Inside the loop where the frame is being processed
                cv2.rectangle(full_frame, (roi_miny, roi_minx), (roi_maxy, roi_maxx), (255, 255, 0), 2)
                # Area d'interesse - parcheggio
                cv2.rectangle(frame, (715, 143), (410, 93), (245, 66, 230), 2)
                # Area d'interesse - esterno del parcheggio
                cv2.rectangle(frame, (716, 110), (1045, 240), (90, 245, 66), 2)

                # Area di interesse - interno del parcheggio per l'uscita dei veicoli
                cv2.rectangle(frame, (5, 93), (200, 185), (245, 66, 230), 2)
                # Area di interesse - esterno del parcheggio per l'uscita dei veicoli
                cv2.rectangle(frame, (65, 190), (450, 320), (90, 245, 66), 2)

                # Distanza in X: 191 pixel - Distanza in Y: 20 pixel
                # Prima X minore, prima Y maggiore
                # cv2.line(frame, (665, 153), (856, 133), (0, 255, 255), 5)
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                # cv2.rectangle(frame, (int(bbox_to_print[0]), int(bbox_to_print[1])), (int(bbox_to_print[2]), int(bbox_to_print[3])), color, 2)
                # print(class_name + "-" + str(track.track_id), ' x0y0: ', int(bbox[0]), ' ', int(bbox[1]), ' x1y1: ',
                # int(bbox[2]), ' ', int(bbox[3]))
                # cv2.rectangle(frame, (int(bbox_to_print[0]), int(bbox_to_print[1]-30)),
                # (int(bbox_to_print[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox_to_print[1])), color, -1)
                # cv2.putText(frame, class_name + str(id_), (int(bbox_to_print[0]), int(bbox_to_print[1]-10)), 0, 0.75, (255,255,255), 2)
                # cv2.rectangle(full_frame, (15, 10), (345, 77), (0, 0, 0), -1)
                # cv2.putText(full_frame, "Veicoli entrati: " + str(len(vehicles_inside)),
                # (20, 32), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0), 2)
                # cv2.putText(full_frame, "Veicoli usciti:  " + str(len(exited_vehicles)),
                # (20, 72), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0), 2)
                # cv2.putText(full_frame, "Veicoli entrati: " + str(len(vehicles_inside)), (20, 30), 0, 0.75, (255, 255, 255), 2)
                # cv2.putText(full_frame, "Veicoli usciti:  " + str(len(exited_vehicles)), (20, 60), 0, 0.75, (255, 255, 255), 2)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
                                                                                                    class_name, (
                                                                                                    int(bbox[0]),
                                                                                                    int(bbox[1]),
                                                                                                    int(bbox[2]),
                                                                                                    int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
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
