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
from counter_module import Counter

# from areas_module import generate_polygons

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
    objects_bbox = dict()
    # numero di bbox per cui eseguire la media (serve per la stabilizzazione dell'algoritmo)
    num_bboxes = 3
    # indica se il veicolo è in movimento
    is_moving = dict()
    # insieme dei veicoli entrati nello stallo di parcheggio
    objects_inside = set()
    # insieme di veicoli che sostano o marciano nella zona di ingresso del parcheggio
    was_outside = set()
    # insieme di veicoli che sostano o marciano all'interno del parcheggio, nella zona di uscita
    was_inside = set()
    # insieme di veicoli usciti dal parcheggio
    exited_objects = set()

    counter = Counter(num_bboxes)
    # il counter restituisce le coordinate dei veicoli generati dal roi_module
    polygons_list = counter.polygons()
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
        with open('./data/classes/allowed_classes.txt', 'r') as file:
            allowed_classes = file.read()

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
                id_ = track.track_id
                counter.initialize_bboxes(str(id_))
                is_moving[str(id_)] = ''

        objects_count = []
        index = 0
        for polygon in polygons_list:
            # Stampa a video del poligono corrispondente alla zona esterna
            cv2.polylines(full_frame, [polygon[0]], True, (245, 66, 230), 2)
            # Stampa a video del poligono corrispondente alla zona interna
            cv2.polylines(full_frame, [polygon[1]], True, (90, 245, 66), 2)

            cv2.rectangle(full_frame, (polygon[1][2][0] + 70, polygon[1][2][1] + 1),
                          (polygon[1][2][0] + 175, polygon[1][2][1] + 25),
                          (45, 122, 50), -1)

            # stamapa a video dei risultati del counter per ogni ROI
            cv2.putText(frame, "Regione " + str(index), (polygon[1][2][0] + 75,
                                                         polygon[1][2][1] + 16),
                        0, 0.6, (255, 255, 255), 2)

            objects_count.append(0)
            index += 1

        n_rows = len(polygons_list)
        y_min = 10
        y_max = y_min + (n_rows * 34)
        cv2.rectangle(full_frame, (15, y_min), (545, y_max), (0, 0, 0), -1)
        for i in range(n_rows):  # entrata
            if polygons_list[i][2]:
                string_to_print = " oggetti entrati: "
            else:  # uscita
                string_to_print = " oggetti usciti: "

            cv2.putText(full_frame, "Regione " + str(i) + string_to_print + str(objects_count[i]),
                        (20, (32 + 40 * i)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            id_ = track.track_id

            # Il counter restituisce per ogni roi il numero di veicoli conteggiati, nonchè il bbox da stampare
            counter_results = counter.area_control(str(id_), class_name, bbox, frame_num)

            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            # il counter restituisce bbox_to_print, bounding box risultante dalla stabilizzazione
            bbox_to_print = counter_results[1]
            # stampa a video grazie alla libreria cv2 dei bounding box per ogni rilevazione
            cv2.rectangle(frame, (int(bbox_to_print[0]), int(bbox_to_print[1])),
                          (int(bbox_to_print[2]), int(bbox_to_print[3])), color, 2)

            cv2.rectangle(frame, (int(bbox_to_print[0]), int(bbox_to_print[1] - 30)),
                          (int(bbox_to_print[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                           int(bbox_to_print[1])), color, -1)
            cv2.putText(frame, class_name + str(id_), (int(bbox_to_print[0]), int(bbox_to_print[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

            # Regione dedicata alla stampa del numero di oggetti rilevati dal counter
            cv2.rectangle(full_frame, (15, y_min), (545, y_max), (0, 0, 0), -1)
            for key, val in counter_results[0].items():
                objects_count[key] = int(val)
                if polygons_list[key][2]:
                    string_to_print = " oggetti entrati: "
                else:  # uscita
                    string_to_print = " oggetti usciti: "

                cv2.putText(full_frame, "Regione " + str(key) + string_to_print + str(val),
                            (20, (32 + 40 * key)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0), 2)



            # se il flag è a true stampa sul terminale le informazioni della rilevazione
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
                                                                                                    class_name, (
                                                                                                        int(bbox[0]),
                                                                                                        int(bbox[1]),
                                                                                                        int(bbox[2]),
                                                                                                        int(bbox[3]))))

        # calcolo dei frame per secondo
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
