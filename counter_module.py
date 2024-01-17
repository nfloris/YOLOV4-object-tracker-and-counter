import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from roi_module import ROI
from roi_module import generate_roi


class Counter:
    def __init__(self, num_bboxes):
        self.objects_bbox = dict()
        self.num_bboxes = num_bboxes
        self.roi_list = []
        self.counter = dict()

    def initialize_bboxes(self, id_):
        self.objects_bbox[id_] = list()

    def polygons(self):
        # L'area module produce la lista delle regioni d'interesse
        self.roi_list = generate_roi()

        # Vengono restituite al tracker solamente le coordinate delle regioni, senza la lista degli oggetti
        polygons_list = []
        for roi in self.roi_list:
            polygons_list.append([roi.region_one_coord, roi.region_two_coord, roi.is_entrance])
            roi.region_one_coord = Polygon(roi.region_one_coord)
            roi.region_two_coord = Polygon(roi.region_two_coord)

        return polygons_list

    def area_control(self, id_, class_name, bbox, frame_num):
        # Sequenza di istruzioni per la stabilizzazione dei bounding box:
        # vengono trattengono gli ultimi "num_bbox" bounding box relativi ad un determinato oggetto,
        # in seguito si esegue la media. Viene prodotto un bbox risultante chiamato bbox_to_print.
        # Questo valore sarà uno dei parametri restituiti al modulo tracker che si occuperà della stsmpa a video
        if id_ not in self.objects_bbox:
            self.objects_bbox[id_] = list()

        current_centre = [((int(bbox[0]) + int(bbox[2])) / 2), ((int(bbox[1]) + int(bbox[3])) / 2)]
        if len(self.objects_bbox[str(id_)]) >= self.num_bboxes:
            self.objects_bbox[id_].pop(0)

        if frame_num >= 3 and len(self.objects_bbox[id_]) < self.num_bboxes:
            self.objects_bbox[str(id_)].append(bbox)
            bbox_to_print = [0, 0, 0, 0]

            for elem in self.objects_bbox[str(id_)]:
                bbox_to_print[0] += elem[0]
                bbox_to_print[1] += elem[1]
                bbox_to_print[2] += elem[2]
                bbox_to_print[3] += elem[3]

            for i in range(4):
                bbox_to_print[i] = bbox_to_print[i] / len(self.objects_bbox[id_])

        else:
            bbox_to_print = [0, 0, 0, 0]
        # ------------------------------------------------------------------------------

        # sequenza di istruzioni per il conteggio degli oggetti
        # il tracker passa come parametro un oggetto, per tale oggetto viene controllata la posizione rispetto alle roi
        # se l'oggetto
        roi_id = 0
        for roi in self.roi_list:

            # region_one rappresenta "l'esterno"
            if roi.region_one_coord.contains(Point(current_centre)):
                if class_name + id_ not in roi.was_outside:
                    # Se non è mai stato in quella zona viene inserito nel set
                    roi.was_outside.add(class_name + id_)
                    if roi.is_entrance:
                        print("Sta per entrare " + class_name + str(id_) + " nella regione " + str(roi_id))
                    else:
                        print("Sta per uscire " + class_name + str(id_) + " dalla regione " + str(roi_id))

            # region_two rappresenta "l'interno"
            if roi.region_two_coord.contains(Point(current_centre)):
                # Se l'oggetto viene rilevato all'interno e precedentemente è stato all'esterno allora
                # può essere considerato "entrato" e viene conteggiato
                if class_name + id_ not in roi.is_inside and class_name + id_ in roi.was_outside:
                    roi.is_inside.add(class_name + id_)
                    if roi.is_entrance:
                        print("Entrato " + class_name + str(id_) + " nella regione " + str(roi_id))
                    else:
                        print("Uscito " + class_name + str(id_) + " dalla regione " + str(roi_id))

            self.counter[roi_id] = len(roi.is_inside)
            roi_id += 1

        return [self.counter, bbox_to_print]
