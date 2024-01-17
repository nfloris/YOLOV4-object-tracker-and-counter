import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from areas_module import generate_polygons

class Counter:
    def __init__(self, num_bboxes):
        self.objects_bbox = dict()
        self.num_bboxes = num_bboxes
        self.objects_inside = set()
        self.was_outside = set()
        self.was_inside = set()
        self.exited_objects = set()
        self.area_mask = None


    def initialize_bboxes(self, id_):
        self.objects_bbox[id_] = list()

    def polygons(self):
        self.area_mask = generate_polygons()
        return self.area_mask

    def area_control(self, id_, class_name, bbox, frame_num):
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

        polygons = []
        for i in range(len(self.area_mask)):
            #points = [(self.area_mask[i][0], self.area_mask[i][2]),
                   #   (self.area_mask[i][1], self.area_mask[i][2]),
                    #  (self.area_mask[i][0], self.area_mask[i][3]),
                    #  (self.area_mask[i][1], self.area_mask[i][3])]
            polygons.append(Polygon(self.area_mask[i]))

        # Controllo sulla zona esterna delimitata per l'ingresso
        if polygons[0].contains(Point(current_centre)):
            '''
            if self.area_mask[0][0] < current_centre[0] < self.area_mask[0][1] \
                    and self.area_mask[0][2] < current_centre[1] < self.area_mask[0][3]:
            '''
            if class_name + id_ not in self.was_outside:
                self.was_outside.add(class_name + id_)
                print("Sta per entrare " + class_name + id_)

        # Controllo sulla zona interna delimitata per l'ingresso
        '''
        if self.area_mask[1][0] < current_centre[0] < self.area_mask[1][1] \
                and self.area_mask[1][2] < current_centre[1] < self.area_mask[1][3]:
        '''
        if polygons[1].contains(Point(current_centre)):
            if class_name + id_ not in self.objects_inside and class_name + id_ in self.was_outside:
                self.objects_inside.add(class_name + id_)
                print("Entrata " + class_name + id_)

        # Controllo sulla zona interna delimitata per l'uscita
        '''
        if self.area_mask[2][0] < current_centre[0] < self.area_mask[2][1] \
                and self.area_mask[2][2] < current_centre[1] < self.area_mask[2][3]:
        '''
        if polygons[2].contains(Point(current_centre)):
            if class_name + id_ not in self.was_inside:
                self.was_inside.add(class_name + str(id_))
                print("Sta per uscire " + class_name + str(id_))

        # Controllo sulla zona esterna delimitata per l'uscita
        '''
        if self.area_mask[3][0] < current_centre[0] < self.area_mask[3][1] \
                and self.area_mask[3][2] < current_centre[1] < self.area_mask[3][3]:
        '''
        if polygons[3].contains(Point(current_centre)):
            if class_name + id_ not in self.exited_objects and class_name + id_ in self.was_inside:
                self.exited_objects.add(class_name + id_)
                print("Uscita " + class_name + id_)

        return [len(self.objects_inside), len(self.exited_objects), bbox_to_print]
