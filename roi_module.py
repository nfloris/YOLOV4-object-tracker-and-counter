import numpy as np
from shapely.geometry.polygon import Polygon


class ROI:
    def __init__(self):
        # regione 1: rappresenta la regione in cui un oggetto deve transitare prima di entrare nella regione 2
        self.region_one_coord = None
        # regione 2 è si vuole controllare la presenza di un oggetto, regione 1 è la "direzione"
        self.region_two_coord = None
        # is_entrance determina se si vuole controllare il flusso in ingresso di oggetti o in uscita
        self.is_entrance = True
        # was_outside è una struttura che contiene gli id di tutti gli oggetti rilevati nella regione 1
        self.was_outside = set()
        # is_inside contiene gli id degli oggetti rilevati nella regione 2,
        # a patto che siano stati precedentemente rilevati nella reione 1
        self.is_inside = set()
        # classes rappresenta la lista di classi permesse per tale regione di interesse
        self.classes = list()


def generate_roi():

    # il metodo generate_roi si occupa di estrarre da file csv le informazioni relative alle regioni di interesse,
    # e convertirle in dati per la rispettiva classe

    # Lettura dal file csv delle coordinate rilevate dal tool di safespotter:
    # viene escluso l'header ed eventuali campi None
    roi_locations = np.genfromtxt('area_mask.csv', delimiter=',', skip_header=1,
                                  converters={i: lambda x: float(x) if x != b'None' else np.nan for i in range(15)})

    # dal file 'area_classes.csv' vengono lette le classi permesse per ciascuna regione
    roi_classes = np.genfromtxt('./data/classes/area_classes.csv', delimiter=',', dtype=None)
    classes = list(roi_classes[0])
    classes = [element.decode('utf-8') for element in classes]
    roi_classes = roi_classes[1:]
    roi_classes = np.vectorize(lambda x: int(x.decode('utf-8')))(roi_classes)

    # lista di regioni di interesse
    roi_list = []
    # Si controlla ci sia almeno una coppia di poligoni
    if roi_locations.shape[0] > 1:
        loop_index = 0
        # si cicla lungo le righe del file csv, corrispondenti a un poligono: questi vengono considerati a coppie
        # (regione 1 e regione 2)
        while loop_index < (len(roi_locations) - 1):
            # si crea un'istanza della classe ROI
            roi = ROI()
            roi_one_index = loop_index
            roi_two_index = loop_index + 1
            roi.is_entrance = roi_locations[loop_index][9]

            # Nella posizione numero 8 dell'array si trova il tipo dell'area, se questi coincidono viene generato un
            # errore
            if roi_locations[roi_one_index][8] == roi_locations[roi_two_index][8]:
                error_message = "Errore, Le aree adiacenti non possono essere dello stesso tipo."
                raise RuntimeError(error_message)
            elif roi_locations[roi_one_index][8] == 0:
                # La prima regione d'interesse rappresenta la zona precedente all'entrata
                roi_one = roi_locations[roi_one_index][0:8]  # Vengono trattenute solo le info sulle coordinate
                roi_two = roi_locations[roi_two_index][0:8]
            else:
                # La seconda regione d'interesse è la zona precedente all'entrata, gli indici vengono scambiati
                roi_two = roi_locations[roi_one_index][0:8]
                roi_one = roi_locations[roi_two_index][0:8]

            # Si convertono le coordinate di safespotter_area_tool(1920*1080) in 1080*720
            for i in range(0, 8):
                roi_one[i] = int(roi_one[i] / 1.5)
                roi_two[i] = int(roi_two[i] / 1.5)

            # Si convertono le coordinate del file csv in un numpy array
            roi.region_one_coord = (np.array([[roi_one[0], roi_one[1]], [roi_one[2], roi_one[3]],
                                                      [roi_one[4], roi_one[5]], [roi_one[6], roi_one[7]]], np.int32))
            roi.region_two_coord = (np.array([[roi_two[0], roi_two[1]], [roi_two[2], roi_two[3]],
                                                      [roi_two[4], roi_two[5]], [roi_two[6], roi_two[7]]], np.int32))

            # Si conmpila il campo classes: il file contiene una serie di valori 0 o 1 a seconda che la classe
            # i-esima sia permessa per quella roi
            for i in range(len(roi_classes[0])):
                if roi_classes[int(loop_index/2)][i]:
                    roi.classes.append(classes[i])

            # tutte le regioni d'interesse (coppie di poligoni) vengono inserite all'interno di una lists
            roi_list.append(roi)
            # si valuta la coppia successiva
            loop_index += 2

            # se l'ultimo elemento fa riferimento ad un solo poligono senza la sua regione adiacente questa viene
            # ignorata dal loop
            if loop_index % 2:
                break

    # la lista delle roi viene restituita in output
    return roi_list
