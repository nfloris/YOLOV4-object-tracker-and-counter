# YOLOV4 Monitoraggio e Conteggio veicoli

# Fonte algoritmo YOLO: yolov4-deepsort
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zmeSTP3J5zu2d5fHgsQC06DyYEYJFXq1?usp=sharing)

## Demo del sistema di monitoraggio su veicoli
<p align="center"><img src="data/helpers/cars.gif"\></p>

## Demo del counter su veicoli che entrano ed escono da un parcheggio
<p align="center"><img src="data/helpers/counter_example.gif"\></p>

### GUIDA ALL'INSTALLAZIONE
Per iniziare, crea l'envirorment Conda, una volta assicurato di trovarti all'interno della cartella "yolov4-deepsort-master".

### Versione 1: sfrutta la GPU della tua macchina (scegliere la seconda opzione nel caso non si possegga una GPU)
```bash
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Versione 2:
```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu
```

## Scarica i pesi ufficiali di YOLOv4 Pre-trainati
scarica qui: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

-> Una volta scaricati, copia il file e incollalo nella cartella 'data' della repository

### In alternativa, puoi scaricare i pesi della versione YOLOV4-Tiny
E' una versione meno precisa di YOLOV4 ma più rapida.
scarica il file qui: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

## Esegui il 'Tracker' con YOLOV4 o YOLOV4-tiny
Per implementare il monitoraggio di oggetti, e in questo caso il monitoraggio e conteggio di veicoli, per prima cosa bisogna convertire i pesi scaricati (file .weights) nel corrispondente modello TensorFlow che verrà salvato come una cartella di checkpoint. Tutto ciò di cui si ha bisogno è eseguire lo script 'object_tracker.py' in modo tale da eseguire il monitoraggio tramite YOLOV4, DeepSort e TensorFlow.
```bash
# Converti i pesi Darknet in TensorFlow
python save_model.py --model yolov4 

# Esegui il tracker su un video presente nella cartella ./data/video tramite l'opzione --video (puoi inseire le tue registrazioni)
# L'opzione --output permette di salvare il risultato dell'esecuzione nella cartella /outputs/demo
python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4

# Puoi anche eseguire il codice avendo come sorgente la registrazione della webcam (impostando il flag 'video' a 0)
python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4
```

## Eseguire il Tracker con YOLOV4-Tiny
I seguenti comandi ti permetteranno di eseguire il codice utilizzando la versione Tiny di YOLOV4. Consente di ottenere una maggiore velocità di esecuzione (FPS) sacrificando una percentuale di accuratezza. Utile per chi non ha una GPU nella propria macchina.
Assicurati di aver scaricato i pesi di yolov4-tiny (link sopra) e di averli copiati nella cartella 'data'. 
Assicurati inoltre che il nome attribuito al file .weights e il nome del file citato nella linea di comando coincidano!

``` bash
# Salva il modello
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny

# Esegui il tracker
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny
```

### References:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
