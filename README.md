# Müdetector
Driver drowsiness detection using machine vision

## Requirements
- Python
- Dlib
- OpenCV
- imutils
- playsound
- NumPy
- SciPy

## Running Müdetector
To run Müdetector, browse to its main directory via the terminal and run ```detector.py```.

Unless altered, Müdetector will use the pretrained face predictor (```predictors/face.dat```).

## Using your own predictor
To use your own  predictor, pass your custom predictor's path to the load_predictor function as predictor_path.
Example: ```load_predictor(predictor_path="predictors/custom_predictor.dat")```
