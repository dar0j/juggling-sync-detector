import re
from keras.utils import Sequence
from collections import OrderedDict
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
import cv2
import csv
import pandas as pd
import numpy as np
import random

class PatternDataLoader:
    def __init__(self, filename, length=60, split=(0.8, 0.1, 0.1)):
        self.length = length
        self.filename = filename
        self.patternsFolder = "../patterns/"
        self.excluded_tricks = ["5_R(8x,2x).csv"] # Ejemplos que no se deben transformar porque no tienen contraparte .csv (L/R)
        # normalize split to sum 1
        s = np.array(split, dtype=float)
        s = s / s.sum()
        (self.trainx, self.trainy), (self.valx, self.valy), (self.testx, self.testy) = self._getSetsBySplit(s)
        # augmentation / shuffle
        self.trainx = self._shuffleBalls(self.trainx)
        self.trainx, self.trainy = shuffle(self.trainx, self.trainy)

    def _getSetsBySplit(self, split):
        trainX, trainY = [], []
        valX, valY = [], []
        testX, testY = [], []
        with open(self.patternsFolder + self.filename) as patternlist:
            count = 0
            for filename in patternlist:
                filename = filename.rstrip('\n')
                # original file
                annotations = pd.read_csv(self.patternsFolder + filename, header=None).values
                n = annotations.shape[0]
                t_end = int(n * split[0])
                v_end = t_end + int(n * split[1])
                # slices
                parts = [
                    annotations[0:t_end],
                    annotations[t_end:v_end],
                    annotations[v_end:n]
                ]
                targets = [(trainX, trainY), (valX, valY), (testX, testY)]
                for part, (Xlist, ylist) in zip(parts, targets):
                    patterns = self._annotationsToX(part)
                    if patterns.size:
                        Xlist.extend(patterns)
                        ylist.extend([count] * patterns.shape[0])

                # flip counterpart (read separately and split según su propio tamaño)
                if filename in self.excluded_tricks:
                    count += 1
                    continue
                pattern_side = r"_(R|L)?"
                side_match = re.search(pattern_side, filename) # si hubiera un nombre que no coincide nada con el patrón, tirará AttributeError
                side = side_match.group(1)
                if side:
                    nuevo = self._convertSide(filename, side)
                    try:
                        annotations_new = pd.read_csv(self.patternsFolder + nuevo, header=None).values
                        annotations_new = self._flipAnnotations(annotations_new)
                        n2 = annotations_new.shape[0]
                        t2_end = int(n2 * split[0])
                        v2_end = t2_end + int(n2 * split[1])
                        parts2 = [
                            annotations_new[0:t2_end],
                            annotations_new[t2_end:v2_end],
                            annotations_new[v2_end:n2]
                        ]
                        for part, (Xlist, ylist) in zip(parts2, targets):
                            patterns = self._annotationsToX(part)
                            if patterns.size:
                                Xlist.extend(patterns)
                                ylist.extend([count] * patterns.shape[0])
                    except FileNotFoundError:
                        # si no existe el fichero flip, se omite
                        pass

                count += 1

        # convertir a np.array y one-hot para cada conjunto
        trainX = np.array(trainX); trainY = to_categorical(np.array(trainY)) if len(trainY) else np.array([])
        valX = np.array(valX);     valY = to_categorical(np.array(valY))     if len(valY) else np.array([])
        testX = np.array(testX);   testY = to_categorical(np.array(testY))   if len(testY) else np.array([])
        return (trainX, trainY), (valX, valY), (testX, testY)

    def _flipAnnotations(self, annotations):
        annotations[:,::2] = -annotations[:,::2]
        annotations[:,[0,1,2,3]] = annotations[:,[2,3,0,1]]
        return annotations

    def _annotationsToX(self, annotations):
        X = []
        for i in range(annotations.shape[0]-self.length+1):
            pattern = np.array(annotations[i:i+self.length], dtype=np.float32)
            pattern[:,::2] = pattern[:,::2] - np.mean(pattern[:,::2])
            pattern[:,1::2] = pattern[:,1::2] - np.mean(pattern[:,1::2])
            pattern = pattern / (pattern.std() + 1e-8) # evita division por cero
            X.append(pattern)

        return np.array(X)

    def _shuffleBalls(self, clean_set):
        reshaped_set = np.reshape(clean_set, (len(clean_set), self.length, -1, 2))
        for i in range(len(reshaped_set)):
            for j in range(self.length):
                np.random.shuffle(reshaped_set[i,j,2:])
        clean_set = np.reshape(reshaped_set, (len(clean_set), self.length, -1))
        return clean_set

    def getNames(self):
        with open(self.patternsFolder + self.filename) as patternlist:
            names = []
            for filename in patternlist:
                filename = filename.rstrip('\n')
                names.append(filename[2:-4])
        return names
    
    def _convertSide(self, filename, side):
        def swap_coords(match):
            a, b = match.group(1), match.group(2)
            return f"({b},{a})"
        swapped = re.sub(r"\(([^,]+),([^,]+)\)", swap_coords, filename)
        if side == 'L':
            flipped = swapped.replace('L', 'R', 1)
        elif side == 'R':
            flipped = swapped.replace('R', 'L', 1)
        return flipped

    # def getRaw(self, start=0, stop=3000): #cambiar  como lee los flipped filenames
    #     """
    #     Devuelve X (patterns normalizados) y y (etiquetas enteras) para todo el conjunto
    #     útil para hacer KFold cross-validation.
    #     """
    #     X = []
    #     y = []
    #     with open(self.patternsFolder + self.filename) as patternlist:
    #         count = 0
    #         for filename in patternlist:
    #             filename = filename.rstrip('\n')
    #             annotations = pd.read_csv(self.patternsFolder + filename, header=None).values
    #             annotations = annotations[start:stop:1]
    #             patterns = self._annotationsToX(annotations)
    #             X.extend(patterns)
    #             y.extend([count] * patterns.shape[0])

    #             # añadir la versión "flip" (igual que en _getSet)
    #             if filename[2] == 'L':
    #                 other = filename[:2] + 'R' + filename[3:]
    #             elif filename[2] == 'R':
    #                 other = filename[:2] + 'L' + filename[3:]
    #             else:
    #                 other = None
    #             if other is not None:
    #                 annotations = pd.read_csv(self.patternsFolder + other, header=None).values
    #                 annotations = annotations[start:stop:1]
    #                 annotations = self._flipAnnotations(annotations)
    #                 patterns = self._annotationsToX(annotations)
    #                 X.extend(patterns)
    #                 y.extend([count] * patterns.shape[0])

    #             count += 1
    #     return np.array(X), np.array(y)
