from patterndataloader import PatternDataLoader
from sklearn.model_selection import KFold
from keras.utils import to_categorical
import numpy as np
import cv2

# Parámetros
FOLDS = 5
EPOCHS = 20
BATCH = 32

pdl = PatternDataLoader("4balls", length=30)  # cambiar a "4balls"/"5balls"/"6balls" según tus ficheros
names = pdl.getNames()
num_classes = len(names)

# Obtener datos brutos (X: (N,length,dim), y: int labels)
X, y_int = pdl.getRaw()

# normalizar/reshape si es necesario para la red (ya lo hace _annotationsToX)
y_onehot = to_categorical(y_int, num_classes=num_classes)

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=0)
fold = 0
scores = []
for train_idx, val_idx in kf.split(X):
    fold += 1
    print(f"Fold {fold}/{FOLDS}")
    trainx, trainy = X[train_idx], y_onehot[train_idx]
    valx, valy = X[val_idx], y_onehot[val_idx]

    # construir modelo (mismo esqueleto que ya usabas; ajustar input_shape y salida)
    model = Sequential()
    model.add(Flatten(input_shape=trainx.shape[1:3]))
    model.add(Dense(units=60, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(LeakyReLU())
    model.add(Dense(units=60, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(LeakyReLU())
    model.add(Dense(units=60, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(LeakyReLU())
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=[metrics.categorical_accuracy])

    model.fit(x=trainx, y=trainy, batch_size=BATCH, epochs=EPOCHS, validation_data=(valx, valy))
    score = model.evaluate(valx, valy)
    print("Fold score:", score)
    scores.append(score[1])  # accuracy

print("CV mean accuracy: %.3f ± %.3f" % (np.mean(scores), np.std(scores)))

model = load_model(saveModelFilename)

metrics = model.evaluate(pdl.testx, pdl.testy)
print("Testset Accuracy: %.1f" % (metrics[1]*100))
pred = model.predict(pdl.testx)
pred = np.argmax(pred, axis=1)
testy  = np.argmax(pdl.testy, axis=1)

names = pdl.getNames()
testy = [names[i] for i in testy]
pred = [names[i] for i in pred]
skplt.metrics.plot_confusion_matrix(testy, pred, x_tick_rotation=45, title=" ", text_fontsize="large")
plt.show()








# sample = np.random.randint(0,pdl.valx.shape[0])
#
# recording = pdl.valx[sample]
#
# recording = recording - np.min(recording)
# recording = recording * 256 / np.max(recording)
# recording = recording.astype(np.uint8)
#
# for i in range(0,recording.shape[0]):
#     canvas = np.zeros((256,256,3), dtype=np.uint8)
#     cv2.line(canvas, (recording[i,0]-10, recording[i,1]), (recording[i,0]+10, recording[i,1]), (0,255,0), 2)
#     cv2.line(canvas, (recording[i,2]-10, recording[i,3]), (recording[i,2]+10, recording[i,3]), (0,0,255), 2)
#     for j in range(4, recording.shape[1], 2):
#         colorshift = j*50 % 255
#         cv2.circle(canvas, (recording[i,j], recording[i,j+1]), 10, (colorshift,255-colorshift,colorshift), 2)
#     cv2.imshow('PlayPattern', canvas)
#     cv2.waitKey(60)
#
#
#
#
# cv2.waitKey()
# print(pdl.valy[sample])
