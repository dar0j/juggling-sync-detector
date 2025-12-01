import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Flatten, LeakyReLU
from keras import regularizers, metrics
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from patterndataloader import PatternDataLoader
from keras.callbacks import ModelCheckpoint

#tf.random.set_seed(42)
np.random.seed(42)

l = 30 # length of window (that is gonna correlate with the fps since at that speed is going to be predicting)

DATASETS = {
    3: {"file": "3balls", "length": l},
    4: {"file": "4balls", "length": l},
    5: {"file": "5balls", "length": l},
    6: {"file": "6balls", "length": l},
}

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    for _ in range(3):
        model.add(Dense(60, kernel_regularizer=regularizers.l2(1e-4)))
        model.add(LeakyReLU())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=[metrics.categorical_accuracy]
    )
    return model

def run_cross_validation():
    for balls, config in DATASETS.items():
        loader = PatternDataLoader(config["file"], length=config["length"])
        names = loader.getNames()
        num_classes = len(names)
        X, y = loader.get_cross_validation_data(shuffle_balls=True)
        if not len(X):
            print(f"{balls} balls: no samples available.")
            continue

        #kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        from collections import Counter
        print(Counter(y))
        # for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
        #     X_train, X_val = X[train_idx], X[val_idx]
        #     y_train, y_val = y[train_idx], y[val_idx]

        #     y_train_cat = to_categorical(y_train, num_classes=num_classes)
        #     y_val_cat = to_categorical(y_val, num_classes=num_classes)

        #     model = build_model(input_shape=X.shape[1:], num_classes=num_classes)
        #     callbacks = [
        #         EarlyStopping(
        #             monitor="val_categorical_accuracy",
        #             patience=5,
        #             restore_best_weights=True
        #         ),
        #         ModelCheckpoint(
        #             filepath=f"best_model_{balls}balls_{l}_128_fold{fold_idx}.h5",
        #             monitor="val_categorical_accuracy",
        #             save_best_only=True
        #         )
        #     ]

        #     model.fit(
        #         X_train,
        #         y_train_cat,
        #         validation_data=(X_val, y_val_cat),
        #         epochs=40,
        #         batch_size=32,
        #         verbose=0,
        #         callbacks=callbacks
        #     )

        #     metrics_values = model.evaluate(X_val, y_val_cat, verbose=0)
        #     acc = metrics_values[1] * 100.0
        #     fold_scores.append(acc)
        #     print(f"{balls} balls | fold {fold_idx}: accuracy = {acc:.2f}%")

        # mean_acc = np.mean(fold_scores)
        # std_acc = np.std(fold_scores)
        # print(f"{balls} balls | mean accuracy = {mean_acc:.2f}% ± {std_acc:.2f}%\n")

if __name__ == "__main__":
    run_cross_validation()



# from patterndataloader import PatternDataLoader
# from sklearn.model_selection import KFold
# from keras.utils import to_categorical
# import numpy as np
# import cv2

# # Parámetros
# FOLDS = 5
# EPOCHS = 20
# BATCH = 32

# pdl = PatternDataLoader("4balls", length=30)  # cambiar a "4balls"/"5balls"/"6balls" según tus ficheros
# names = pdl.getNames()
# num_classes = len(names)

# # Obtener datos brutos (X: (N,length,dim), y: int labels)
# X, y_int = pdl.getRaw()

# # normalizar/reshape si es necesario para la red (ya lo hace _annotationsToX)
# y_onehot = to_categorical(y_int, num_classes=num_classes)

# kf = KFold(n_splits=FOLDS, shuffle=True, random_state=0)
# fold = 0
# scores = []
# for train_idx, val_idx in kf.split(X):
#     fold += 1
#     print(f"Fold {fold}/{FOLDS}")
#     trainx, trainy = X[train_idx], y_onehot[train_idx]
#     valx, valy = X[val_idx], y_onehot[val_idx]

#     # construir modelo (mismo esqueleto que ya usabas; ajustar input_shape y salida)
#     model = Sequential()
#     model.add(Flatten(input_shape=trainx.shape[1:3]))
#     model.add(Dense(units=60, kernel_regularizer=regularizers.l2(0.0001)))
#     model.add(LeakyReLU())
#     model.add(Dense(units=60, kernel_regularizer=regularizers.l2(0.0001)))
#     model.add(LeakyReLU())
#     model.add(Dense(units=60, kernel_regularizer=regularizers.l2(0.0001)))
#     model.add(LeakyReLU())
#     model.add(Dense(units=num_classes, activation='softmax'))

#     model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=[metrics.categorical_accuracy])

#     model.fit(x=trainx, y=trainy, batch_size=BATCH, epochs=EPOCHS, validation_data=(valx, valy))
#     score = model.evaluate(valx, valy)
#     print("Fold score:", score)
#     scores.append(score[1])  # accuracy

# print("CV mean accuracy: %.3f ± %.3f" % (np.mean(scores), np.std(scores)))

# model = load_model(saveModelFilename)

# metrics = model.evaluate(pdl.testx, pdl.testy)
# print("Testset Accuracy: %.1f" % (metrics[1]*100))
# pred = model.predict(pdl.testx)
# pred = np.argmax(pred, axis=1)
# testy  = np.argmax(pdl.testy, axis=1)

# names = pdl.getNames()
# testy = [names[i] for i in testy]
# pred = [names[i] for i in pred]
# skplt.metrics.plot_confusion_matrix(testy, pred, x_tick_rotation=45, title=" ", text_fontsize="large")
# plt.show()



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
