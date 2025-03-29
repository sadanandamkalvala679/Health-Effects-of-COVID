import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras import backend as K


def Model_1DCNN(trainX, trainy, testX, testy, epoches=10, batchSize=32, dropout=0.5, hN=100):
    verbose, epochs, batch_size = 0, epoches, batchSize
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(hN, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    pred = model.predict(testX)

    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layerNo = 4
    data = np.append(trainX, testX, axis=0)
    Feats = []
    for i in range(data.shape[0]):
        test = data[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()  # [func([test]) for func in functors]
        Feats.append(layer_out)
    Feats = np.asarray(Feats)
    return Feats, pred


if __name__ == '__main__':
    a = np.random.random((200, 100, 3))
    b = np.random.randint(0, 2, (200, 1))
    c = np.random.random((200, 100, 3))
    d = np.random.randint(0, 2, (200, 1))
    Feats, pred = Model_1DCNN(a, b, c, d)
    corr = np.mean(np.corrcoef(Feats))
    breakp = 5
