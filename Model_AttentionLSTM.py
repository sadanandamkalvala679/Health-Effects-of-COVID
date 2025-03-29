import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import initializers, regularizers, constraints
from keras.layers import *
from keras.models import Model
from keras.preprocessing.text import Tokenizer

from Evaluate_Error import evaluat_error


class Attention(Layer):
    """
    Keras Layer that implements an Attention mechanism for temporal data.
    Supports Masking.
    Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(Attention())
    """

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def build_model(maxlen, vocab_size, embedding_size, embedding_matrix, target_count):
    input_words = Input((maxlen,))
    x_words = Embedding(vocab_size,
                        embedding_size,
                        weights=[embedding_matrix],
                        mask_zero=True,
                        trainable=False)(input_words)
    x_words = SpatialDropout1D(0.3)(x_words)
    x_words = Bidirectional(LSTM(50, return_sequences=True))(x_words)
    x = Attention(maxlen)(x_words)
    x = Dropout(0.2)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.2)(x)
    pred = Dense(target_count, activation='softmax')(x)

    model = Model(inputs=input_words, outputs=pred)
    return model


def filter_embeddings(embeddings, word_index, vocab_size, dim=300):
    embedding_matrix = np.zeros([vocab_size, dim])
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        vector = embeddings.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix


def load_embeddings(filename):
    embeddings = {}
    with open(filename) as f:
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def Model_AttentionLSTM(trainData, trainTarget, testData, testTarget):
    seed = 7
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)
    maxlen = len(max((s for s in np.r_[trainData, testData]), key=len))
    embedding_size = 200
    tokenizer = Tokenizer(lower=True, filters='\n\t')
    tokenizer.fit_on_texts(np.append(trainData, testData, axis=0))
    trainData = tokenizer.texts_to_sequences(trainData)
    testData = tokenizer.texts_to_sequences(testData)
    vocab_size = len(tokenizer.word_index) + 1  # +1 is for zero padding.
    EMBEDDING_FILE = './GLOVE/glove.6B.200d.txt'

    embeddings = load_embeddings(EMBEDDING_FILE)
    embedding_matrix = filter_embeddings(embeddings, tokenizer.word_index,
                                         vocab_size, embedding_size)
    model = build_model(maxlen, vocab_size, embedding_size, embedding_matrix, trainTarget.shape[1])
    model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    save_file = 'model.h5'
    history = model.fit(trainData, trainTarget,
                        epochs=15, verbose=1,
                        batch_size=1024, shuffle=True)

    y_pred = model.predict(testData, batch_size=1024)
    y_pred = y_pred.argmax(axis=1).astype(int)
    Eval = evaluat_error(y_pred, testTarget)
    return Eval, y_pred
