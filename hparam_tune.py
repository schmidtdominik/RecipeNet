import random

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import numpy as np
import datetime
import os
from util import Recipes

r = Recipes()

HP_BS = hp.HParam('batch_size', hp.Discrete([64, 512]))
HP_LAYER_CONFIG = hp.HParam('layer_config', hp.Discrete([0, 1, 2, 3]))
HP_ACTIVATION_F = hp.HParam('activation_function', hp.Discrete(['relu', 'leaky_relu']))
HP_USE_BN = hp.HParam('use_batch_norm', hp.Discrete(['with_bn', 'without_bn']))
HP_USE_DROPOUT = hp.HParam('use_dropout', hp.Discrete(['with_dropout', 'without_dropout']))
METRIC_ACCURACY = 'hparam_categorical_accuracy'


with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_BS, HP_LAYER_CONFIG, HP_ACTIVATION_F, HP_USE_BN, HP_USE_DROPOUT],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='hparam_categorical_accuracy')],
    )

def train_test_model(hparams, logdir):
    epochs = 6
    valid_batch_size = 512

    layer_config = hparams[HP_LAYER_CONFIG]
    use_batch_norm = hparams[HP_USE_BN] == 'with_bn'
    actf = hparams[HP_ACTIVATION_F]
    use_do = hparams[HP_USE_DROPOUT]

    batch_size = hparams[HP_BS]

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(len(r.ingredients)))
    if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

    if layer_config == 0:
        model.add(tf.keras.layers.Dense(1500))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        if use_do: model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(1000))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dense(1500))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

    elif layer_config == 1:
        model.add(tf.keras.layers.Dense(1500))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        if use_do: model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(1250))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        if use_do: model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(1250))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dense(1500))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

    elif layer_config == 2:
        model.add(tf.keras.layers.Dense(800))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        if use_do: model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(600))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dense(500))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        if use_do: model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(500))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dense(500))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        if use_do: model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(600))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dense(800))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

    elif layer_config == 3:
        model.add(tf.keras.layers.Dense(1800))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        if use_do: model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(1400))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dense(1400))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        if use_do: model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(1400))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dense(1800))
        if use_batch_norm: model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU() if actf == 'leaky_relu' else tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Dense(len(r.ingredients), activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])


    #logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0, profile_batch=0)
    history = model.fit_generator(r.batch_iterator(r.train_recipes, batch_size=batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=r.get_batch_count(r.train_recipes, batch_size),
                                  callbacks=[tensorboard_callback,
                                             hp.KerasCallback(logdir, hparams)])
    _, accuracy = model.evaluate_generator(r.batch_iterator(r.test_recipes, batch_size=valid_batch_size),
                                           steps=r.get_batch_count(r.test_recipes, valid_batch_size))
    return accuracy
session_num = 0

domains = [HP_BS.domain.values, HP_LAYER_CONFIG.domain.values, HP_USE_BN.domain.values, HP_USE_DROPOUT.domain.values, HP_ACTIVATION_F.domain.values]
completed_configs = []

while True:
    config = []
    for d in domains:
        config.append(random.choice(d))
    if config in completed_configs:
        continue

    hparams = {
        HP_BS: config[0],
        HP_LAYER_CONFIG: config[1],
        HP_USE_BN: config[2],
        HP_USE_DROPOUT: config[3],
        HP_ACTIVATION_F: config[4]
    }
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    with tf.summary.create_file_writer('logs/hparam_tuning/' + run_name).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams, 'logs/hparam_tuning/' + run_name)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    print({h.name: hparams[h] for h in hparams}, accuracy)
    session_num += 1