import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Add, Lambda, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD


def evaluator_net(config):
    """
    This function creates the evaluator network with keras
    :param obs: The number of possible observations in the environment
    :param actions: the number of possible action in the environment
    :return: a model of the neural network
    """
    units = config.units
    obs_env = config.obs_env
    obs_det = config.obs_det
    actions = config.action_size

    inp_env = Input((obs_env,))
    inp_det = Input((obs_det,))
    inp_act = Input((3,))
    inp = concatenate([inp_env, inp_det, inp_act])
    x_skip = Dense(units, activation='relu')(inp)
    x = BatchNormalization()(x_skip)
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_skip])
    x = BatchNormalization()(x)
    x_skip2 = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x_skip2)
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_skip2])
    x = BatchNormalization()(x)
    x_skip3 = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x_skip3)
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_skip3])
    x = BatchNormalization()(x)
    x_skip4 = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x_skip4)
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_skip4])
    x = BatchNormalization()(x)

    x_ska = Dense(units, activation='relu')(x)
    x_a = BatchNormalization()(x_ska)
    x_a = Dense(units, activation='relu')(x_a)
    x_a = BatchNormalization()(x_a)
    x_a = Dense(units, activation='relu')(x_a)
    x_a = BatchNormalization()(x_a)
    x_a = Add()([x_a, x_ska])
    x_a = BatchNormalization()(x_a)
    x_a = Dense(units, activation='relu')(x_a)
    x_a = BatchNormalization()(x_a)

    out_reward = Dense(1, activation="linear")(x_a)

    m = Model([inp_env, inp_det, inp_act], [out_reward])
    m.compile(optimizer=Adam(config.learning_rate, decay=config.lr_decay),
              loss="MSE", metrics=["MAE"])
    return m



