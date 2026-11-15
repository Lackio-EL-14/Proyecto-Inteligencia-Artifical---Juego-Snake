import tensorflow as tf

HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "linear"


def build_q_network(state_size: int, action_size: int, hidden_units: tuple[int, ...]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(state_size,))
    features = inputs
    for units in hidden_units:
        features = tf.keras.layers.Dense(units, activation=HIDDEN_ACTIVATION)(features)
    outputs = tf.keras.layers.Dense(action_size, activation=OUTPUT_ACTIVATION)(features)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_optimizer(learning_rate: float, clipnorm: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)


def copy_weights(source: tf.keras.Model, target: tf.keras.Model) -> None:
    for source_variable, target_variable in zip(source.weights, target.weights, strict=True):
        target_variable.assign(source_variable)
