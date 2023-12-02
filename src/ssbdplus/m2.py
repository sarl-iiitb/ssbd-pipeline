import tensorflow as tf

M2_PATH = "path/to/m2"

"""
Creates an instance of the SSBD Identifier model
"""
def create_ssbd_model2():
    feature_extractor = tf.keras.applications.Xception(
        weights = 'imagenet',
        input_shape = (256, 341, 3),
        include_top = False,
        pooling = 'avg'
    )

    feature_extractor.trainable = False

    frame_input = tf.keras.Input(shape = (256, 341, 3))
    joint_input = tf.keras.Input(shape = (40, 34,))

    spatial_features = feature_extractor(frame_input, training = False)

    temporal_features = tf.keras.layers.LSTM(units = 4)(joint_input)
    combined = tf.keras.layers.Concatenate()([spatial_features, temporal_features])

    pred_action = tf.keras.layers.UnitNormalization()(combined)
    pred_action = tf.keras.layers.GaussianDropout(rate = 0.7, seed = 42)(pred_action)
    pred_action = tf.keras.layers.Dense(3)(pred_action)
    pred_action = tf.keras.layers.Softmax()(pred_action)

    model = tf.keras.Model(inputs = [frame_input, joint_input], outputs = pred_action)

    return model

def load_ssbd_model2():
    model = create_model()
    model.load_weights(checkpoint_path)

    return model
