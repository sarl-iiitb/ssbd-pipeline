import tensorflow as tf
from frame_selector import get_movenet_data, frame_with_max_change

M2_PATH = "mdl_wts.hdf5"

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
    model = create_ssbd_model2()
    model.load_weights(M2_PATH)

    return model

def train_model(model, train_set, test_set):
    [train_frames, train_keypts], train_y = train_set
    [test_frames, test_keypts], test_y = test_set
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1, amsgrad = True),
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False),
                metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    mcp_save = tf.keras.callbacks.ModelCheckpoint(M2_PATH, save_best_only=True, monitor='val_categorical_accuracy', mode='max')

    model.fit(x = [train_frames, train_keypts], y = train_y,
                epochs = 100, validation_data = ([test_frames, test_keypts], test_y),
                callbacks=[mcp_save], batch_size = 64, shuffle = True)


def m2_identify(model, video_path):
    keypts, frames = get_movenet_data(video_path)
    _, max_loc = frame_with_max_change(keypts)
    best_frame = frames[max_loc + 1]
    return model.predict([best_frame, keypts])

