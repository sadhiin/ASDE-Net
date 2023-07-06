from tensorflow import keras


def calc_IOU(y_true, y_pred, smooth=1):
    y_true_f = keras.layers.Flatten()(y_true)
    y_pred_f = keras.layers.Flatten()(y_pred)

    intersection = keras.backend.sum(y_true_f * y_pred_f)

    return (2 * (intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth))


def calc_IOU_loss(y_true, y_pred):
    return -calc_IOU(y_true, y_pred)
