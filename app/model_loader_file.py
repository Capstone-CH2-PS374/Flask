from keras.models import load_model


def load_model_h5():
    # Ganti dengan path model TensorFlow Anda
    model_path = '../model/my_model.h5'
    model = load_model(model_path)
    return model
