"""
The MLP model's wrapper.
"""
import tensorflow as tf
import numpy as np

tf.config.experimental.set_visible_devices([], 'GPU')
MODEL = tf.keras.models.load_model("model")


LMBDA = 1/10
def transform(ys: float)->float:
    """pre processing function

    Parameters
    ----------
    ys : float
        input number, can be a list

    Returns
    -------
    float
        returned number, can be a list
    """
    return (ys**LMBDA-1)/LMBDA

def inverse_transform(ys: float)->float:
    """inverse function of the `transfom`

    Parameters
    ----------
    ys : float
        input number, can be a list

    Returns
    -------
    float
        returned number, can be a list
    """
    return (ys*LMBDA+1)**(1/LMBDA)

def get_emis(xs: np.ndarray)->np.ndarray:
    """
    Return the predicted emissivity from the MLP models

    Parameters
    ----------
    xs : np.ndarray
        The size is [n, 6]. The columns stand for: p, T, pL, x_h2o, x_co2, x_co.
        p: bar
        T: K
        pL: bar*cm

    Returns
    -------
    np.ndarray
        Predicted emissivity.
    """
    xs[:, 2] = np.log(xs[:, 2])
    xs[:, 3:6] = transform(xs[:, 3:6])
    return MODEL.predict(xs)
