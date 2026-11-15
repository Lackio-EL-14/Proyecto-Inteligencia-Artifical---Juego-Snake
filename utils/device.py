import tensorflow as tf

DEVICE_INITIALIZED_MESSAGE = (
    "No se pudo ocultar la GPU: TensorFlow ya inicializó sus dispositivos. "
    "Llama a use_cpu_only() antes de crear cualquier tensor o modelo."
)


def gpu_is_visible() -> bool:
    return bool(tf.config.get_visible_devices("GPU"))


def hide_gpus() -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise RuntimeError(DEVICE_INITIALIZED_MESSAGE) from error


def use_cpu_only() -> None:
    if gpu_is_visible():
        hide_gpus()
