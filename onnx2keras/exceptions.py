from tensorflow import keras


class LambdaLayerError(Exception):
    """Error with the presence of a keras `Lambda` layer.

    :param lambda_layer: the Keras lambda layer object
    """
    def __init__(self, lambda_layer: keras.layers.Lambda):

        self.lambda_layer: keras.layers.Lambda = lambda_layer
        error_msg = f"{self._lambda_layer_as_str(lambda_layer)}"

        super().__init__(error_msg)

    @staticmethod
    def _lambda_layer_as_str(layer: keras.layers.Lambda) -> str:
        return (
            f"{layer.__class__.__name__}("
            f"name={layer.name}, "
            f"function={layer.function}, "
            f"output_shape={layer.output_shape}, "
            f"mask={layer.mask}, "
            f"arguments={layer.arguments}"
            ")"
        )
