from tensorflow import keras


class LambdaLayerError(Exception):
    """Error with the presence of a keras `Lambda` layer.

    :param lambda_layer: the Keras lambda layer object
    """
    def __init__(self, lambda_layer: keras.layers.Lambda):

        self.lambda_layer: keras.layers.Lambda = lambda_layer
        error_msg = f"the lambda layer is: {str(lambda_layer)}"

        super().__init__(error_msg)
