from typing import List
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from onnx2keras.utils import count_params_keras


def _create_multi_output_model(
    model: tf.keras.Model, output_layers_names: List[str]
):
    """Make a keras model a multi-output model."""
    new_outputs = [
        model.get_layer(layer_name).output for layer_name in output_layers_names
    ]
    return tf.keras.Model(name=model.name, inputs=model.inputs, outputs=new_outputs)


def main(in_file: Path, out_file: Path, fpn_layers_names: List[str]=None):

    tf.keras.backend.clear_session()

    model = tf.keras.models.load_model(in_file, compile=False)

    n_params = count_params_keras(model)
    n_trainable_params = count_params_keras(model, trainable_only=True)

    print("Loaded model named '{}'; param count: {}, trainable param count: {}".format(
        model.name,
        n_params,
        n_trainable_params,
        )
    )

    if fpn_layers_names is None:
        # assumes model is ResNet50 and get the activations at the end of each block.
        #@fixme: make the `fpn_layers_names` argument non-default
        raise ValueError("missing value for `fpn_layer_names`")
        fpn_layers_names = ["activation_9", "activation_21", "activation_39", "activation_48"]

    mo_model = _create_multi_output_model(model, fpn_layers_names)

    # save the converted keras model to output file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    mo_model.save(
        out_file,
        include_optimizer=False,
        save_format="tf",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--fpn-layers-names", nargs="+", default=None)
    args = parser.parse_args()

    main(
        args.in_file,
        args.out_file,
        args.fpn_layers_names,
    )

