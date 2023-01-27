from typing import List

import argparse
import io
from pathlib import Path
import re


import onnx
import torch
from torch import nn
import torchvision
import tensorflow as tf
import numpy as np


from onnx2keras import onnx_to_keras, check_torch_keras_error
from onnx2keras.utils import count_params_torch, count_params_keras

# utilities to extract the statedict of the backbone
def _unwrap_state_dict(state_dict: dict) -> dict:
    """
    If the state dict was saved from a module wrapped in *DataParallel, remove the prefix
    'module.' from all keys in the state dict, and do not change the key names otherwise

    Args:
       state_dict (dict): State dict from module

    Returns:
        dict: state dict with updated keys
    """
    return {re.sub(r"^module\.", "", key): value for key, value in state_dict.items()}


def _get_feature_extractor_state_dict(
    state_dict: dict, output: str = "encoder"
) -> dict:
    """Extract the ResNet encoder state dict from the loaded state dict.

    Args:
       state_dict (dict): State dict from module, can be a MoCo state dict or regular torchvision module state dict
        output (str): Code to select output of feature extractor. If "encoder", will discard the fully-connected (fc)
            layer(s). Else, it will keep the fc layer(s)

    Returns:
        dict: the feature extractor's state dict

    Raises:
        ValueError
    """
    # @fixme: do not check if is MoCo format inside function, check outside
    def _is_moco(state_dict: dict) -> bool:
        """Returns whether the state dict corresponds to a MoCo model or not."""
        for key in state_dict.keys():
            if re.match(r"^encoder_q\.", key):
                return True
        return False

    if _is_moco(state_dict):
        # get keys for the encoder to extract, remove prefix
        def _filter_fn(key):
            if output == "encoder":
                return re.match(r"^encoder_q\.", key) and not re.search(r"\.fc\.", key)
            else:
                return re.match(r"^encoder_q\.", key)

        state_dict = {
            re.sub(r"^encoder_q\.", "", key): value
            for key, value in state_dict.items()
            if _filter_fn(key)
        }
    else:
        if output != "encoder":
            raise ValueError(
                f"state dict for regular torchvision module does not support output value '{output}', only 'encoder'"
            )

        def _filter_fn(key):
            return not re.search(r"^fc\.", key)

        state_dict = {
            key: value for key, value in state_dict.items() if _filter_fn(key)
        }

    return state_dict


def main(
    in_file,
    out_file,
    intermediate_onnx_file=None,
    arch="resnet50",
    output_layers_names=None,
    input_shape=None,
):
    tf.keras.backend.clear_session()

    checkpoint = torch.load(in_file, map_location="cpu")

    arch = checkpoint.get("arch", arch)
    if not arch.startswith("resnet"):
        raise NotImplementedError("Only ResNet architectures are supported")

    print("Loaded PyTorch checkpoint (arch = {})".format(arch))

    # get the ResNet encoder state dict
    state_dict = _get_feature_extractor_state_dict(
        _unwrap_state_dict(checkpoint["state_dict"])
    )

    # instanciate model and discard fc layer
    # only ResNet architectures are supported
    model = torchvision.models.__dict__[arch]()
    model.fc = nn.Identity()

    model.load_state_dict(state_dict)

    print(
        "Created PyTorch model '{}'; param count: {}, trainable param count: {}".format(
            checkpoint["arch"],
            count_params_torch(model),
            count_params_torch(model, trainable_only=True),
        )
    )

    model.train()

    # create placeholder input and outputs tensors
    tensor_input_shape = (224, 224) if input_shape is None else input_shape

    input_np = np.random.randn(1, 3, *tensor_input_shape)
    input_t = torch.from_numpy(input_np).float()

    # export PyTorch model to ONNX
    input_names = ["input_1"]
    output_names = ["output_1"]

    if intermediate_onnx_file is not None:
        file = intermediate_onnx_file
        file.parent.mkdir(parents=True, exist_ok=True)
    else:
        file = io.BytesIO()

    torch.onnx.export(
        model,
        input_t,
        file,
        verbose=True,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        do_constant_folding=False,
        dynamic_axes={
            "input_1": {0: "batch_size"},  # variable batch axis
            "output_1": {0: "batch_size"},  # variable batch axis
        },
        training=torch.onnx.TrainingMode.TRAINING,
    )
    if isinstance(file, io.BytesIO):
        file.seek(0)

    onnx_model = onnx.load(file)

    # check the ONNX model
    onnx.checker.check_model(onnx_model)

    # export ONNX model to TF Keras
    if input_shape is None:
        input_shapes = [(3, None, None)]
    else:
        input_shapes = [(3, *input_shape)]

    k_model = onnx_to_keras(
        onnx_model,
        input_names=input_names,
        input_shapes=input_shapes,
        name_policy="keras",
        verbose=True,
        change_ordering=True,
    )

    # assert outputs are all close up to an absolute tolerance or 1e-04
    # note: max absolute difference sould be around 5e-05
    model.eval()

    error = check_torch_keras_error(
        model,
        k_model,
        input_np,
        epsilon=5e-4,
        change_ordering=True,
    )
    print(f"Outputs max error: {error}")

    # verify the number of trainable parameters is the same
    # Note: we don't verify that the number of total parameters is the same
    # because it is possible that the total nuber of parametres is different:
    # for particular layers such as BatchNorm, some variables are not considered parameters
    # in PyTorch ('buffers', instead), while they are in Keras.

    n_trainable_params = count_params_torch(model, trainable_only=True)
    k_n_trainable_params = count_params_keras(k_model, trainable_only=True)

    assert n_trainable_params == k_n_trainable_params

    def _create_multi_output_model(
        model: tf.keras.Model, output_layers_names: List[str]
    ):
        """Make a keras model a multi-output model."""
        new_outputs = [
            model.get_layer(layer_name).output for layer_name in output_layers_names
        ]
        return tf.keras.Model(inputs=model.inputs, outputs=new_outputs)

    if output_layers_names is not None:
        k_model = _create_multi_output_model(k_model, output_layers_names)

    k_model.summary()

    # save the converted keras model to output file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    k_model.save(
        out_file,
        include_optimizer=False,
        save_format="tf",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--intermediate-onnx-file", type=Path, default=None)
    parser.add_argument("--arch", type=str, default="resnet50")  #@fixme: not default for arch, should be in checkpoint
    parser.add_argument("--output-layers-names", nargs="+", default=None)
    parser.add_argument("--input-shape", nargs=2, type=int, default=None)
    args = parser.parse_args()

    main(
        args.in_file,
        args.out_file,
        args.intermediate_onnx_file,
        args.arch,
        args.output_layers_names,
        args.input_shape,
    )
