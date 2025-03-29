#!/usr/bin/env python3

"""
Python 3 wrapper for identifying objects in images with quantization support.

This version applies uniform quantization to all layers, reducing precision of
weights and activations to the specified bit-width.
"""

import ctypes as ct
import random
import os
import cv2
import numpy as np


class BOX(ct.Structure):
    _fields_ = (
        ("x", ct.c_float),
        ("y", ct.c_float),
        ("w", ct.c_float),
        ("h", ct.c_float),
    )


FloatPtr = ct.POINTER(ct.c_float)
IntPtr = ct.POINTER(ct.c_int)


class DETECTION(ct.Structure):
    _fields_ = (
        ("bbox", BOX),
        ("classes", ct.c_int),
        ("best_class_idx", ct.c_int),
        ("prob", FloatPtr),
        ("mask", FloatPtr),
        ("objectness", ct.c_float),
        ("sort_class", ct.c_int),
        ("uc", FloatPtr),
        ("points", ct.c_int),
        ("embeddings", FloatPtr),
        ("embedding_size", ct.c_int),
        ("sim", ct.c_float),
        ("track_id", ct.c_int),
    )


DETECTIONPtr = ct.POINTER(DETECTION)


class DETNUMPAIR(ct.Structure):
    _fields_ = (
        ("num", ct.c_int),
        ("dets", DETECTIONPtr),
    )


DETNUMPAIRPtr = ct.POINTER(DETNUMPAIR)


class IMAGE(ct.Structure):
    _fields_ = (
        ("w", ct.c_int),
        ("h", ct.c_int),
        ("c", ct.c_int),
        ("data", FloatPtr),
    )


class METADATA(ct.Structure):
    _fields_ = (
        ("classes", ct.c_int),
        ("names", ct.POINTER(ct.c_char_p)),
    )


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = round(x - (w / 2))
    xmax = round(x + (w / 2))
    ymin = round(y - (h / 2))
    ymax = round(y + (h / 2))
    return xmin, ymin, xmax, ymax


def class_colors(names):
    """
    Create a dict with one random BGR color for each class name.
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def uniform_quantization(tensor, bit_width):
    """
    Applies uniform quantization to a given tensor.
    Args:
        tensor (np.ndarray): Input array to be quantized.
        bit_width (int): Bit width for quantization (e.g., 8 for 8-bit quantization).
    Returns:
        quantized_tensor (np.ndarray): Quantized tensor.
        scale (float): Scale factor used for quantization.
    """
    qmin = -(2**(bit_width - 1))
    qmax = (2**(bit_width - 1)) - 1
    scale = (tensor.max() - tensor.min()) / (qmax - qmin)
    zero_point = qmin - np.round(tensor.min() / scale)
    quantized_tensor = np.round(tensor / scale) + zero_point
    quantized_tensor = np.clip(quantized_tensor, qmin, qmax)
    return quantized_tensor.astype(np.int32), scale


def dequantize(quantized_tensor, scale):
    """
    Dequantizes a quantized tensor using the scale factor.
    """
    return quantized_tensor.astype(np.float32) * scale


def load_network(config_file, data_file, weights, bit_width=8, batch_size=1):
    """
    Load network with uniform quantization applied to all layers.
    """
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)

    # Apply uniform quantization to all layers in the network
    for layer in network.layers:
        if hasattr(layer, 'weights'):
            quantized_weights, scale = uniform_quantization(layer.weights, bit_width)
            layer.weights = quantized_weights
            layer.weight_scale = scale
        if hasattr(layer, 'bias'):
            quantized_bias, scale = uniform_quantization(layer.bias, bit_width)
            layer.bias = quantized_bias
            layer.bias_scale = scale

    return network, class_names, colors


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))


def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
    Perform detection with quantized weights and activations.
    """
    width = network_width(network)
    height = network_height(network)
    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    darknet_image = make_image(width, height, 3)
    copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = predict_image(network, darknet_image)
    free_image(darknet_image)

    return detections


if os.name == "posix":
    cwd = os.path.dirname(__file__)
    print(cwd)
    lib = ct.CDLL(cwd + "/libdarknet.so", ct.RTLD_GLOBAL)
elif os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ["PATH"] = os.path.pathsep.join((cwd, os.environ["PATH"]))
    lib = ct.CDLL("darknet.dll", winmode=0, mode=ct.RTLD_GLOBAL)
else:
    lib = None  # Intellisense
    print("Unsupported OS")
    exit()

lib.network_width.argtypes = (ct.c_void_p,)
lib.network_width.restype = ct.c_int
lib.network_height.argtypes = (ct.c_void_p,)
lib.network_height.restype = ct.c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = (IMAGE, ct.c_char_p)

predict_image = lib.network_predict_image
predict_image.argtypes = (ct.c_void_p, IMAGE)
predict_image.restype = FloatPtr

