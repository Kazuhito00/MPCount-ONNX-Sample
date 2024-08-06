#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default='model/MPCount_qnrf.onnx',
    )

    args = parser.parse_args()

    return args


def get_padding(h, w, new_h, new_w):
    if h >= new_h:
        top = 0
        bottom = 0
    else:
        dh = new_h - h
        top = dh // 2
        bottom = dh // 2 + dh % 2
        h = new_h
    if w >= new_w:
        left = 0
        right = 0
    else:
        dw = new_w - w
        left = dw // 2
        right = dw // 2 + dw % 2
        w = new_w

    return (left, top, right, bottom), h, w


def run_inference(onnx_session, image, unit_size=16, log_para=1000):
    # Pre process:Paddig
    image_width, image_hight = image.shape[1], image.shape[0]
    new_width = (image_width // unit_size + 1
                 ) * unit_size if image_width % unit_size != 0 else image_width
    new_hight = (image_hight // unit_size + 1
                 ) * unit_size if image_hight % unit_size != 0 else image_hight
    padding, _, _ = get_padding(image_hight, image_width, new_hight, new_width)

    left, top, right, bottom = padding[0], padding[1], padding[2], padding[3]
    pad_value = 0
    padded_image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )

    # Pre process:BGR→RGB
    x = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

    # Pre process：Normalization, Convert BCHW
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    x = ((x / 255) - mean) / std
    x = np.array(x, dtype=np.float32)
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, padded_image.shape[0], padded_image.shape[1])

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result, _ = onnx_session.run(None, {input_name: x})

    # Post process:Delete padding
    result_map = np.array(result)
    result_map = result_map[
        :,
        :,
        top:result_map.shape[2] - bottom,
        left:result_map.shape[3] - right,
    ]

    return result_map, int(np.sum(result_map) / log_para)


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie
    image_path = args.image

    model_path = args.model

    # Initialize video capture
    if image_path is None:
        cap = cv2.VideoCapture(cap_device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    if image_path is not None:
        image = cv2.imread(image_path)
        debug_image = copy.deepcopy(image)

        start_time = time.time()

        # Inference execution
        result_map, peaple_count = run_inference(
            onnx_session,
            image,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            result_map,
            peaple_count,
        )

        cv2.imshow('MPCount Demo : Original Image', image)
        cv2.imshow('MPCount Demo : Activation Map', debug_image)
        cv2.waitKey(0)
    else:
        while True:
            start_time = time.time()

            # Capture read
            ret, frame = cap.read()
            if not ret:
                break
            debug_image = copy.deepcopy(frame)

            # Inference execution
            result_map, peaple_count = run_inference(
                onnx_session,
                frame,
            )

            elapsed_time = time.time() - start_time

            # Draw
            debug_image = draw_debug(
                debug_image,
                elapsed_time,
                result_map,
                peaple_count,
            )

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            cv2.imshow('MPCount Demo : Original Image', frame)
            cv2.imshow('MPCount Demo : Activation Map', debug_image)

        cap.release()
    cv2.destroyAllWindows()


def draw_debug(image, elapsed_time, result_map, peaple_count):
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(result_map[0, 0])

    # Apply ColorMap
    debug_image = (debug_image - debug_image.min()) / (
        debug_image.max() - debug_image.min() + 1e-5)
    debug_image = (debug_image * 255).astype(np.uint8)
    debug_image = cv2.applyColorMap(debug_image, cv2.COLORMAP_JET)

    debug_image = cv2.resize(debug_image, dsize=(image_width, image_height))

    # addWeighted
    debug_image = cv2.addWeighted(image, 0.35, debug_image, 0.65, 1.0)

    # Inference elapsed time
    cv2.putText(
        debug_image,
        "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    # Peaple Count
    cv2.putText(debug_image, "People Count : " + str(peaple_count), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()
