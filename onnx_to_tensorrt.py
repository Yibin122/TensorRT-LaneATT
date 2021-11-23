import os
import cv2
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from nms import nms
import common

TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 30  # 1GB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run laneatt_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 360, 640]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # FIXME: getPluginCreator could not find plugin: ScatterND version: 1
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def do_nms(proposals, conf_threshold=0.4, nms_thres=50., nms_topk=4):
    proposals = torch.from_numpy(proposals).cuda()
    scores = proposals[:, 1]
    # apply confidence threshold
    above_threshold = scores > conf_threshold
    proposals = proposals[above_threshold]
    scores = scores[above_threshold]
    # cuda implementation
    keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
    keep = keep[:num_to_keep]
    proposals = proposals[keep]
    return proposals


def post_process(img, proposals, n_offsets=72):
    # proposals_to_pred
    n_strips = n_offsets - 1
    anchor_ys = torch.linspace(1, 0, steps=n_offsets, dtype=torch.float32, device='cuda:0')
    anchor_ys = anchor_ys.double()
    lanes = []
    for lane in proposals:
        lane_xs = lane[5:] / 640
        start = int(round(lane[2].item() * n_strips))
        length = int(round(lane[4].item()))
        end = start + length - 1
        end = min(end, len(anchor_ys) - 1)
        # end = label_end
        # if the proposal does not start at the bottom of the image,
        # extend its proposal until the x is outside the image
        mask = ~((((lane_xs[:start] >= 0.) &
                   (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
        lane_xs[end + 1:] = -2
        lane_xs[:start][mask] = -2
        lane_ys = anchor_ys[lane_xs >= 0]
        lane_xs = lane_xs[lane_xs >= 0]
        lane_xs = lane_xs.flip(0).double()
        lane_ys = lane_ys.flip(0)
        if len(lane_xs) <= 1:
            continue
        points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
        lanes.append(points.cpu().numpy())
    # print('Number of lanes: {}'.format(len(lanes)))

    # Visualize
    img_h, img_w = img.shape[:2]
    for lane_points in lanes:
        # scale back to input size
        lane_points[:, 0] *= img_w
        lane_points[:, 1] *= img_h
        lane_points = lane_points.round().astype(int)
        for point in lane_points:
            cv2.circle(img, tuple(point), 2, (0, 255, 0), -1)
    cv2.imshow('LaneATT_tensorrt', img)
    cv2.waitKey(0)


def engine_inference(onnx_file_path, image_file_path):
    """Create a TensorRT engine for ONNX-based LaneATT and run inference."""

    engine_file_path = onnx_file_path.split('.onnx')[0] + '.trt8'

    # Load test image
    image_raw = cv2.imread(image_file_path)
    image = cv2.resize(image_raw, (640, 360), cv2.INTER_LINEAR)
    # normalize and flatten
    image = image.astype(np.float32) / 255.0
    image = image.transpose([2, 0, 1]).flatten()

    # Do inference with TensorRT
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        print('\nRunning inference on image {}...'.format(image_file_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    output = trt_outputs[0].reshape([1000, 77])

    proposals = do_nms(output)
    post_process(image_raw, proposals)


if __name__ == '__main__':
    image_file = './02610.jpg'
    onnx_file = './LaneATT_test.onnx'
    engine_inference(onnx_file, image_file)
