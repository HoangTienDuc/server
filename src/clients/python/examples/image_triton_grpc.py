import argparse
import os
import struct
import sys
import time

import grpc
import numpy as np
import tritonclient.grpc.model_config_pb2 as mc
from PIL import Image
from tritonclient.grpc import service_pb2, service_pb2_grpc


class TritonModelSpecs:
    def __init__(self, inception_config):
        self.model_name = inception_config["model_name"]
        self.model_version = inception_config["model_version"]
        channel = grpc.insecure_channel(inception_config["url"])
        self.is_streaming = inception_config["streaming"]
        self.is_async_set = inception_config["async_set"]
        self.batch_size = inception_config["batch_size"]
        self.grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
        self.classes = 1
        self.triton_client = None

    def parse_model(self, model_metadata, model_config):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(model_metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(model_metadata.inputs)))
        if len(model_metadata.outputs) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(model_metadata.outputs)))

        if len(model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config.input)))

        input_metadata = model_metadata.inputs[0]
        input_config = model_config.input[0]
        output_metadata = model_metadata.outputs[0]

        if output_metadata.datatype != "FP32":
            raise Exception("expecting output datatype to be FP32, model '" +
                            model_metadata.name + "' output type is " +
                            output_metadata.datatype)

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = (model_config.max_batch_size > 0)
        non_one_cnt = 0
        for dim in output_metadata.shape:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = (model_config.max_batch_size > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                format(expected_input_dims, model_metadata.name,
                       len(input_metadata.shape)))

        if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
                (input_config.format != mc.ModelInput.FORMAT_NHWC)):
            raise Exception("unexpected input format " +
                            mc.ModelInput.Format.Name(input_config.format) +
                            ", expecting " +
                            mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                            " or " +
                            mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

        if input_config.format == mc.ModelInput.FORMAT_NHWC:
            h = input_metadata.shape[1 if input_batch_dim else 0]
            w = input_metadata.shape[2 if input_batch_dim else 1]
            c = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            c = input_metadata.shape[1 if input_batch_dim else 0]
            h = input_metadata.shape[2 if input_batch_dim else 1]
            w = input_metadata.shape[3 if input_batch_dim else 2]

        return (input_metadata.name, output_metadata.name, c, h, w,
                input_config.format, input_metadata.datatype)

    def get_model_specs(self):
        metadata_request = service_pb2.ModelMetadataRequest(
            name=self.model_name, version=self.model_version)
        metadata_response = self.grpc_stub.ModelMetadata(metadata_request)

        config_request = service_pb2.ModelConfigRequest(name=self.model_name,
                                                        version=self.model_version)
        config_response = self.grpc_stub.ModelConfig(config_request)

        self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype = self.parse_model(
            metadata_response, config_response.config)


class TritonIS(TritonModelSpecs):
    def __init__(self, inception_config):
        super().__init__(inception_config)

    def requestGenerator(self, input_name, output_name, image_data, dtype):
        request = service_pb2.ModelInferRequest()
        request.model_name = self.model_name
        request.model_version = self.model_version

        output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        output.name = output_name
        output.parameters['classification'].int64_param = self.classes
        request.outputs.extend([output])

        input = service_pb2.ModelInferRequest().InferInputTensor()
        input.name = input_name
        input.datatype = dtype
        if self.format == mc.ModelInput.FORMAT_NHWC:
            input.shape.extend([self.batch_size, self.h, self.w, self.c])
        else:
            input.shape.extend([self.batch_size, self.c, self.h, self.w])

        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        image_idx = 0
        last_request = False
        while not last_request:
            input_bytes = None
            input_filenames = []
            request.ClearField("inputs")
            request.ClearField("raw_input_contents")
            for idx in range(self.batch_size):
                if input_bytes is None:
                    input_bytes = image_data[image_idx].tobytes()
                else:
                    input_bytes += image_data[image_idx].tobytes()

                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True

            request.inputs.extend([input])
            request.raw_input_contents.extend([input_bytes])
            yield request

    def deserialize_bytes_tensor(self, encoded_tensor):
        strs = list()
        offset = 0
        val_buf = encoded_tensor
        while offset < len(val_buf):
            l = struct.unpack_from("<I", val_buf, offset)[0]
            offset += 4
            sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
            offset += l
            strs.append(sb)
        return (np.array(strs, dtype=bytes))

    def postprocess(self, response):
        """
        Post-process response to show classifications.
        """
        if self.is_streaming:
            response = response.infer_response
        if len(response.outputs) != 1:
            raise Exception("expected 1 output, got {}".format(len(
                response.outputs)))

        if len(response.raw_output_contents) != 1:
            raise Exception("expected 1 output content, got {}".format(
                len(response.raw_output_contents)))

        batched_result = self.deserialize_bytes_tensor(
            response.raw_output_contents[0])
        contents = np.reshape(batched_result, response.outputs[0].shape)

        if len(contents) != self.batch_size:
            raise Exception("expected {} results, got {}".format(
                self.batch_size, len(contents)))
        return contents

    def execute(self, image_data):
        requests = []
        responses = []
        results = []

        # Send request
        if self.is_streaming:
            for response in self.grpc_stub.ModelStreamInfer(
                    self.requestGenerator(self.input_name, self.output_name, image_data, self.dtype)):
                responses.append(response)
        else:
            for request in self.requestGenerator(self.input_name, self.output_name, image_data, self.dtype):
                if not self.is_async_set:
                    responses.append(self.grpc_stub.ModelInfer(request))
                else:
                    requests.append(self.grpc_stub.ModelInfer.future(request))
        # For async, retrieve results according to the send order
        if self.is_async_set:
            for request in requests:
                responses.append(request.result())
        for response in responses:
            results.append(self.postprocess(response))

        return results


def main():
    inception_config = {
        "model_name": "inception_graphdef",
        "model_version": "",
        "protocol": "GRPC",
        "url": "localhost:8001",
        "verbose": False,
        "streaming": False,
        "async_set": True,
        "batch_size": 1,
        "classes": 1
    }

    def model_dtype_to_np(model_dtype):
        if model_dtype == "BOOL":
            return np.bool
        elif model_dtype == "INT8":
            return np.int8
        elif model_dtype == "INT16":
            return np.int16
        elif model_dtype == "INT32":
            return np.int32
        elif model_dtype == "INT64":
            return np.int64
        elif model_dtype == "UINT8":
            return np.uint8
        elif model_dtype == "UINT16":
            return np.uint16
        elif model_dtype == "FP16":
            return np.float16
        elif model_dtype == "FP32":
            return np.float32
        elif model_dtype == "FP64":
            return np.float64
        elif model_dtype == "BYTES":
            return np.dtype(object)
        return None

    def preprocess(img, format, dtype, c, h, w, scaling):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        """
        # np.set_printoptions(threshold='nan')

        if c == 1:
            sample_img = img.convert('L')
        else:
            sample_img = img.convert('RGB')

        resized_img = sample_img.resize((w, h), Image.BILINEAR)
        resized = np.array(resized_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        npdtype = model_dtype_to_np(dtype)
        typed = resized.astype(npdtype)

        if scaling == 'INCEPTION':
            scaled = (typed / 128) - 1
        elif scaling == 'VGG':
            if c == 1:
                scaled = typed - np.asarray((128,), dtype=npdtype)
            else:
                scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
        else:
            scaled = typed

        # Swap to CHW if necessary
        if format == mc.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled

        # Channels are in RGB order. Currently model configuration data
        # doesn't provide any information as to other channel orderings
        # (like BGR) so we just assume RGB.
        return ordered

    triton_is = TritonIS(inception_config)
    triton_is.get_model_specs()
    image_data = []
    result_filenames = []
    filename = "/home/tienduchoang/Documents/tris/server/src/clients/python/examples/1.png"
    img = Image.open(filename)
    result_filenames.append(filename)
    image_data.append(preprocess(img, 1, "FP32", 3, 299, 299,
                                 "NONE"))
    for i in range(10):
        tic = time.time()
        contents = triton_is.execute(image_data)
        toc = time.time()
        print('processing time: ', toc - tic)
    for content in contents:
        for (index, results) in enumerate(content):
            for result in results:
                cls = "".join(chr(x) for x in result).split(':')
                print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))


if __name__ == '__main__':
    main()
