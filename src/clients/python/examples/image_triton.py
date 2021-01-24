import argparse
import os
import sys
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

import time


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


class TritonModelSpecs:
    def __init__(self, inception_config):
        self.model_name = inception_config["model_name"]
        self.model_version = inception_config["model_version"]
        self.is_streaming = inception_config["streaming"]
        self.is_async_set = inception_config["async_set"]
        self.batch_size = inception_config["batch_size"]
        self.protocol = inception_config["protocol"]
        self.url = inception_config["url"]
        self.verbose = inception_config["verbose"]
        self.classes = 1
        self.triton_client = None

    def parse_model_grpc(self, model_metadata, model_config):
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

        return (model_config.max_batch_size, input_metadata.name,
                output_metadata.name, c, h, w, input_config.format,
                input_metadata.datatype)

    def parse_model_http(self, model_metadata, model_config):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(model_metadata['inputs']) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(model_metadata['inputs'])))
        if len(model_metadata['outputs']) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(model_metadata['outputs'])))

        if len(model_config['input']) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config['input'])))

        input_metadata = model_metadata['inputs'][0]
        input_config = model_config['input'][0]
        output_metadata = model_metadata['outputs'][0]

        max_batch_size = 0
        if 'max_batch_size' in model_config:
            max_batch_size = model_config['max_batch_size']

        if output_metadata['datatype'] != "FP32":
            raise Exception("expecting output datatype to be FP32, model '" +
                            model_metadata['name'] + "' output type is " +
                            output_metadata['datatype'])

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = (max_batch_size > 0)
        non_one_cnt = 0
        for dim in output_metadata['shape']:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model input must have 3 dims (not counting the batch dimension),
        # either CHW or HWC
        input_batch_dim = (max_batch_size > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata['shape']) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                format(expected_input_dims, model_metadata['name'],
                       len(input_metadata['shape'])))

        if ((input_config['format'] != "FORMAT_NCHW") and
                (input_config['format'] != "FORMAT_NHWC")):
            raise Exception("unexpected input format " + input_config['format'] +
                            ", expecting FORMAT_NCHW or FORMAT_NHWC")

        if input_config['format'] == "FORMAT_NHWC":
            h = input_metadata['shape'][1 if input_batch_dim else 0]
            w = input_metadata['shape'][2 if input_batch_dim else 1]
            c = input_metadata['shape'][3 if input_batch_dim else 2]
        else:
            c = input_metadata['shape'][1 if input_batch_dim else 0]
            h = input_metadata['shape'][2 if input_batch_dim else 1]
            w = input_metadata['shape'][3 if input_batch_dim else 2]

        return (max_batch_size, input_metadata['name'], output_metadata['name'], c,
                h, w, input_config['format'], input_metadata['datatype'])

    def get_model_specs(self):
        if self.is_streaming and self.protocol.lower() != "grpc":
            raise Exception("Streaming is only allowed with gRPC protocol")

        try:
            if self.protocol.lower() == "grpc":
                # Create gRPC client for communicating with the server
                self.triton_client = grpcclient.InferenceServerClient(
                    url=self.url, verbose=self.verbose)
            else:
                # Specify large enough concurrency to handle the
                # the number of requests.
                concurrency = 20 if self.is_async_set else 1
                self.triton_client = httpclient.InferenceServerClient(
                    url=self.url, verbose=self.verbose, concurrency=concurrency)
        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)

        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        try:
            model_metadata = self.triton_client.get_model_metadata(
                model_name=self.model_name, model_version=self.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)

        try:
            model_config = self.triton_client.get_model_config(
                model_name=self.model_name, model_version=self.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)

        if self.protocol.lower() == "grpc":
            self.max_batch_size, self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype = self.parse_model_grpc(
                model_metadata, model_config.config)
        else:
            self.max_batch_size, self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype = self.parse_model_http(
                model_metadata, model_config)


class TritonIS(TritonModelSpecs):
    def __init__(self, inception_config):
        super().__init__(inception_config)

    def requestGenerator(self, batched_image_data, input_name, output_name, dtype):
        # Set the input data
        inputs = []
        if self.protocol.lower() == "grpc":
            inputs.append(
                grpcclient.InferInput(input_name, batched_image_data.shape, dtype))
            inputs[0].set_data_from_numpy(batched_image_data)
        else:
            inputs.append(
                httpclient.InferInput(input_name, batched_image_data.shape, dtype))
            inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

        outputs = []
        if self.protocol.lower() == "grpc":
            outputs.append(
                grpcclient.InferRequestedOutput(output_name,
                                                class_count=self.classes))
        else:
            outputs.append(
                httpclient.InferRequestedOutput(output_name,
                                                binary_data=True,
                                                class_count=self.classes))

        yield inputs, outputs

    def execute(self, image_data, user_data):
        # Send requests of self.batch_size images. If the number of
        # images isn't an exact multiple of self.batch_size then just
        # start over with the first images until the batch is filled.
        responses = []
        image_idx = 0
        last_request = False

        # Holds the handles to the ongoing HTTP async requests.
        async_requests = []

        sent_count = 0

        if self.is_streaming:
            self.triton_client.start_stream(
                partial(completion_callback, user_data))

        while not last_request:
            repeated_image_data = []

            for idx in range(self.batch_size):
                repeated_image_data.append(image_data[image_idx])
                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True

            if self.max_batch_size > 0:
                batched_image_data = np.stack(repeated_image_data, axis=0)
            else:
                batched_image_data = repeated_image_data[0]

            # Send request
            try:
                for inputs, outputs in self.requestGenerator(
                        batched_image_data, self.input_name, self.output_name, self.dtype):
                    sent_count += 1
                    if self.is_streaming:
                        self.triton_client.async_stream_infer(
                            self.model_name,
                            inputs,
                            request_id=str(sent_count),
                            model_version=self.model_version,
                            outputs=outputs)
                    elif self.is_async_set:
                        if self.protocol.lower() == "grpc":
                            self.triton_client.async_infer(
                                self.model_name,
                                inputs,
                                partial(completion_callback, user_data),
                                request_id=str(sent_count),
                                model_version=self.model_version,
                                outputs=outputs)
                        else:
                            async_requests.append(
                                self.triton_client.async_infer(
                                    self.model_name,
                                    inputs,
                                    request_id=str(sent_count),
                                    model_version=self.model_version,
                                    outputs=outputs))
                    else:
                        responses.append(
                            self.triton_client.infer(self.model_name,
                                                     inputs,
                                                     request_id=str(
                                                         sent_count),
                                                     model_version=self.model_version,
                                                     outputs=outputs))

            except InferenceServerException as e:
                print("inference failed: " + str(e))
                if self.is_streaming:
                    self.triton_client.stop_stream()
                sys.exit(1)

        if self.is_streaming:
            self.triton_client.stop_stream()

        if self.protocol.lower() == "grpc":
            if self.is_streaming or self.is_async_set:
                processed_count = 0
                while processed_count < sent_count:
                    (results, error) = user_data._completed_requests.get()
                    processed_count += 1
                    if error is not None:
                        print("inference failed: " + str(error))
                        sys.exit(1)
                    responses.append(results)
        else:
            if self.is_async_set:
                # Collect results from the ongoing async requests
                # for HTTP Async requests.
                for async_request in async_requests:
                    responses.append(async_request.get_result())
        return responses


def preprocess(img, format, dtype, c, h, w, scaling, protocol):
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

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 127.5) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if protocol == "grpc":
        if format == mc.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled
    else:
        if format == "FORMAT_NCHW":
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(results, output_name, batch_size, batching):
    """
    Post-process results to show classifications.
    """

    output_array = results.as_numpy(output_name)
    if len(output_array) != batch_size:
        raise Exception("expected {} results, got {}".format(
            batch_size, len(output_array)))

    # Include special handling for non-batching models
    for results in output_array:
        if not batching:
            results = [results]
        for result in results:
            if output_array.dtype.type == np.bytes_:
                cls = "".join(chr(x) for x in result).split(':')
            else:
                cls = result.split(':')
            print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))


inception_config = {
    "model_name": "inception_graphdef",
    "model_version": "",
    "protocol": "GRPC",
    "url": "localhost:8001",
    "verbose": False,
    "streaming": True,
    "async_set": True,
    "batch_size": 1
}

triton_is = TritonIS(inception_config)
triton_is.get_model_specs()
image_data = []
filename = "/home/tienduchoang/Documents/tris/server/src/clients/python/examples/1.png"
img = Image.open(filename)
image_data.append(
    preprocess(img, format, "FP32", 2, 299, 299, None,
               inception_config["protocol"].lower()))

user_data = UserData()
for i in range(10):
    tic = time.time()
    responses = triton_is.execute(image_data, user_data)
    toc = time.time()
    print("processing time: ", toc - tic)
# Include special handling for non-batching models
for response in responses:
    if triton_is.protocol.lower() == "grpc":
        this_id = response.get_response().id
    else:
        this_id = response.get_response()["id"]
    print("Request {}, batch size {}".format(this_id, triton_is.batch_size))
    postprocess(response, triton_is.output_name,
                triton_is.batch_size, triton_is.max_batch_size > 0)
