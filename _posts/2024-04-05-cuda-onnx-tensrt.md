---
layout: post
title: Exploring distributed training and inference - CUDA, DDP vs FSDP, ONNX, TensorRT, Triton
---

In this post, we will explore: i) Distributed training on GPUs based on [torch.cuda](https://pytorch.org/docs/stable/cuda.html) and [torch.distributed](https://pytorch.org/docs/stable/distributed.html), ii) Model optimization based on [onnx](https://onnx.ai/get-started.html) and [torch_tensorrt](https://pytorch.org/TensorRT/tutorials/use_from_pytorch.html), and iii) Distributed inference with the [nvidia triton inference server](https://developer.nvidia.com/triton-inference-server) [[repo](https://github.com/sriramgkn/blog-cuda-onnx-tensorrt-triton)]

## `torch.cuda` introduction

`torch.cuda` is the PyTorch package that provides support for CUDA tensor types and GPU operations. It allows you to utilize NVIDIA GPUs for computation with PyTorch.

Key features of torch.cuda:
- Keeps track of the currently selected GPU device
- All CUDA tensors allocated will be created on the currently selected device by default
- Allows changing the selected device with `torch.cuda.set_device(device)`
- Provides functions like `torch.cuda.device_count()` to get the number of GPUs available and `torch.cuda.is_available()` to check if CUDA is supported on the system

Example of using `torch.cuda`:

```python
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device count:", torch.cuda.device_count())
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# Create a tensor directly on GPU
x = torch.randn(5, 3, device=device)
print(x)

# Move a tensor from CPU to GPU using .to()
y = torch.randn(5, 3)
y = y.to(device)
print(y)
```

In this example, we first check if CUDA is available using `torch.cuda.is_available()`. If CUDA is supported, we set the device to "cuda" and print the number of available GPUs using `torch.cuda.device_count()`. If CUDA is not available, we set the device to "cpu".

We then demonstrate two ways to create tensors on the GPU:
1. Directly creating a tensor on the GPU by specifying `device=device` when calling `torch.randn()`.
2. Creating a tensor on the CPU and then moving it to the GPU using the `.to()` method.

By default, PyTorch operations are performed on the currently selected device. You can switch devices using `torch.cuda.set_device(device_index)` [[1](#ref-1)] [[2](#ref-2)].

## `torch.distributed` introduction

`torch.distributed` is PyTorch's package for distributed training across multiple machines or GPUs. It provides primitives for multi-GPU communication and synchronization.

Key features of `torch.distributed`:
- Supports multiple backends like NCCL, Gloo, and MPI for communication
- Provides collective communication operations like `all_reduce`, `broadcast`, and `gather`
- Supports point-to-point communication with `send` and `recv`
- Integrates with higher-level APIs like `FSDP` and `DDP` for easy distributed training

**More about FSDP**

FullyShardedDataParallel (FSDP) is a distributed training API introduced in PyTorch v1.11. FSDP shards model parameters, gradients and optimizer states across data parallel workers, reducing memory usage while still performing synchronous distributed training [[21](#ref-21)] [[22](#ref-22)]. 

Key features of FSDP:
- Shards model parameters across multiple GPUs, with each GPU only holding a portion of the full model
- Shards optimizer states and gradients in addition to model parameters
- Supports CPU offloading of parameters and gradients to further reduce GPU memory usage
- Performs communication and computation in an optimized manner to achieve good training efficiency

**FSDP vs DDP**

DistributedDataParallel (DDP) is another widely used distributed training paradigm in PyTorch. Here are the key differences between FSDP and DDP [[24](#ref-24)] [[26](#ref-26)] [[27](#ref-27)]:

- Parameter Sharding:
  - In DDP, each GPU has a full copy of the model parameters.
  - In FSDP, the model parameters are sharded across GPUs, with each GPU holding only a portion of the parameters.

- Memory Usage:
  - DDP requires each GPU to have enough memory to hold the full model, activations, and optimizer states.
  - FSDP reduces memory usage by sharding the model across GPUs. It can further reduce memory by CPU offloading.

- Communication:
  - DDP performs an `all-reduce` operation to synchronize gradients across GPUs.
  - FSDP performs `reduce-scatter` to shard gradients and `all-gather` to collect parameters before computation.

- Computation:
  - DDP performs computation on the full model on each GPU.
  - FSDP performs computation on the sharded model parameters on each GPU.

- Ease of Use:
  - DDP is generally easier to use, as it requires minimal changes to existing single-GPU code.
  - FSDP requires more setup and configuration, such as wrapping modules and handling unsharding of parameters.

**FSDP code example**

Here's an example of using FSDP to train a simple model:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import default_auto_wrap_policy

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(10, 20)
        self.lin2 = nn.Linear(20, 10)

    def forward(self, x):
        return self.lin2(self.lin1(x))

# Initialize the model and optimizer
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Wrap the model with FSDP
model = FSDP(model, fsdp_auto_wrap_policy=default_auto_wrap_policy)

# Training loop
for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

In this example:
1. We define a simple model `MyModel` with two linear layers.
2. We initialize the model and optimizer as usual.
3. We wrap the model with `FullyShardedDataParallel` using the `default_auto_wrap_policy` to automatically shard the model parameters.
4. We perform the training loop, which looks similar to regular single-GPU training. FSDP handles the sharding and communication behind the scenes.

To use DDP instead of FSDP, you would replace the `FSDP` wrapping with `DDP`:

```python
model = DDP(model)
```

FSDP provides more advanced features like CPU offloading, mixed precision, and different sharding strategies [[28](#ref-28)] [[31](#ref-31)] [[35](#ref-35)]. These can be configured through the `FullyShardedDataParallel` constructor and additional configuration classes.

For example, to enable CPU offloading:

```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model, 
    fsdp_auto_wrap_policy=default_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True)
)
```

This offloads the model parameters to CPU when not in use, freeing up GPU memory [[31](#ref-31)].

Overall, FSDP is a powerful tool for training large models that cannot fit into a single GPU's memory. It requires more setup compared to DDP but can significantly reduce memory usage and enable training of bigger models [[21](#ref-21)] [[22](#ref-22)] [[26](#ref-26)].

## ONNX introduction

ONNX (Open Neural Network Exchange) is an open format to represent machine learning models. It defines a common set of operators and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers [[39](#ref-39)]. ONNX is available both as a [python library](https://pypi.org/project/onnx/) and natively in pytorch as [`torch.onnx`](https://pytorch.org/docs/stable/onnx.html).

Key features of ONNX include:

- Interoperability between different frameworks and platforms
- Ability to optimize models for inference
- Definition of a standard set of operators
- Extensibility for custom operators and data types

Here's an example of creating a simple ONNX model from scratch in Python [[38](#ref-38)]:

```python
import onnx 
from onnx import helper, TensorProto

# Create inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 2])
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [None, 2])

# Create outputs (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 1]) 

# Create a node (NodeProto) 
node_def = helper.make_node(
    'Gemm', # node name
    ['X', 'A'], # inputs
    ['Y'], # outputs
    transA=1
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X, A], # graph inputs
    [Y], # graph outputs
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')
```

This creates a simple ONNX model that performs a matrix multiplication using the Gemm operator. The model is defined using the ONNX Python API by creating the inputs, outputs, nodes, graph, and model in sequence [[38](#ref-38)].

## `torch_tensorrt` introduction

`torch_tensorrt` is a compiler that optimizes PyTorch and TorchScript models using NVIDIA's TensorRT (TRT) deep learning optimizer and runtime. It accelerates inference by optimizing the model graph and generating efficient CUDA kernels.

Key features of `torch_tensorrt`:
- Compiles PyTorch models to optimized TRT engines
- Supports reduced precision inference (FP16 and INT8) for faster execution
- Performs layer fusion, kernel auto-tuning, and memory optimization
- Provides easy-to-use APIs for converting and deploying models

Example of optimizing a PyTorch model with `torch_tensorrt`:

```python
import torch
import torchvision.models as models
import torch_tensorrt

# Load a pretrained PyTorch model
model = models.resnet50(pretrained=True)

# Create example input tensor
input_data = torch.randn(1, 3, 224, 224).to("cuda")

# Compile the model with Torch-TensorRT
model.eval()
trt_model = torch_tensorrt.compile(model, 
    inputs= [torch_tensorrt.Input(input_data.shape)],
    enabled_precisions= {torch.float, torch.half},  # Run with FP32 and FP16
    workspace_size=1 << 22
)

# Run inference with the optimized model
output = trt_model(input_data)
print(output)
```

In this example:
1. We load a pretrained ResNet-50 model using torchvision.models.
2. We create an example input tensor to specify the input shape for compilation.
3. We compile the model using torch_tensorrt.compile, specifying the input shape, enabled precisions (FP32 and FP16), and workspace size.
4. We run inference with the optimized TRT model (the "TRT engine") passing in the input data.

`torch_tensorrt` applies various optimizations like layer fusion, kernel tuning, and precision calibration to generate an efficient TRT engine. This can significantly speed up inference compared to the original PyTorch model [[5](#ref-5)] [[11](#ref-11)].

By leveraging `torch.cuda` for GPU acceleration, `torch.distributed` for multi-GPU training, and `torch_tensorrt` for optimized inference, you can build high-performance PyTorch applications that can scale across multiple devices and machines [[1](#ref-1)] [[2](#ref-2)] [[4](#ref-4)] [[5](#ref-5)].

## All encompassing example (CUDA+FSDP+ONNX+TensorRT, excluding Triton)

Let's go through a detailed example that illustrates the process of developing a PyTorch model, using FSDP for training, exporting it to ONNX, optimizing it with TensorRT, and deploying it for inference. We'll use a simple image classification model as an example.

**Step 1: Develop and Train the PyTorch Model with FSDP**

First, let's define a simple convolutional neural network (CNN) model in PyTorch:

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

Next, let's use FSDP to enable sharding the model across multiple GPUs for training:

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def main():
    model = SimpleCNN()
    model = FSDP(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_dataloader = ...  # Define your dataloader
    for epoch in range(num_epochs):
        train(model, train_dataloader, optimizer, criterion, device)
```

In this example, we wrap the `SimpleCNN` model with `FSDP` to enable sharding across multiple GPUs. The `train` function performs the training loop, and the `main` function sets up the model, optimizer, criterion, and calls the training loop.

**Step 2: Export the Trained Model to ONNX**

After training the PyTorch model, we can export it to the ONNX format using `torch.onnx.export`:

```python
import torch

def export_to_onnx(model, input_shape, onnx_file_path):
    model.eval()
    dummy_input = torch.randn(input_shape, requires_grad=True)
    torch.onnx.export(model, dummy_input, onnx_file_path, opset_version=11)

# Export the trained model to ONNX
input_shape = (1, 3, 32, 32)  # Adjust based on your model's input shape
onnx_file_path = "simple_cnn.onnx"
export_to_onnx(model, input_shape, onnx_file_path)
```

The `export_to_onnx` function takes the trained model, input shape, and the desired ONNX file path. It sets the model to evaluation mode, creates a dummy input tensor, and exports the model to ONNX format using `torch.onnx.export`.

**Step 3: Optimize the ONNX Model with TensorRT**

Now that we have the ONNX model, we can optimize it using TensorRT for efficient inference on NVIDIA GPUs. Here's an example of how to optimize the ONNX model using the TensorRT Python API:

```python
import tensorrt as trt

def optimize_with_tensorrt(onnx_file_path, trt_file_path, max_batch_size, precision):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = max_batch_size
        builder.max_workspace_size = 1 << 30  # 1GB
        if precision == "fp16":
            builder.fp16_mode = True
        elif precision == "int8":
            builder.int8_mode = True
            # Add calibration dataset for INT8 quantization if needed

        with open(onnx_file_path, "rb") as model:
            parser.parse(model.read())

        engine = builder.build_cuda_engine(network)
        with open(trt_file_path, "wb") as f:
            f.write(engine.serialize())

# Optimize the ONNX model with TensorRT
trt_file_path = "simple_cnn.trt"
max_batch_size = 1
precision = "fp16"  # or "fp32" or "int8"
optimize_with_tensorrt(onnx_file_path, trt_file_path, max_batch_size, precision)
```

The `optimize_with_tensorrt` function takes the ONNX file path, desired TensorRT engine file path, maximum batch size, and precision (e.g., FP16, FP32, or INT8). It creates a TensorRT builder, parses the ONNX model, sets the builder configurations (e.g., max batch size, workspace size, precision), and builds the TensorRT engine. Finally, it serializes the engine and saves it to a file.

**Step 4: Deploy the Optimized TensorRT Engine for Inference**

With the optimized TensorRT engine, we can deploy it for efficient inference in an application. Here's an example of how to load the TensorRT engine and perform inference:

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def perform_inference(trt_file_path, input_data):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(trt_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        inputs[0].host = input_data
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        return trt_outputs

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

# Load the TensorRT engine and perform inference
input_data = ...  # Prepare your input data
trt_outputs = perform_inference(trt_file_path, input_data)
```

In this example, the `perform_inference` function loads the TensorRT engine from the serialized file, allocates input and output buffers using `allocate_buffers`, copies the input data to the device, executes the inference using `do_inference`, and finally copies the output back to the host.

The `allocate_buffers` function allocates memory for input and output buffers on both the host and device. The `do_inference` function performs the actual inference by copying input data to the device, executing the inference asynchronously, and copying the output back to the host.

Finally, you can prepare your input data and call the `perform_inference` function to get the inference results.

This example demonstrates the end-to-end process of developing a PyTorch model, using FSDP for training, exporting it to ONNX, optimizing it with TensorRT, and deploying it for efficient inference on NVIDIA GPUs using CUDA acceleration.

## Triton Inference Server

Until now, we learnt about and performed distributed training, but only single GPU inference. Turns out we can do both distributed training and **distributed inference** in sequence in the Nvidia ecosystem. Very cool stuff. Distributed inference is possible due to the [Nvidia Triton Inference Server](https://developer.nvidia.com/triton-inference-server), which we'll look at now.

The Triton Inference Server allows you to scale inference workloads across multiple GPUs and even multiple nodes. Triton handles the distribution and load balancing of inference requests to optimize resource utilization and maximize throughput. Here's a more detailed explanation with code examples:

1. Model Repository:
   - Triton serves models from a model repository, which is a file system directory containing the model files and configuration.
   - Each model in the repository has its own directory, and the model configuration is defined in a `config.pbtxt` file.
   - Example model repository structure:
     ```
     model_repository/
     ├── model1/
     │   ├── 1/
     │   │   ├── model.onnx
     │   │   └── config.pbtxt
     │   └── config.pbtxt
     └── model2/
         ├── 1/
         │   ├── model.plan
         │   └── config.pbtxt
         └── config.pbtxt
     ```

2. Starting Triton Server:
   - Run Triton Inference Server with the model repository path:
     ```bash
     docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /path/to/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models
     ```
   - This command starts Triton server with the specified model repository, exposing the HTTP (8000), gRPC (8001), and Metrics (8002) ports.

3. Sending Inference Requests:
   - Clients can send inference requests to Triton using HTTP/REST or gRPC protocols.
   - Example Python code using Triton's Python client library:
     ```python
     import tritonclient.http as httpclient

     triton_client = httpclient.InferenceServerClient(url="localhost:8000")

     input_data = ...  # Prepare input data
     inputs = [httpclient.InferInput("input_name", input_data.shape, "FLOAT32")]
     inputs[0].set_data_from_numpy(input_data)

     outputs = [httpclient.InferRequestedOutput("output_name")]

     response = triton_client.infer("model_name", inputs, outputs=outputs)
     output_data = response.as_numpy("output_name")
     ```
   - This code snippet demonstrates sending an inference request to Triton using the HTTP client, specifying the model name, input data, and requested output.

4. Distributed Inference:
   - Triton automatically distributes the inference requests across available GPU resources to maximize utilization and throughput.
   - You can specify the number of instances for each model in the `config.pbtxt` file:
     ```
     name: "model_name"
     platform: "tensorrt_plan"
     max_batch_size: 8
     instance_group {
       count: 2
       kind: KIND_GPU
     }
     ```
   - In this example, Triton will create 2 instances of the model on GPUs, allowing parallel execution of inference requests.
   - Triton also supports dynamic batching to combine multiple inference requests into a single batch for improved efficiency.

5. Multi-Node Deployment:
   - For multi-node deployment, you can run Triton Inference Server on multiple nodes and use a load balancer to distribute the inference requests.
   - Clients can send requests to the load balancer endpoint, which forwards them to the appropriate Triton server instance.
   - Triton's model control APIs allow you to load, unload, and manage models across the cluster.

6. Monitoring and Metrics:
   - Triton provides a Prometheus metrics endpoint (default port 8002) for monitoring performance and resource utilization.
   - You can use tools like Grafana to visualize the metrics and gain insights into the inference workload.

By leveraging Triton Inference Server's distributed inference capabilities, you can scale your inference workloads across multiple GPUs and nodes, handle high-throughput scenarios, and optimize resource utilization. Triton abstracts away the complexities of distributed inference, allowing you to focus on deploying and serving your models efficiently.

---
## References

[1] <a id="ref-1"></a> [pytorch.org: CUDA Semantics - PyTorch Documentation](https://pytorch.org/docs/stable/cuda.html)  
[2] <a id="ref-2"></a> [nvidia.com: Accelerating Inference Up to 6x Faster in PyTorch with Torch-TensorRT](https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/)  
[3] <a id="ref-3"></a> [youtube.com: Getting Started with NVIDIA Torch-TensorRT](https://www.youtube.com/watch?v=TU5BMU6iYZ0)  
[4] <a id="ref-4"></a> [github.com: PyTorch TensorRT Repository](https://github.com/pytorch/TensorRT)  
[5] <a id="ref-5"></a> [pytorch.org: Using Torch-TensorRT in PyTorch](https://pytorch.org/TensorRT/tutorials/use_from_pytorch.html)  
[6] <a id="ref-6"></a> [towardsai.net: How to Set Up and Run CUDA Operations in PyTorch](https://towardsai.net/p/l/how-to-set-up-and-run-cuda-operations-in-pytorch)  
[7] <a id="ref-7"></a> [geeksforgeeks.org: How to Set Up and Run CUDA Operations in PyTorch](https://www.geeksforgeeks.org/how-to-set-up-and-run-cuda-operations-in-pytorch/)  
[8] <a id="ref-8"></a> [learnopencv.com: How to Convert a Model from PyTorch to TensorRT and Speed Up Inference](https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/)  
[9] <a id="ref-9"></a> [pytorch.org: Torch-TensorRT Getting Started - ResNet50 Example](https://pytorch.org/TensorRT/_notebooks/Resnet50-example.html)  
[10] <a id="ref-10"></a> [roboflow.com: What is TensorRT?](https://blog.roboflow.com/what-is-tensorrt/)  
[11] <a id="ref-11"></a> [pytorch.org: Torch-TensorRT Getting Started - EfficientNet Example](https://pytorch.org/TensorRT/_notebooks/EfficientNet-example.html)  
[12] <a id="ref-12"></a> [nvidia.com: TensorRT Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)  
[13] <a id="ref-13"></a> [pytorch.org: Torch-TensorRT Tutorials and Notebooks](https://pytorch.org/TensorRT/tutorials/notebooks.html)  
[14] <a id="ref-14"></a> [pytorch.org: Distributed Communication Package - torch.distributed](https://pytorch.org/docs/stable/distributed.html)  
[15] <a id="ref-15"></a> [stackoverflow.com: Using CUDA with PyTorch](https://stackoverflow.com/questions/50954479/using-cuda-with-pytorch)  
[16] <a id="ref-16"></a> [cnvrg.io: PyTorch CUDA: The Ultimate Guide](https://cnvrg.io/pytorch-cuda/)  
[17] <a id="ref-17"></a> [edge-ai-vision.com: Speeding Up Deep Learning Inference Using TensorRT](https://www.edge-ai-vision.com/2020/04/speeding-up-deep-learning-inference-using-tensorrt/)  
[18] <a id="ref-18"></a> [pytorch.org: CUDA Semantics - PyTorch Documentation](https://pytorch.org/docs/stable/notes/cuda.html)  
[19] <a id="ref-19"></a> [run.ai: How to Run PyTorch on GPUs](https://www.run.ai/guides/gpu-deep-learning/pytorch-gpu)  
[20] <a id="ref-20"></a> [pytorch.org: Distributed Data Parallel - PyTorch Tutorials](https://pytorch.org/tutorials/beginner/dist_overview.html)  
[21] <a id="ref-21"></a> [discuss.pytorch.org: torch.dist.DistributedParallel vs Horovod](https://discuss.pytorch.org/t/torch-dist-distributedparallel-vs-horovod/123217)  
[22] <a id="ref-22"></a> [arxiv.org: Fully Sharded Data Parallel: Fast Training on 1024 GPUs](https://arxiv.org/pdf/2304.11277.pdf)  
[23] <a id="ref-23"></a> [pytorch.org: Fully Sharded Data Parallel (FSDP) Advanced Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html)  
[24] <a id="ref-24"></a> [discuss.pytorch.org: DataParallel vs DistributedDataParallel](https://discuss.pytorch.org/t/dataparallel-vs-distributeddataparallel/77891)  
[25] <a id="ref-25"></a> [pytorch.org: Fully Sharded Data Parallel (FSDP) - PyTorch Documentation](https://pytorch.org/docs/stable/fsdp.html)  
[26] <a id="ref-26"></a> [huggingface.co: Introducing Fully Sharded Data Parallel in PyTorch](https://huggingface.co/blog/pytorch-fsdp)  
[27] <a id="ref-27"></a> [pytorch.org: Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)  
[28] <a id="ref-28"></a> [pytorch.org: Fully Sharded Data Parallel (FSDP) Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)  
[29] <a id="ref-29"></a> [youtube.com: Fully Sharded Data Parallel (FSDP) in PyTorch](https://www.youtube.com/watch?v=PjEwLgyzuzQ)  
[30] <a id="ref-30"></a> [pytorch.org: Distributed Data Parallel - PyTorch Tutorials](https://pytorch.org/tutorials/beginner/dist_overview.html)  
[36] <a id="ref-31"></a> [pytorch.org: Introducing PyTorch Fully Sharded Data Parallel API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)  
[35] <a id="ref-32"></a> [github.com: PyTorch FSDP Training Configuration Example](https://github.com/pytorch/examples/blob/main/distributed/FSDP/configs/training.py)  
[34] <a id="ref-33"></a> [engineering.fb.com: Introducing Fully Sharded Data Parallel in PyTorch](https://engineering.fb.com/2021/07/15/open-source/fsdp/)  
[33] <a id="ref-34"></a> [discuss.pytorch.org: How Does FSDP Algorithm Work?](https://discuss.pytorch.org/t/how-does-fsdp-algorithm-work/173277)  
[32] <a id="ref-35"></a> [lightning.ai: Fully Sharded Data Parallel (FSDP) - PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)  
[31] <a id="ref-36"></a> [github.com: PyTorch FSDP Runtime Utilities](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_runtime_utils.py)  
[37] <a id="ref-37"></a> [github.com: PyTorch FSDP Wrapping Utilities](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/wrap.py)  
[38] <a id="ref-38"></a> [towardsdatascience.com: Creating ONNX Models from Scratch](https://towardsdatascience.com/creating-onnx-from-scratch-4063eab80fcd)  
[39] <a id="ref-39"></a> [github.com: ONNX Tutorials](https://github.com/onnx/tutorials)  
[40] <a id="ref-40"></a> [nvidia.com: TensorRT Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)  
[41] <a id="ref-41"></a> [pytorch.org: torch.onnx - PyTorch Documentation](https://pytorch.org/docs/stable/onnx.html)  
[42] <a id="ref-42"></a> [deci.ai: How to Convert a PyTorch Model to ONNX in 5 Minutes](https://deci.ai/blog/how-to-convert-a-pytorch-model-to-onnx/)  
[43] <a id="ref-43"></a> [roboflow.com: What is TensorRT?](https://blog.roboflow.com/what-is-tensorrt/)  
[44] <a id="ref-44"></a> [mmcv.readthedocs.io: TensorRT Plugin](https://mmcv.readthedocs.io/en/v1.4.3/deployment/tensorrt_plugin.html)  
[45] <a id="ref-45"></a> [pytorch.org: Exporting a Simple Model to ONNX](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)  
[46] <a id="ref-46"></a> [pytorch.org: Deploying a Super Resolution Model in C++ using ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)  
[47] <a id="ref-47"></a> [nvidia.com: Speed Up Inference with TensorRT](https://developer.nvidia.com/blog/speed-up-inference-tensorrt/)  
[48] <a id="ref-48"></a> [github.com: NVIDIA TensorRT Repository](https://github.com/NVIDIA/TensorRT)  
[49] <a id="ref-49"></a> [genesiscloud.com: Deployment of Deep Learning Models on Genesis Cloud - TensorRT](https://blog.genesiscloud.com/2022/deployment-of-deep-learning-models-on-genesis-cloud-tensorrt)  
[50] <a id="ref-50"></a> [microsoft.com: Optimizing and Deploying Transformer INT8 Inference with ONNX Runtime & TensorRT on NVIDIA GPUs](https://cloudblogs.microsoft.com/opensource/2022/05/02/optimizing-and-deploying-transformer-int8-inference-with-onnx-runtime-tensorrt-on-nvidia-gpus/)  
[51] <a id="ref-51"></a> [microsoft.com: Convert a PyTorch Model to ONNX](https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model)  
[52] <a id="ref-52"></a> [huggingface.co: Introducing Fully Sharded Data Parallel in PyTorch](https://huggingface.co/blog/pytorch-fsdp)  
[53] <a id="ref-53"></a> [lightning.ai: Fully Sharded Data Parallel (FSDP) - PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)  
[54] <a id="ref-54"></a> [onnxruntime.ai: Export PyTorch Model to ONNX](https://onnxruntime.ai/docs/tutorials/export-pytorch-model.html)  
[55] <a id="ref-55"></a> [pytorch.org: Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)  
[56] <a id="ref-56"></a> [edge-ai-vision.com: Speeding Up Deep Learning Inference Using TensorRT](https://www.edge-ai-vision.com/2020/04/speeding-up-deep-learning-inference-using-tensorrt/)  
[57] <a id="ref-57"></a> [pytorch-geometric.readthedocs.io: Multi-GPU Training (Vanilla PyTorch)](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/multi_gpu_vanilla.html)  
[58] <a id="ref-58"></a> [huggingface.co: Exporting Transformers Models](https://huggingface.co/docs/transformers/main/en/serialization)  
[59] <a id="ref-59"></a> [txstate.edu: Accelerating Deep Learning Inference with TensorRT](https://userweb.cs.txstate.edu/~k_y47/webpage/pubs/icess22.pdf)  
[60] <a id="ref-60"></a> [paloaltonetworks.com: ML Inference Workloads on the NVIDIA Triton Inference Server](https://live.paloaltonetworks.com/t5/community-blogs/ml-inference-workloads-on-the-triton-inference-server/ba-p/545039)  
[61] <a id="ref-61"></a> [bentoml.org: Deploying to NVIDIA Triton Inference Server](https://docs.bentoml.org/en/v1.1.11/integrations/triton.html)  
[62] <a id="ref-62"></a> [run.ai: NVIDIA Triton Inference Server: The Complete Guide](https://www.run.ai/guides/machine-learning-engineering/triton-inference-server)  
[63] <a id="ref-63"></a> [towardsai.net: Build a Triton Inference Server with MNIST Example](https://pub.towardsai.net/build-a-triton-inference-server-with-mnist-example-part-1-4-1233445ab56f?gi=75f20d3bc7b9)  
[64] <a id="ref-64"></a> [marvik.ai: Deploying LLaMA2 with NVIDIA Triton Inference Server](https://blog.marvik.ai/2023/10/16/deploying-llama2-with-nvidia-triton-inference-server/)  
[65] <a id="ref-65"></a> [forums.developer.nvidia.com: Triton Inference Server Example - Simple gRPC Infer Client](https://forums.developer.nvidia.com/t/triton-infererence-server-example-simple-grpc-infer-client-py/204947)  
[66] <a id="ref-66"></a> [github.com: NVIDIA Triton Inference Server - FAQ](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/faq.md)  
[67] <a id="ref-67"></a> [github.com: NVIDIA Triton Inference Server - Documentation](https://github.com/triton-inference-server/server/blob/main/docs/README.md)

_Assisted by claude-3-opus on [perplexity.ai](https://perplexity.ai)_

<!-- -------------------------------------------------------------- -->
<!-- 
sequence: renumber, accumulate, format

to increment numbers, use multiple cursors then emmet shortcuts

regex...
\[(\d+)\]
to
 [[$1](#ref-$1)]

regex...
\[(\d+)\] (.*)
to
[$1] <a id="ref-$1"></a> [display text]($2)  

change "Citations:" to "## References"
-->
<!-- 
Include images like this:  
<figure style="text-align: center; width:100%;">
    <img src="{{site.baseurl}}/images/experimenting_files/experimenting_18_1.svg" alt="___" style="max-width:90%; 
    height: auto; margin:3% auto; display:block;">
    <figcaption>___</figcaption>
</figure> 
-->
<!-- 
Include code snippets like this:  
```python 
def square(x):
    return x**2
``` 
-->
<!-- 
Cite like this [[2](#ref-2)], and this [[3](#ref-3)]. Use two extra spaces at end of each line for line break
---
## References  
[1] <a id="ref-1"></a> [display text](hyperlink)  
[2] <a id="ref-2"></a> [display text](hyperlink) 
[3] <a id="ref-3"></a> [display text](hyperlink)  
_Assisted by claude-3-opus on [perplexity.ai](https://perplexity.ai)_ 
-->
<!-- -------------------------------------------------------------- -->