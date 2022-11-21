# Robyn Marketing Mix Modeling with Python SageMaker

This is sample code to run [Robyn](https://github.com/facebookexperimental/Robyn) Marketing Mix Model library from Python using SageMaker.

The main purpose of this sample code is showing how to run encapsulated Robyn docker image in isolated SageMaker runtime and pass data / hyperparameters from python.

You basically need to upload data to S3 and run SageMaker Robyn Image as remote procedure and get result in S3.

The main Robyn code is in `app/train.R`, but it has minimal change from [Robyn demo code](https://github.com/facebookexperimental/Robyn/blob/main/demo/demo.R), so you still need to modify R code to handle production workload.

**Benefit of using SageMaker**

- You can isolate Robyn runtime and call it as remote procedure, so you can use it from any other language which has SageMaker SDK or AWS SDK such as Python
- You can use powerful machine on-demand for only duration of the training to speed up the training as well as minimizing the cost.

## Usage

1. Clone this repository and follow `train.ipynb` in either SageMaker Notebook or SageMaker Studio Notebook.

