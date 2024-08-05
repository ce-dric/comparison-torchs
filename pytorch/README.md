# Pytorch

### 1. virtual env. :seedling:

#### using docker with CMD,

```
# docker pull pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
$ docker run -it --rm --gpus all -v %CD%:/workspace pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
```

### 2. exec :rocket:

#### in container,
```
# python main.py
```