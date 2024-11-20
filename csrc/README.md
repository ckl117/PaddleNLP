# PaddleNLP 自定义 OP

此文档介绍如何编译安装 PaddleNLP 自定义 OP。

## 安装 C++ 依赖

```shell
pip install -r requirements.txt
```

## 编译 Cuda 算子

编译
```shell
python setup_cuda.py install
```

### 手动安装 Cutlass 库
1. 访问 Cutlass 仓库: [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

2. 拉取代码:
    git clone -b v3.5.0 --single-branch https://github.com/NVIDIA/cutlass.git

3. 将下载的 `cutlass` 目录放在 `csrc/third_party/cutlass`下

4. 重新编译 Cuda 算子
```shell
python setup_cuda.py install
```

### FP8 GEMM 自动调优

确保 `cutlass` 库已经安装，然后执行以下命令进行自动调优。
- 对于89架构的 GPU，CUDA 版本至少为12.4
- 对于90架构的 GPU，CUDA 版本至少为12.0
```shell
sh tune_fp8_gemm.sh
```
