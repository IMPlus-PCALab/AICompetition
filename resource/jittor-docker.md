---
title: Jittor 框架开发环境的 Docker 镜像制作
authors: admin
tags: []
categories: []
featured: false
draft: true
---

最近参加了一场指定需要使用 Jittor 框架进行开发的遥感目标检测的比赛，进入决赛的队伍需要提交 Docker 镜像，这边文章记录一下打包流程以及遇到的一些问题。其中, 主要流程在第 2 节, 第 1 和第 3 节为辅助内容.

- [1 准备工作](#1-准备工作)
  - [1.1 宿主机器的选择](#11-宿主机器的选择)
  - [1.2 关于 CUDA 的一些概念厘清](#12-关于-cuda-的一些概念厘清)
  - [1.3 GPU 型号的选择](#13-gpu-型号的选择)
  - [1.4 CUDA 相关的版本选择](#14-cuda-相关的版本选择)
- [2 Docker 制作全流程](#2-docker-制作全流程)
  - [2.1 宿主机器的配置](#21-宿主机器的配置)
    - [2.1.1 修改 apt-get 源为国内镜像源的方法](#211-修改-apt-get-源为国内镜像源的方法)
    - [2.1.2 安装一些必要的工具](#212-安装一些必要的工具)
    - [2.1.3 安装 CUDA Toolkit (nvidia)](#213-安装-cuda-toolkit-nvidia)
    - [2.1.4 安装 Docker](#214-安装-docker)
    - [2.1.5 安装 NVIDIA Container Toolkit](#215-安装-nvidia-container-toolkit)
  - [2.2 在 Docker 内配置好代码环境](#22-在-docker-内配置好代码环境)
    - [2.2.1 启动 Docker 镜像](#221-启动-docker-镜像)
    - [2.2.2 安装好代码环境](#222-安装好代码环境)
  - [2.3 保存镜像并测试](#23-保存镜像并测试)
- [3 其他](#3-其他)
  - [3.1 CUDA Toolkit (NVIDIA) 已经安装，但 NVCC 找不到](#31-cuda-toolkit-nvidia-已经安装但-nvcc-找不到)
  - [3.2 docker 从一个容器中 exit 后，怎么再进入这个容器？](#32-docker-从一个容器中-exit-后怎么再进入这个容器)
  - [3.3 如何查看当前环境的 CUDA 版本](#33-如何查看当前环境的-cuda-版本)
  - [3.4 Conda 的 cudatoolkit](#34-conda-的-cudatoolkit)
  - [3.5 runtime API 与 driver API](#35-runtime-api-与-driver-api)
- [参考资料](#参考资料)

## 1 准备工作

### 1.1 宿主机器的选择

首先需要选择一台机器，这台机器需要满足以下条件：

1. 这台机器不能是以 Docker 的方式运行的，因为我们不能在 Docker 容器中运行 Docker 容器（不考虑 Docker in Docker 等 Hack 方式），这就排除了 AIMAX 机器以及部分 AI 云服务器上制作 Docker 的可能性（运行 `sudo systemctl restart docker` 报错让我意识到可能自己正在 Docker 内制作 Docker 镜像）；
2. 本地的 Linux 机器的显存也不能被其他训练程序占满，因为 Jittor 框架编译的时候需要占用显存，如果显存被占满，编译会失败；

最后，我们在阿里云上开了一台 V100 的虚拟机，才能进行后面的操作。

### 1.2 关于 CUDA 的一些概念厘清

根据[官方文档](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)，如下图所示，CUDA 主要由以下 2 个部分组成：

<table>
  <tr>
    <center><img src="https://docs.nvidia.com/deploy/cuda-compatibility/graphics/CUDA-components.png" style="width:50vw"></center>
    <center>图 1 Components of CUDA </center>
  </tr>
</table>

1. **CUDA Toolkit** (缩写 CTK)，主要用于 **build applications**, 包括运行环境 (runtime，即 cudart)，库文件 (libraries)，开发工具 (tools) 等;
2. **NVIDIA Display Driver Package**, 主要用于驱动 GPU 来 **run applications**，可以进一步细分为：
   1. **CUDA user-mode driver** (libcuda.so)，即 CUDA Driver, 用户驱动组件，用于**运行 CUDA 程序**，可以理解为 CUDA 运行环境；
   2. **NVIDIA kernel-mode driver** (nvidia.ko)，即 GPU Driver, 内核驱动组件，用于**驱动 GPU**，就是硬件驱动；

由于 driver package 即包括了 user mode CUDA driver (libcuda.so)，也包括了 kernel mode driver (nvidia.ko)，所以我们后面只考虑统一的 **NVIDIA Driver**，不再细分两者。操作系统、GPU Driver 和 CUDA Toolkit 以及 PyTorch 等框架之间的关系如下图所示：

<table>
  <tr>
    <center><img src="https://pic2.zhimg.com/v2-486d15d8f627a38eb59a0ec0793f3a73_r.jpg?source=1940ef5c" style="width:50vw"></center>
    <center>图 2 CUDA 的软硬件联系 </center>
  </tr>
</table>

CUDA Toolkit 以及 GPU Driver 各自的组成如下图所示：

<table>
  <tr>
    <center><img src="https://blogs.nvidia.com/wp-content/uploads/2012/09/cuda-apps-and-libraries.png" style="width:90vw"></center>
    <center>图 3 CUDA 生态示意图 </center>
  </tr>
</table>

CUDA 的编译器，就是 nvcc，全称为 NVIDIA's CUDA Compiler。当然，某些场合，用户也可以跳过 CUDA Runtime API 直接跟 GPU Driver 通过 Driver API 进行交互，如下图所示。

<table>
  <tr>
    <center><img src="https://www.researchgate.net/profile/Irina-Mocanu-2/publication/227487008/figure/fig1/AS:669571841277957@1536649775213/The-CUDA-software-stack.png" style="width:50vw"></center>
    <center>图 4 CUDA Driver API 与 Runtime API </center>
  </tr>
</table>

更详细的运行结构全景图如下所示：

<table>
  <tr>
    <center><img src="https://ask.qcloudimg.com/draft/1215004/bfh9gsoilv.png?imageView2/2/w/1620" style="width:50vw"></center>
    <center>图 5 CUDA 运行结构 </center>
  </tr>
</table>

好在，我们通常是通过 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archivE) 来安装，其实是包含了真正的 CUDA Toolkit 和 NVIDIA Driver 这两样东西的。因此，我们通过该种方式来安装 CUDA 的时候，其实是同时在升级 toolkit 和 driver，如下图所示：

<table>
  <tr>
    <center><img src="https://docs.nvidia.com/deploy/cuda-compatibility/graphics/forward-compatibility.png" style="width:50vw"></center>
    <center>图  6 CUDA Upgrade Path </center>
  </tr>
</table>

### 1.3 GPU 型号的选择

GPU 型号其实没啥可说的，一般我们都选择 3090 或者 V100 这种。这里顺道记录一下关于 GPU 型号的基础知识，分别是 GPU 的架构、GPU 的系列。

最先需要关注的概念是 GPU 的**架构**，即硬件的设计方式，例如流处理器簇中有多少个 core、是否有 L1 or L2 缓存、是否有双精度计算单元等等。每一代的架构都代表了一种如何去更好完成并行的思想。对我们来说，架构的重要性在于，它决定了我们能够使用的 CUDA Toolkit 的最低版本。
这是因为架构代表了底层的硬件特性（GPU 指令集），CUDA 则是 building application 的软件生态，而驱动（Diriver）则是连接底层架构与上层软件生态之间的桥梁。

CUDA、驱动、硬件架构这三者会在一定范围内相互兼容，但是肯定是先有的硬件架构，然后上层的软件去适配硬件。迄今为止，英伟达已经出了 8 代硬件架构，分别是 Tesla、Fermi、Kepler、Maxwell、Pascal、Volta、Turing 以及 Ampere。为了方便我们搞清楚 CUDA、驱动、显卡型号之间的适配关系，英伟达有一个 **Compute capability version** 的概念。每个显卡型号都有一个相应的 Compute capability version，在 [CUDA wiki 的 GPUs supported](https://en.wikipedia.org/wiki/CUDA) 这张大表下可以看到每一型号显卡所需的 Compute capability 版本。比如 RTX 3090 的 Compute capability 是 8.6。同样，从同一页面的 GPUs supported 这里也可以查到 CUDA SDK 11.1 及之后的版本就开始支持 Compute capability 8.6 了。由此，我们就可以知道 3090 的最低 CUDA Toolkit 版本为 11.1。

GeFore、Quadro、Tesla（不是 Tesla 架构的那个 Tesla） 这些则属于显卡系列，同时期的显卡在架构上是一致的，之所以会分为不同的系列，是出于商业上的考虑，在性能上有所区别使得不同的系列有不同的定位，例如 GeForce 是游戏显卡，Quadro 是专业显卡，Tesla 是专业计算显卡。

### 1.4 CUDA 相关的版本选择

在确定了具体的显卡型号之后，我们就可以确定 CUDA Toolkit 和 Driver 的版本了。因为软件总是后向兼容的，因此我们可以在满足包依赖的情况下选择尽量新的CUDA Toolkit 版本即可，那么会有哪些依赖呢？

我们从后向前捋一下：

1. MMCV 会要求 CUDA 版本和 PyTorch 版本，具体可见 <https://mmcv.readthedocs.io/en/latest/get_started/installation.html>;
2. PyTorch 会要求 CUDA 版本，具体可见 <https://pytorch.org/get-started/previous-versions/>
3. Jittor 会要求 CUDA 版本（NVCC 版本），具体可见 <https://cg.cs.tsinghua.edu.cn/jittor/download/>；
4. CUDA Toolkit (nvidia) 会要求操作系统版本，具体可见 <https://developer.nvidia.com/cuda-downloads>;
5. GPU 型号会要求 CUDA 版本，具体可见 <https://docs.nvidia.com/deploy/cuda-compatibility/index.html>。

各个环节的要求如下：

1. 根据 MMCV 的要求，PyTorch 的版本只能选择最高到 **torch 1.11**，而 CUDA 版本只能选择 11.5、**11.3** 以及 10.2；
2. 根据 PyTorch 的要求，CUDA 版本只能选择 **11.3** 和 10.2；
3. 根据 Jittor 的要求，CUDA 版本需要大于 10.0（Windows 大于 10.2），但是 [`python -m jittor_utils.install_cuda`](https://github.com/Jittor/jittor/blob/master/python/jittor_utils/install_cuda.py) 命令会根据 nvidia-smi 返回的 CUDA 版本安装 Jittor 所需的 CUDA；
4. 根据 CUDA Toolkit (nvidia) 的要求，Ubuntu 系统只能选择 **18.04**, 20.04 和 22.04 这三种。
5. 具体的显卡型号也会对 CUDA Toolkit 和 Driver 有一定的要求. 从 [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) 可以知道，Ampere 架构的 GPU 的 CUDA Toolkit (CTK Support) 必须是 11.0 或者更高，而 Turing 架构的 GPU 的 CTK Support 必须是 10.0 或者更高。在 PyTorch，Jittor，MMCV 这些软件生态支持的情况下，在确定了 CUDA 的大版本之后（11.x、10.x 这种），往往选择越新的版本即可，比如对于 3090 我们可以选择 [CUDA Toolkit 11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive).
   1. 更快的获取 GPU 型号的最低 CUDA 版本的途径是查阅 [CUDA wiki 的 GPUs supported](https://en.wikipedia.org/wiki/CUDA) 这张大表，找到对应 GPU 型号，可以获得对应的 Compute capability 版本，比如 3090 是 8.6，而通过表格上面的 Supported CUDA level of GPU and card 列表可以知道，3090 必须是 11.1 之后。

需要多提一嘴的是，mmcv-full 是仅在 PyTorch 1.x.0 上编译的。由于 PyTorch 1.x.0 和 1.x.1 通常是兼容的，如果当前的 PyTorch 版本是 1.x.1，那么我们可以选择安装 PyTorch 1.x.0 对应的 mmcv-full。例如，如果 PyTorch 版本是 1.8.1，CUDA 版本是 11.1，那么我们可以用如下命令安装 mmcv-full：

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
```

上面我们是通过打包了 CUDA Toolkit 和 Driver 的 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archivE) 来安装 CUDA 的。然而我们知道，NVIDIA Driver 和 CUDA Toolkit（真正的，不打包 Driver）是可以分开安装的。在 [NVIDIA > 驱动程序下载 > 高级搜索](https://www.nvidia.cn/Download/Find.aspx?lang=cn) 中，我们可以找到适配具体某个显卡型号的所有 GPU Driver 版本。但确定版本的逻辑还是一样的，PyTorch、MMCV 这些上层应用决定了 CUDA Toolkit 的版本，比如 11.3.1, 而 CUDA Toolkit 版本又决定了 NVIDIA Driver 的最低版本。比如，根据 [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) 可以知道，CUDA 11.x 所需要的 Linux x86_64 的最低 Driver 版本是 450.80.02。

但是通常，从 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archivE) 中下载的打包的 Driver 的版本是远高于最低版本的。我们点击具体的 Toolkit 的 Versioned Online Documentation 页面中的 Release Notes 页面，可以查看到这个 Toolkit 中所包含的每个组件的具体版本，当然也包括 NVIDIA Driver 的版本，比如：

- CUDA 11.0.3 的 NVIDIA Linux Driver 的版本是 450.51.06，不满足最低 450.80.02 的要求；
- CUDA 11.1.0 的 NVIDIA Linux Driver 的版本是 455.23.05，满足大于等于 450.80.02 的要求；
- CUDA 11.3.0 的 NVIDIA Linux Driver 的版本是 465.19.01。

因此，3090 对应的 CUDA Toolkit 必须是 11.1.0 之后的，这是 GPU Driver 版本的要求。

最后，因为 CUDA Toolkit 这个名次被好几个不同的概念使用，这里区分一下：

1. CUDA Toolkit (nvidia)： CUDA 完整的工具安装包，其中提供了 Nvidia 驱动程序、开发 CUDA 程序相关的开发工具包等可供安装的选项。包括 CUDA 程序的编译器、IDE、调试器等，CUDA 程序所对应的各式库文件以及它们的头文件。
2. CUDA Toolkit (Pytorch)： CUDA 不完整的工具安装包，其主要包含在使用 CUDA 相关的功能时所依赖的动态链接库。不会安装驱动程序。

CUDA Toolkit 完整和不完整的区别：在安装了 CUDA Toolkit (Pytorch) 后，只要系统上存在与当前的 cudatoolkit 所兼容的 Nvidia 驱动，则已经编译好的 CUDA 相关的程序就可以直接运行，不需要重新进行编译过程。例如，CUDA Toolkit (nvidia) 11.3 所打包的 Linux Driver 的版本是 465.19.01，但 CUDA 11.x 所需要的 Linux x86_64 的最低 Driver 版本是 450.80.02，因此即使当前机器上的 Driver 版本低于 465.19.01，但只要其大于等于 450.80.02，那么 CUDA Toolkit (Pytorch) 11.3 也可以正常使用。

至此，我们就选用 CUDA Toolkit 11.3.0 作为我们的 CUDA Toolkit 版本。

## 2 Docker 制作全流程

大体的步骤应该为以下几步：

1. 在宿主机器上安装好 CUDA, Docker 以及 nvidia-docker
2. 在 Docker 内配置好代码所需的环境
3. 保存镜像并测试该镜像可以正常运行

需要提的一点是，在这里我们计划通过 docker commit 来创建一个镜像，而非通过 docker build 命令，两者的差别在于：

- docker commit：从容器创建一个新的镜像；
- docker build：配合 Dockerfile 文件创建镜像。

但 docker commit 其实是有很多隐忧的，具体可见以下内容：

- [Docker 运维:docker commit 真有那么香么？](https://zhuanlan.zhihu.com/p/147026163)
- [慎用 docker commit](https://yeasy.gitbook.io/docker_practice/image/commit#shen-yong-docker-commit)
- [docker commit 和 docker build （实战使用以及区别）](https://blog.csdn.net/alwaysbefine/article/details/111375658)

其实，写 dockerfile 就跟写代码一样，直接在容器里操作就像写解释型语言随时验证，而直接写 dockerfile 就像编译型的，每次写完要编译后才能验证。

对于创建好的镜像，我们是使用 docker save 来保存，而非 docker export，具体差别可看 [docker：export/save/commit 谁才是你心中那个她](https://zhuanlan.zhihu.com/p/152219012).

但上面这些是更加进阶的内容，不影响我们的主要工作，因此这里就不再展开了。

### 2.1 宿主机器的配置

我们的宿主机器采用的阿里云上的一台新的 Ubuntu 18.04 机器，如果是本地机器可以适当跳过某些步骤。
如果是以 root 身份登录，需要去掉 sudo 命令。

#### 2.1.1 修改 apt-get 源为国内镜像源的方法

Ubuntu 默认的源在国外，而阿里云机器上的 Ubuntu 默认的源是阿里云的源，但速度也非常慢，我们可以将源修改为其他国内的镜像源，这里我们选择中科大的源，这样可以加快下载速度。

先将原文件备份

```shell
$ sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
```

然后编辑源列表文件

```shell
$ sudo vim /etc/apt/sources.list
```

如果当前机器没有 vim，可以先单独安装一下 vim：

```shell
$ sudo apt-get update && sudo apt-get install -y vim
```

用 `vim` 打开 `/etc/apt/sources.list` 之后，将原来的列表删除，添加如下内容：

```shell
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
```

之后再运行如下命令来更新源：

```shell
$ sudo apt-get update
```

#### 2.1.2 安装一些必要的工具

为了后续 CUDA 的安装，我们需要更新并安装编译需要的包

```shell
$ sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3 tmux
```

注意，如果是 Ubuntu 20.04, 则要将 libgfortran3 换成 libgfortran5

由于整个 Docker 过程制作和测试过程颇为耗时（3 个小时左右），需要使用 tmux 来防止 ssh 断开连接。

```shell
$ tmux new -s docker
```

这里，docker 是 tmux 的会话名，可以自行修改。

#### 2.1.3 安装 CUDA Toolkit (nvidia)

访问 [NVIDIA 官方网站](https://developer.nvidia.com/cuda-toolkit-archive) 获取相应的 CUDA 版本, 我们以 CUDA Toolkit 11.3.1 为例。

```shell
$ wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
$ sudo sh cuda_11.3.1_465.19.01_linux.run
```

新的 toolkit 不会像以前那样不停地显示进度，而是在一长段空白后直接就好了，需要耐心等待一下。

当安装完成后，运行下面的命令就可以看到该实例的 GPU 了：

```shell
$ nvidia-smi
```

#### 2.1.4 安装 Docker

按照 [Install using the repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) 这个安装方式. 注意记得 Check 一下 OS requirements，支持的最老的 OS 是 Ubuntu Bionic 18.04 (LTS)。

**Step 1**: Set up the repository

```shell
# Update the apt package index and install packages to allow apt to use a repository over HTTPS:
$ sudo apt-get update

$ sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker’s official GPG key:
$ sudo mkdir -p /etc/apt/keyrings
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Use the following command to set up the stable repository.
$ echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

**Step 2**: Install Docker Engine

```shell
# Update the apt package index, and install the latest version of Docker Engine and containerd
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

#### 2.1.5 安装 NVIDIA Container Toolkit

标准的 Docker 是不支持 GPU 访问的，为此 NVIDIA 提供了 NVIDIA Container Toolkit，在 docker 上做了一层封装，使得 docker 可以访问 GPU。根据官方文档 [INSTALLING DOCKER AND THE DOCKER UTILITY ENGINE FOR NVIDIA GPUS](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html)，流程如下。

**Step 1:** Add the package repositories:

```shell
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

注意，这里很可能由于机器无法访问 <github.com>, <github.io>, <nvidia.github.io> 导致后续安装无法进行，可以通过在 `/etc/hosts` 中添加对应的 ip 地址解决：

```shell
192.30.255.112  github.com
192.30.255.112  raw.githubusercontent.com
185.199.108.153 github.io
185.199.111.153 nvidia.github.io
```

上述 ip 在我写这些文字的时候是有效的，但保不齐后面会变，建议使用前 [iP 或域名查询](https://site.ip138.com/) 这个工具查询一下。修改 /etc/hosts 后能够 ping 通就好了。

**Step 2:** install the `nvidia-container-toolkit` package:

```shell
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

**Step 3:** Restart the Docker service:

```shell
$ sudo systemctl restart docker
```

**Step 4:** 测试是否安装成功

```shell
#### Test nvidia-smi with the latest official CUDA image
$ sudo docker run --rm --gpus all nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04 nvidia-smi
```

$ sudo docker run --rm --gpus all 

`nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04` 这个是 NVIDIA 官方提供的 Docker 镜像，其他版本可以在 [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/tags) 这个页面找到。

### 2.2 在 Docker 内配置好代码环境

#### 2.2.1 启动 Docker 镜像

在我们上一步测试 `nvidia-container-toolkit` 是否安装成功的过程中，docker 会自动下载 `nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04` 这个镜像，我们就基于这个镜像来配置我们的代码环境。

```shell
$ sudo docker run -it --gpus all -v /root/:/guazai nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04
```

稍微解释下这个命令：

- `-it`：交互式启动镜像
- `-v /root/:/guazai`：将宿主机的 `/root/` 目录挂载到 Docker 镜像的 `/guazai` 目录下；

#### 2.2.2 安装好代码环境

**Step 1：** 按照 2.1.1 修改 apt-get 源为国内镜像源的方法

**Step 2：** 安装必要的依赖

因为进入 Docker 镜像的是 root，我就不再加 sudo 命令了。

```shell
# 更新源
apt-get update
apt-get install -y build-essential cmake curl python3 python3-pip libjpeg-dev libpng-dev ca-certificates ffmpeg libsm6 libxext6 ninja-build git wget vim libglib2.0-0 libgl1-mesa-glx
```

`libglib2.0-0` 和 `libgl1-mesa-glx` 这两个必须安装，否则后面模型测试会报错。

**Step 3:** 安装 miniconda

```shell
# 以 Miniconda 官方网站上的下载链接和安装文件名为准
$ wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
# or
$ wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
$ sh Miniconda3-py38_4.12.0-Linux-x86_64.sh
# 让 CUDA 和 conda 生效
$ source ~/.bashrc
```

直接在 Docker 镜像中下载可能网速会比宿主机上下载要慢，可以在宿主机上下载到 /root 文件夹，然后在镜像中 cd 到 /guazai 文件夹下，就可以看到刚刚下载的文件了。

**Step 4:** 修改 conda 源和 pip 源

`vim ~/.condarc` 以及 `vim /.condarc` 添加以下内容，以换成北大源（比清华源快）：

北外源，清华大学开源软件镜像站的姐妹站，下载体验极佳

```shell
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud
  msys2: https://mirrors.bfsu.edu.cn/anaconda/cloud
  bioconda: https://mirrors.bfsu.edu.cn/anaconda/cloud
  menpo: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.bfsu.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.bfsu.edu.cn/anaconda/cloud
```

北大源：

```shell
channels:
    - defaults
show_channel_urls: true
default_channels:
    - https://mirrors.pku.edu.cn/anaconda/pkgs/main
    - https://mirrors.pku.edu.cn/anaconda/pkgs/r
custom_channels:
    conda-forge: https://mirrors.pku.edu.cn/anaconda/cloud
    pytorch: https://mirrors.pku.edu.cn/anaconda/cloud
    bioconda: https://mirrors.pku.edu.cn/anaconda/cloud
```

清华源：

```shell
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

然后再运行 `conda clean -i` 清除索引缓存，保证用的是镜像站提供的索引。

接着是 pip 源，以下源可以选用其一。个人实测，华为的源比清华快很多很多（10-20 倍），但是更新比较慢，可能个别模块的特定版本没有。所以下载特别大的模块的时候，可以先用华为的源，如果没有再再换成清华的源。

```shell
# 清华：
$ pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 阿里：
$ pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 华为：
$ pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple
# 豆瓣：
$ pip config set global.index-url https://pypi.douban.com/simple
```

**Step 5：**创建虚拟环境

```shell
$ conda create --name jittor python=3.8
$ source activate jittor
```

虽然 Jittor 文档和 JDet 文档的安装示例都是 Python 3.7，但是我在实际测试的时候发现必须是 3.8 或者 3.9 才能正常安装，否则会报错。

**Step 6：**安装 Jittor

根据官方文档 <https://cg.cs.tsinghua.edu.cn/jittor/download/>:

```shell
$ sudo apt install -y libomp-dev openmpi-bin openmpi-common libopenmpi-dev
$ python -m pip install jittor
# 手动为 Jittor 安装 CUDA
$ python -m jittor_utils.install_cuda
$ python -m jittor.test.test_example
# 如果您电脑包含Nvidia显卡，检查cudnn加速库
$ python -m jittor.test.test_cudnn_op
```

到这一步，我感觉我不该选一个啥都没有的 docker 镜像开始，而应该是在别人配置好 cuda、conda 之类的 docker 镜像之上，比如 pytorch 的官方 docker 镜像 [pytorch/manylinux-cuda113](https://hub.docker.com/r/pytorch/manylinux-cuda113/tags). 这样可以省去很多之前的步骤。

**Step 7：**准备好代码

如果是自己的 repo，需要先把 repo 代码挪到宿主机的 /root 目录下，也就是将其挂载到镜像中的 /guazai 目录下，然后再将其 cp 到镜像的 /root 目录下。如果是官方的 JDet，那么直接 git clone 即可。后面的步骤是一样的：

```shell
# Install the requirements
git clone https://github.com/Jittor/JDet # 自己 repo 不需要这一行
cd JDet
python -m pip install -r requirements.txt
# Install repo
cd JDet
# suggest this
python setup.py develop
```

**Step 8:** 准备好数据

在本地机器上将数据上传到宿主机：

```shell
scp -i <xxx.pem> ~/Downloads/testa-3.zip root@<ip-address>:~
```

在宿主机上解压数据：

```shell
cd /root
unzip testa-3.zip
```

还需要将解压后的 testa-3 文件夹 rename 成 test。

**Step 9:** 模型测试

首先是准备好权重，将 ckpt 放到对应的 works_dir 的 checkpoints 之下；
其次是准备好数据，`vim ./configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_test.py`  修改一下三个数据路径参数：

```shell
source_fair_dataset_path='/guazai/data/test'
source_dataset_path='/guazai/data/testa_dota'
target_dataset_path='/guazai/data/testa_ms'
```

然后运行 `python tools/preprocess.py --config-file configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_test.py` 进行数据预处理。

然后怎么运行测试以及模型融合就是具体 repo 的事情了，这里就不再赘述了。

**Step 10:** 删除不需要的文件和文件夹

不需要的文件和文件夹包括：

1. 生成的预处理图片；

### 2.3 保存镜像并测试

**Step 1：** 首先退出容器

```shell
# 退出容器
exit
```

**Step 2：** 查看容器 ID 并 commit 成镜像

```shell
$ sudo docker ps -a # Check the <CONTAINER ID>
$ sudo docker commit <CONTAINER ID> username/jittor:latest #  Create the docker image
```

这里的 `username/jittor:latest` 是镜像的名字，`latest` 是 tag 名称，都可以取成其他的。`username` 需要替换成自己实际的用户名.

**Step 3：** 保存镜像

保存镜像有两种方式，一种是 push to Docker Hub

```shell
$ sudo docker login
$ sudo docker push username/jittor:latest
```

另一种是 save to local

```shell
# docker images 获取 repositoryname:tag，Save the image as a tarball
$ docker save repositoryname:tag -o ~/username-jittor.tar
```

这里提一点，在 Mac 上，在硬盘空间明明是够的情况下，`docker save` 可能会遇到报错 `no space left on device`。原因在于 Docker for Mac 会有一个默认的总的 Disk image size 上限，默认是 60GB。如果你的镜像（我觉得应该是镜像 + 打包文件）超过了这个限制，就会报错。可以到 Docker for Mac 的 `Preferences -> Resources -> Advanced` 中修改这个上限, 我拉到 144GB 就不报错了。

**Step 4：** 测试镜像

我们先要删掉已有的 docker image，然后再 load 进刚才保存的镜像：

```shell
# docker remove image
$ docker image rm username/jittor:latest
# docker load image
$ docker load --input ~/username-jittor.tar
```

然后再启动容器：

```shell
$ sudo docker run -it --gpus all -v /root/:/guazai username/jittor:latest
```

## 3 其他

### 3.1 CUDA Toolkit (NVIDIA) 已经安装，但 NVCC 找不到

CUDA Toolkit (NVIDIA) 安装后需要进行环境配置：

`vim ~/.bashrc` 后添加如下内容：

```shell
export PATH="/usr/local/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH"
```

注意需要把 cuda-10.1 换成你安装的具体 CUDA 版本号。保存退出后，执行 `source ~/.bashrc` 使配置生效。再输入 `nvcc --version` 即可查看 `nvcc` 版本。

这里顺带解释一下 `PATH`、`LIBRARY_PATH`、`LD_LIBRARY_PATH` 这三者的概念以及区别：

首先是 `PATH` **环境变量**，用于指定可执行文件的搜索路径，多个路径之间用**冒号**分隔。例如，我们常用的 `ls`、`top`、`ps` 等命令就是系统先通过 `PATH` 找到了这些命令执行文件的所在位置（`/usr/local/bin`），再 run 这个命令（可执行文件）。如果搜索不到输入的命令名称，则会报错。添加新的可执行文件的方式如下：

```shell
# vim ~/.bashrc
PATH=$PATH:~/mycode/bin
# source ~/.bashrc 生效
```

其次是 `LIBRARY_PATH` 和 `LD_LIBRARY_PATH` 这两个路径，可以放在一起讨论，其中：

- `LIBRARY_PATH` 是**程序编译期间**查找动态链接库时指定查找共享库的路径；
- `LD_LIBRARY_PATH` 是**程序加载运行期间**查找动态链接库时指定除了系统默认路径之外的其他路径。

两者的共同点是**库**，库是这两个路径和 `PATH` 路径的区别，**PATH** 是**可执行文件**。两者的差异点是使用时间不一样。一个是**编译期**，对应的是**开发阶段**，如 gcc 编译；一个是**加载运行期**，对应的是**程序已交付的使用阶段**。配置方法也是类似：

```shell
# vim ~/.bashrc
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:XXXX
# source ~/.bashrc 生效
```

### 3.2 docker 从一个容器中 exit 后，怎么再进入这个容器？

先用 `docker ps -a` 找到对应的已经停止了的容器 id，然后用 `docker start -ia <container-id>` 启动容器

### 3.3 如何查看当前环境的 CUDA 版本

在回答这个问题前，首先要搞清楚究竟要查看 CUDA 什么的版本？目前，有三种通常意义上的查看 CUDA 版本的方式：

- `nvcc --version`
- `cat /usr/local/cuda/version.txt`
- `nvidia-smi`

我们从图 5 中可以知道，nvcc 返回的是 CUDA Runtime 的版本，属于狭义的 CUDA Toolkit 层面；
而 nvidia-smi 返回的是 CUDA Driver 的版本，属于 Driver 层面，这两者的 CUDA 版本是可以不一致的。而 `cat /usr/local/cuda/version.txt` 是直接查看对应 cuda 的版本文件，我感觉是 CUDA Toolkit 的版本。

其实 nvcc 返回的是 CUDA Toolkit 的版本也很容易理解，因为 nvcc 的全称为 NVIDIA's CUDA Compiler，是个编译器，用于 building application 的，所以属于 CUDA Toolkit 层面的版本。
而 nvidia-smi 的全称是 NVIDIA System Management Interface，是一个基于 NVIDIA Management Library (NVML) 构建的命令行工具，旨在帮助管理和监控 NVIDIA GPU 设备，从图 5 可以知道其调用的是 Driver API，所以返回的是 CUDA Driver 的版本，属于 Driver 层面的版本。

由于在一些机器上，Driver 和（狭义的）CUDA Toolkit 不是由 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archivE) 来一起安装的，而是独立安装了 GPU driver 和 CUDA Toolkit。这就会导致 nvcc 返回的 CUDA 版本与 nvidia-smi 返回的 CUDA 版本不一致，比如 nvcc 返回的是 9.2，而 nvidia-smi 返回的是 10.0。
在了解了 Runtime API 和 Driver API 的区别后，这就很自然了。

- 用于支持 driver API 的必要文件 (如 libcuda.so) 是由 GPU driver installer 安装的。nvidia-smi 就属于这一类 API。
- 用于支持 runtime API 的必要文件 (如 libcudart.so 以及 nvcc) 是由狭义的 CUDA Toolkit installer 安装的。nvcc 是与 CUDA Toolkit 一起安装的 CUDA compiler-driver tool，它只知道它自身构建时的 CUDA runtime 版本。它不知道安装了什么版本的 GPU driver，甚至不知道是否安装了 GPU driver。

因此，如果 driver API 和 runtime API 的 CUDA 版本不一致可能是因为你使用的是单独的 GPU driver installer，而不是完整的 CUDA Toolkit installer 里的 GPU driver installer。

### 3.4 Conda 的 cudatoolkit

通过 `conda` 安装 `cudatoolkit` 包含的库文件在 `~/miniconda3/lib` 中, `conda` 的 `cudatoolkit` 只包含 pytorch 或其他框架（ tensorflow、xgboost、Cupy）会使用到的 so 库文件。
因此，当我们使用 conda 时，实际上是不需要安装狭义的 CUDA Toolkit 的, 只需要 driver 和 conda 里面的 cudatoolkit。

### 3.5 runtime API 与 driver API

CUDA 主要有两个 API：runtime API、driver API

- 用于支持 driver API 的必要文件 (如 libcuda.so) 是由 GPU driver installer 安装的。
- 用于支持 runtime API 的必要文件 (如 libcudart.so 以及 nvcc) 是由（狭义的）CUDA Toolkit installer 安装的。
- nvidia-smi 属于 driver API、nvcc 属于 runtime API。
- nvcc 属于 CUDA compiler-driver tool，只知道 runtime API 版本，甚至不知道是否安装了 GPU driver。

## 参考资料

- [一文讲清楚 CUDA、CUDA toolkit、CUDNN、NVCC 关系](https://blog.csdn.net/qq_41094058/article/details/116207333)
- [显卡，显卡驱动，nvcc, cuda driver,cudatoolkit,cudnn 区别？](https://cloud.tencent.com/developer/article/1536738)
- [显卡，显卡驱动，nvcc, cuda driver,cudatoolkit,cudnn 到底是什么？](https://zhuanlan.zhihu.com/p/91334380)
- [cuda 已经安装，但 nvcc 找不到](https://zhuanlan.zhihu.com/p/338507526)
- [三种方法查看的 CUDA 版本均不同如何排查](https://zhuanlan.zhihu.com/p/438273611)
- [NVIDIA 驱动程序下载](https://www.nvidia.cn/Download/Find.aspx?lang=cn)
- [Docker saving and loading images](https://gist.github.com/developerinlondon/8a9dc6060f933c724fc6)
- [docker save](https://docs.docker.com/engine/reference/commandline/save/)
- [Dockerfile 编写指南](https://zhuanlan.zhihu.com/p/105885669)
- [如何编写最佳的 Dockerfile](https://zhuanlan.zhihu.com/p/26904830)
- [看完这篇，再也不用担心不会写 dockerfile 了](https://zhuanlan.zhihu.com/p/340550675)
- [北京大学 - Anaconda 镜像使用指南](https://mirrors.pku.edu.cn/Help/Anaconda)
