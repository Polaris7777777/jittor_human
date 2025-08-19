FROM docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 使用国内源加速 apt
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list \
 && sed -i 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list \
 && apt update && apt install -y software-properties-common  \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt update && apt install -y \
    build-essential cmake libomp-dev \
    wget curl git unzip vim \
    python3.9 python3.9-dev python3.9-venv libcudnn8 libcudnn8-dev\
 && apt purge -y python3.10 python3.10-minimal python3.10-dev \
 && rm -rf /var/lib/apt/lists/*

# 设置 Python 3.9 为默认
RUN ln -sf /usr/bin/python3.9 /usr/bin/python \
 && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
 && python -m ensurepip \
 && ln -sf /usr/bin/pip3 /usr/bin/pip

# pip 使用清华源
RUN python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 升级 pip 并安装 Jittor（GPU）
RUN python -m pip install --upgrade pip \
 && python -m pip install jittor -f https://cg.cs.tsinghua.edu.cn/jittor/assets/build/jtcuda.html

# 安装项目依赖
COPY requirements.txt /tmp/
RUN python -m pip install --no-cache-dir --ignore-installed -r /tmp/requirements.txt \
 && rm -rf /root/.cache/pip

# 修改 compile_extern.py 中的 URL
RUN sed -i '459s|.*|    url = "https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip"|' \
    /usr/local/lib/python3.9/dist-packages/jittor/compile_extern.py

# 拷贝项目代码
COPY . /workspace/project
WORKDIR /workspace/project

# # 设置环境变量
# ENV CUDNN_INCLUDE_DIR=/usr/include \
#     CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu \
#     LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV nvcc_path="/usr/local/cuda/bin/nvcc"
RUN echo 'export nvcc_path="/usr/local/cuda/bin/nvcc"' >> /root/.bashrc

CMD ["bash"]
