# TCP Inference Server

> **Target platform:** DuckieBot — Jetson Nano, JetPack 4.6, ARM64, CUDA 10.2

SSH into your DuckieBot before running the commands below.

---

## Setup

### 1. Install system dependencies

```bash
sudo apt-get install curl libssl-dev libboost-python-dev libboost-thread-dev \
    cuda-libraries-10-2 cuda-toolkit-10-2
```

### 2. Add environment variables

```bash
cat >> ~/.bashrc << 'EOF'

export CUDA_HOME=/usr/local/cuda-10.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=/usr/src/tensorrt/bin:$PATH
export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH
EOF
```

### 3. Install pyenv

```bash
curl -fsSL https://pyenv.run | bash

cat >> ~/.bashrc << 'EOF'

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"
EOF

exec "$SHELL"
```

### 4. Set up the project

```bash
cd tcp-inference-server/
pyenv install 3.6.15 --verbose
pyenv local 3.6.15
```

### 5. Configure TensorRT

> TensorRT is pre-installed on JetPack — do **not** install it via pip. Just make it accessible:

```bash
echo 'export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### 6. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 7. Install Python dependencies

```bash
pip install -r requirements.txt
```

#### ONNX Runtime (optional)

```bash
pip install onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl "protobuf<=3.20.3"
```

### 8. Install pycuda

```bash
pip install Cython
echo 'export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}' >> venv/bin/activate
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> venv/bin/activate
echo 'export CUDA_HOME=/usr/local/cuda-10.2' >> venv/bin/activate
deactivate
source venv/bin/activate

pip download pycuda==2022.1 --no-deps -d /tmp/pycuda_src
cd /tmp/pycuda_src
tar xzf pycuda-2022.1.tar.gz
cd pycuda-2022.1
python configure.py --cuda-root=/usr/local/cuda-10.2 --cuda-inc-dir=/usr/local/cuda-10.2/include
pip install .
```

---

## Running

### TCP inference server

The server waits for a client connection, runs inference on received images, and returns the results.

```bash
cd tcp-inference-server/
source venv/bin/activate
python main_socket.py
```

### ROS driver (client)

The ROS node connects to the inference server, streams camera images, and receives model outputs. Build and run your ROS code like before:

```bash
# Build
dts devel build -H <ROBOT_NAME> -f

# Run
dts devel run -f -H <ROBOT_NAME>
```