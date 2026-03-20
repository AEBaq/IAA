# Guide to setup on the DuckieBot (Jetson Nano, Jetpack 4.6, Arm 64, Cuda 10.2)

SSH to your DuckieBot to install the TCP inference server.

## install deps
sudo apt-get install curl libssl-dev libboost-python-dev libboost-thread-dev cuda-libraries-10-2 cuda-toolkit-10-2


## add env variables to .bashrc
cat >> ~/.bashrc << 'EOF'

export CUDA_HOME=/usr/local/cuda-10.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=/usr/src/tensorrt/bin:$PATH
export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH
EOF


## install pyenv
curl -fsSL https://pyenv.run | bash
cat >> ~/.bashrc << 'EOF'

## pyenv
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"
EOF
exec "$SHELL"


## setup project
cd tcp-inference-server/
pyenv install 3.6.15 --verbose
pyenv local 3.6.15


## tensorrt déjà installé par défaut; ne pas installer par pip; il suffit de le rendre accessible
echo 'export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc


## create venv
python3 -m venv venv
source venv/bin/activate


## install Python dependencies
pip install -r requirements.txt

### ONNX Runtime (if needed)
pip install onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl "protobuf<=3.20.3"


## install pycuda
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


# How to run

## ROS side

This ROS code is the client. It connects to the TCP Inference server, streams camera images, and waits for the model outputs.

Build the driver-client-node on the robot
dts devel build -H <ROBOT_NAME> -f

Run the driver-client-node on the robot
dts devel run -f -H <ROBOT_NAME>

## TCP Inference server
This code is the server. It waits for a client to connect, awaits images and runs model inference before sending back the ouputs to the client.

Activate the virtual environment
cd tcp-inference-server/
source venv/bin/activate

Start the inference server
python main_socket.py