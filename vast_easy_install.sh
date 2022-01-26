
# intended to work inside a vast.ai instance
# if not using vast.ai update your Docker container with the below

# install update and gcc
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
apt install build-essential
apt-get install manpages-dev

# some useful dependencies
apt-get install git
apt-get install vim
apt-get install screen

# install gdown
pip install gdown

# clone yolor and cd into folder
git clone https://github.com/obravo7/yolor.git
cd yolor/

# install requirements
pip install cython
pip install opencv-python-headless
pip install -r requirements.txt

# download starfish dataset and extract
gdown --id 1Q8odmIiyzKa71ypcMgV32k-seYetJ9pa
tar xf starfish_dataset.tar.xz

# download yolor_p6.pt: pretrained coco model (for initial model)
gdown --id 1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76