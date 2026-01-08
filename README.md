conda create -n da3-slam python=3.10 -y
conda activate da3-slam
pip install xformers torch\>=2 torchvision

cd ../Depth-Anything-3/ && pip install -e .
cd ../DA3-SLAM/ && pip install -r requirements.txt


SLAM Solver
python ./main_slam.py --image_dir /home/zhouyi/repo/dataset/C3VD2/brightness

point cloud alignment