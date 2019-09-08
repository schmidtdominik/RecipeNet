export LC_ALL=C
pip install jupyter
pip install -U pip
pip install tensorflow-gpu==2.0.0-rc0
pip install matplotlib
pip install requests

wget https://gist.githubusercontent.com/schmidtdominik/4d520346c6e5e528f51b332bb7bb8788/raw/0a332b580098c84003370fcdab2afc575252e3ff/dl_from_gdrive.py;
python3 dl_from_gdrive.py 1cnZrLd0ZOb-0M3v7iygDqjCHvyZCo7_U recipes.npz

tensorboard --logdir logs --port 6006 &

# [remote] jupyter notebook --ip=127.0.0.1 --port=8080 --allow-root &
# [local] ssh -N -f -L localhost:16006:localhost:6006 -p <port> root@<host>
