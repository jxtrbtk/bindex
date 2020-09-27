# bindex
Binance DEX Market Making

Machine Learning Docker Container on Debian Buster
- PyTorch with GPU support
- XGBoost, CatBoost, LightGBM
- TPot

sudo docker build -t machine:alpha.1 docker/machine/
sudo docker run -it --gpus device=1 machine:alpha.1
