# burnout streamlit interface


## Install pyenv & pipenv
```sh
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zprofile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zprofile
echo 'eval "$(pyenv init --path)"' >> ~/.zprofile

echo 'eval "$(pyenv init -)"' >> ~/.zshrc

$ git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
	
```

## Set Env
```sh
#pipenv --python 3.8.2 --verbose install
pipenv --python 3.8.7 --verbose install

. ~/.local/share/virtualenvs/burnout-kH9yOvj2-python/bin/activate
make deps/update
```

## Persiapan Data
1. Buat folder data pada parrent folder, simpan video test.mp4
```sh
(burnout) ➜  burnout git:(main) ✗ cd data 
(burnout) ➜  data git:(main) ✗ tree
└── test.mp4
```
2. Buat folder models simpan semua model yang telah dicompile di TF pada folder models
```sh
(burnout) ➜  burnout git:(main) ✗ cd models 
(burnout) ➜  models git:(main) ✗ tree
.
├── checkpoint
│   ├── checkpoint
│   ├── ckpt-0.data-00000-of-00001
│   └── ckpt-0.index
├── labelmap.pbtxt
├── pipeline.config
└── saved_model
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

## Install models dari tensorflow

```sh
export PYTHONPATH=/home/balaplumpat/Projects/Streamlit/burnout/tfod-api:/home/balaplumpat/Projects/Streamlit/burnout/tfod-api/research:/home/balaplumpat/Projects/Streamlit/burnout/tfod-api/research/slim

git clone -b b1d973e https://github.com/tensorflow/models.git
# git clone -b b1d973ea103ee50d0933efe3c10136908b9909d9 https://github.com/tensorflow/models.git
git clone https://github.com/tensorflow/models tfod_api

```

```sh
sudo apt-get install protobuf-compiler

cd tfod_api/research
touch use_protobuf.py

import os
import sys
args = sys.argv
directory = args[1]
protoc_path = args[2]
for file in os.listdir(directory):
    if file.endswith(".proto"):
        os.system(protoc_path+" "+directory+"/"+file+" --python_out=.")

python use_protobuf.py  ./object_detection/protos/ /usr/bin/protoc
cp object_detection/packages/tf2/setup.py .

# cek model berjalan
python ./object_detection/builders/model_builder_tf2_test.py

```

## Execute Streamlit
```sh
export STREAMLIT_SERVER_PORT=8080
streamlit run --server.port $STREAMLIT_SERVER_PORT app.py
```

## Streamlit timeout error

ubah timeout dari 10 menjadi 100 line 438 pada virtualenvs/{burnout-kH9yOvj2-python}/lib/python3.8/site-packages/streamlit_webrtc/webrtc.py

```python
self, sdp, type_, timeout: Union[float, None] = 10.0

self, sdp, type_, timeout: Union[float, None] = 100.0
```

## Streamlit Ngorx Google Collabs
[link1](https://medium.com/@jcharistech/how-to-run-streamlit-apps-from-colab-29b969a1bdfc)  
[link2](https://gist.github.com/tuffacton/da5a9b42c0a2e9e355353689f93c84b3)  
[link3](https://gist.github.com/MrFCow/8e0b497755d1f9ff1cf330f4af800911#file-instagrammatch-removed-core-functions-ipynb)  
[link4](https://github.com/napoles-uach/streamlit_apps/blob/main/Streamlit_Colab/06_Streamlit__Colab_BrainTumor.ipynb)  


## Misc

### github
```
git remote add origin https://github.com/trisgelar/burnout.git
git branch -M main
git push -u origin main

```
