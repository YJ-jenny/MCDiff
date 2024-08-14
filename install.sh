# environment setting
conda create -n MRDiff python=3.11
conda activate MRDiff
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
