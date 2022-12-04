# learning-reid

The purpose of this repo is two-fold. First is to give an overview of `torchreid` package, which makes training and evaluating ReID model quite convenient. An overview of how to train a model using this framework is shown in `notebooks/train.ipynb` as well as how to evaluate a pre-trained model in `notebooks/eval.ipynb`. The second part of this repo focuses on using a pre-trained `torchreid` model to perform person ReID on a test video or set of frames.


## Data

Tsere are specific instructions as to what data one needs to upload in the notebook themselves. For our main script however, you must download the `View_001` camera frames from S2 L2 Walking dataset found in https://cs.binghamton.edu/~mrldata/pets2009.

    cd <root>/data
    wget https://cs.binghamton.edu/~mrldata/public/PETS2009/S2_L2.tar.bz2
    tar -xvjf S2_L2.tar.bz2


## Environment

We use conda to install necessary dependencies. The most important package we need to install is `torchreid`, which has been cloned inside this repo for your convenience. Run the following commands (found in the official readme [here](https://github.com/KaiyangZhou/deep-person-reid)).

    # cd to your preferred directory and clone this repo
    git clone https://github.com/KaiyangZhou/deep-person-reid.git

    # create environment
    cd deep-person-reid/
    conda create --name torchreid python=3.7
    conda activate torchreid

    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop
    # if you find yourself with a distutils version error, downgrade setuptools
    pip install setuptools==59.5.0


Also, make sure to install tensorflow

    pip install tensorflow


## Models

There are two models you must download in order to run the main script.

- Inception model (`faster_rcnn_inception_v2_coco`): can download using this [link](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
- ReID model (`osnet_ain_x1_0`): model weights (look under multi-source domain generatlization and download the `MS+D+C->M` model version using this [link](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html))


## How to run

Once you have modified the config params in `confs/config.yml`, run the following.

    PYTHONPATH=./ python src/main.py


There are few steps happening in this script:
- We instantiate two model classes (inception to detect humans; reid to create embeddings or latent features)
- For each frame, we generate human detection bounding boxes which are then used to crop the frame
- Each crop is then further processed to generate feature vector using our reid model
- In order to reid a query, we choose a query frame where the first detection is chosen as our query feature
- Then, all the frames appearing afterwards are considered target features
- We compute the Euclidean distance of the query feature against the target features and rank by increasing order

Here are some conclusions I drew from running the main script.
- Using a model trained across different domains (i.e. Market1501, DukeMTMC-ReID, and MSMT17), made the embedding more robust regardless of the subject's pose and movement.
- The choice of the inception model was mainly due to the total size; needed something light to run the whole pipeline smoothly without a GPU (i.e. running 100 frames takes a few minutes).
- One interesting next step would be to look at nonoverlapping camera view angles and test how well the reid model performs.


## Notebooks

There are two notebooks (`train.ipynb` and `eval.ipynb`) in `notebooks/`. These are intended to be run as colab notebooks to make use of gpu resources. Successfully trained a ReID model on Market1501 data. Also, successfully evaluated the pre-trained model `osnet_x1_0` (MS+D+C->M version). For more detailed instructions as to how to run and setup paths, please go to the notebooks themselves.