# DeepCell Spots

`deepcell-spots` is a deep learning library for spot detection. It is written in Python and built using [TensorFlow](https://github.com/tensorflow/tensorflow), [Keras](https://www.tensorflow.org/guide/keras) and [DeepCell](https://github.com/vanvalenlab/deepcell-tf).


## DeepCell-spots for Developers
Build and run a local docker container, similarly to the instructions for deepcell-tf. The relevant parts are copied here with modifications to work for deepcell-spots. For more elaborate instructions, see the [deepcell-tf README](https://github.com/vanvalenlab/deepcell-tf/blob/master/README.md).

### Build a local docker container, specifying the tensorflow version with TF_VERSION

```bash
git clone https://github.com/vanvalenlab/deepcell-spots.git
cd deepcell-spots
docker build --build-arg TF_VERSION=1.15.0-gpu -t $USER/deepcell-spots . 
```

### Run the new docker image

```bash
# '"device=0"' refers to the specific GPU(s) to run DeepCell-tf on, and is not required
docker run --gpus '"device=0"' -it \
-p 8888:8888 \
$USER/deepcell-spots
```

It can also be helpful to mount the local copy of the repository and the notebooks to speed up local development.

```bash
# you can now start the docker image with the code mounted for easy editing
docker run --gpus '"device=0"' -it \
    -p 8888:8888 \
    -v $PWD/deepcell-spots/deepcell_spots:/usr/local/lib/python3.6/dist-packages/deepcell_spots \
    -v $PWD/notebooks:/notebooks \
    -v /$PWD:/data \
    $USER/deepcell-spots
```


## Copyright

Copyright © 2019-2020 [The Van Valen Lab](http://www.vanvalen.caltech.edu/) at the California Institute of Technology (Caltech), with support from the Shurl and Kay Curci Foundation, Google Research Cloud, the Paul Allen Family Foundation, & National Institutes of Health (NIH) under Grant U24CA224309-01.
All rights reserved.

## License

This software is licensed under a modified [APACHE2](https://github.com/vanvalenlab/deepcell-spots/blob/master/LICENSE). See [LICENSE](https://github.com/vanvalenlab/deepcell-spots/blob/master/LICENSE) for full details.

## Trademarks

All other trademarks referenced herein are the property of their respective owners.

## Credits

[![Van Valen Lab, Caltech](https://upload.wikimedia.org/wikipedia/commons/7/75/Caltech_Logo.svg)](http://www.vanvalen.caltech.edu/)