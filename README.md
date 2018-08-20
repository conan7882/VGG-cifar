# VGG19-FCN for Image Classification
 - TensorFlow implementation of [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556). 
 
 - The VGG19 model is defined in [`vgg.py`](vgg.py).

 
## Requirements
- Python 3.3+
- [Tensorflow 1.0+](https://www.tensorflow.org/)
- [TensorCV](https://github.com/conan7882/DeepVision-tensorflow)

<!--## TODO

- [x] Test pre-trained model
- [ ] Fine tuning-->


## Implementation Details

For testing the pre-trained model

- The last three fully connected layers are converted to convolutional layers making it a fully convolutional network. Then the input images can be arbitrary size.

- Images are rescaled so that the smallest side equals 224 before fed into the model.

- [Global average pooling](https://arxiv.org/abs/1312.4400) is used to get fix size of class scores for all the test images.

## Results

*Data Source* | *Image* | *Result* |
|:--|:--:|:--|
COCO |<img src='data/000000000285.jpg' height='200px'>| 1: probability: 1.00, label: brown bear, bruin <br>2: probability: 0.00, label: American black bear, black bear <br>3: probability: 0.00, label: ice bear, polar bear <br>4: probability: 0.00, label: sloth bear, Melursus ursinus <br>5: probability: 0.00, label: chow, chow chow
COCO |<img src='data/000000000724.jpg' height='200px'>| 1: probability: 0.61, label: street sign<br>2: probability: 0.27, label: traffic light, traffic signal, stoplight<br>3: probability: 0.02, label: mailbox, letter box<br>4: probability: 0.02, label: parking meter<br>5: probability: 0.01, label: pay-phone, pay-station
COCO |<img src='data/000000001584.jpg' height='200px'>|1: probability: 0.48, label: passenger car, coach, carriage<br>2: probability: 0.36, label: trolleybus, trolley coach, trackless trolley<br>3: probability: 0.10, label: minibus<br>4: probability: 0.02, label: school bus<br>5: probability: 0.01, label: streetcar, tram, tramcar, trolley, trolley car
COCO |<img src='data/000000003845.jpg' height='200px'>|1: probability: 0.17, label: burrito<br>2: probability: 0.13, label: plate<br>3: probability: 0.10, label: Dungeness crab, Cancer magister<br>4: probability: 0.06, label: mashed potato<br>5: probability: 0.06, label: guacamole
ImageNet |<img src='data/ILSVRC2017_test_00000004.jpg' height='200px'>|1: probability: 1.00, label: goldfish, Carassius auratus<br>2: probability: 0.00, label: rock beauty, Holocanthus tricolor<br>3: probability: 0.00, label: anemone fish<br>4: probability: 0.00, label: coral reef<br>5: probability: 0.00, label: puffer, pufferfish, blowfish, globefish
Self Collection | <img src='data/IMG_4379.jpg' height='200px'>|1: probability: 0.33, label: tabby, tabby cat<br>2: probability: 0.20, label: Egyptian cat<br>3: probability: 0.11, label: tiger cat<br>4: probability: 0.03, label: Cardigan, Cardigan Welsh corgi<br>5: probability: 0.02, label: bookcase
Self Collection | <img src='data/IMG_7940.JPG' height='200px'>|1: probability: 1.00, label: streetcar, tram, tramcarr<br>2: probability: 0.00, label: trolleybus, trolley coach<br>3: probability: 0.00, label: passenger car, coach, carriage<br>4: probability: 0.00, label: electric locomotive<br>5: probability: 0.00, label: minibus


## Usage
### Download pre-trained VGG19 model
Download the pre-trained parameters VGG19 NPY [here](https://github.com/machrisaa/tensorflow-vgg#tensorflow-vgg16-and-vgg19).
### Config path
All directories are setup in [`config.py`](config.py).

- Put the pre-trained paramenters in `config.model_dir`.
- Put testing images in `config.valid_data_dir`.



### ImageNet Classification
- Put test image in folder `config.valid_data_dir`, then run the script:

```
python VGG_FCN_pre_trained.py --type IMAGE_FILE_EXTENSION(.jpg or .png or other types of images)
```
       
   The output are the top-5 class labels and probabilities, and the top-1 human label.
   
## Author
Qian Ge
