# Protest_Monitor

## My Goal
Identifying and analyzing protests through data from Instagram

![Location Summary Example](data/readme_images/location%20summary.png)

![Protest Example](data/readme_images/Protest%20example.png)

## How?
1. Retrieving images taged with a certain geo-location from Instagram 
2. Classifying them using a machine learning model
3. Gather information about the protests: signs, amount of people, etc.
4. Building a Web Page the summarises the researched location.
 
![Reseach Sceme](data/readme_images/Reseach%20scheme.png)


## Requirements
* Python 3.7
    * [Opencv-python 4.2.0.34](https://opencv.org/)
    * [Torch 1.5.0](https://pytorch.org/)
    * [Numpy 1.18.4](https://numpy.org/)
    * [Flask 1.1.2](https://flask.palletsprojects.com/en/1.1.x/)
    * [Matplotlib 3.2.1](https://matplotlib.org/)
    * [Jinja 2.10.3](https://jinja.palletsprojects.com/en/2.11.x/)
    * [Pillow 7.0.0](https://pillow.readthedocs.io/en/stable/)
    * [Pytesseract 0.3.4](https://opensource.google/projects/tesseract)
    * [Instaloader 4.4.2](https://instaloader.github.io/)
    * [Werkzeug 0.16.0](https://werkzeug.palletsprojects.com/en/1.0.x/)
    * [Imutils 0.5.3](https://pypi.org/project/imutils/)
    * [Spellchecker 0.5.4](https://pypi.org/project/pyspellchecker/)
    * [Shapely 1.7.0](https://pypi.org/project/Shapely/)
    * [Bentley-Ottmann 0.6.0](https://pypi.org/project/bentley-ottmann/) 

* Download models' weights 
    * [Yolov3-416](https://pjreddie.com/media/files/yolov3.weights)
    * [Protest Detector](https://www.dropbox.com/s/rxslj6x01otf62i/model_best.pth.tar?dl=0)
    * [Crowd Detection](https://drive.google.com/file/d/1KY11yLorynba14Sg7whFOfVeh2ja02wm/view)
* Update 'search_protests\constants.py' paths


## Sources
[Joseph Redmon, et al. "YOLOv3: An Incremental Improvement." (2018).](https://arxiv.org/pdf/1804.02767.pdf)
[Yuhong Li, et al. "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes." (2018).](https://arxiv.org/pdf/1802.10062.pdf)
[Donghyeon Won, et al. "Protest Activity Detection and Perceived Violence Estimation from Social Media Images." (2017).](https://arxiv.org/pdf/1709.06204.pdf)
