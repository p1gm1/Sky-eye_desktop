# Plant count for guava, passion fruit and palm oil

## Name: Sky-eye 

### Abstract: 

The goal is to use object detection and computer vision to determine the amount of coffe, guava, passion fruit and palm oil in a specific area. the images provided in this repo were captured by Drones. 

The model uses a neural network pretrained with the COCO dataset, and as a backbone uses a ResNet50 with FPN and bbox regressor in order to be able to count the number of plants that are in an hectare. 
The image dataset was collected through a multispectral camera that features a RGB camera, and cameras in the range of green, red. The images were captured from several fields in Colombia. 

-------------------------------------------------------
### Instructions

The program was made using python it doesn't require an extensive pc to operate, just run the main.py using the python engine, this will deploy the graphic interface which contains the instrucctions to operate, for you to do the rest. This version only works for UNIX based systems, in the next version I will be working on compatibility with windows. 

Make sure you have downloaded the models for each crop and save it in it's respective folder.

* oil_palm
* guava
* passion_fruit

--------------------------------------------------------

### Disclaimer

Currently the program only supports .JPG files, in the next version I will be working on compatibilty with .tiff and .tif files.