# Plant count for coffee, guava, passion fruit and palm oil
Name: Drone Sky S.A.S <br />
Abstract: The goal is to use image segmentation to determine amount of coffe, guava, passion fruit and palm oil in a specific area. Will be utilized a aerial image captured by Drones. <br />
<br />
The objective of the work will be the separation of the target plants from the soil, that is, it separates the main object from the bottom, in order to be able to count the number of plants that are in an hectare. <br />
The image dataset was collected through a multispectral camera that features a RGB camera, and cameras in the range of green, red. The images were captured from several fields in Colombia. <br />
The methods that will be used for the separation of the components will be the segmentation of images and also filters for noise removal. First, the bands R, G and B will be separated to see how best to separate the soil from the plant, then a medium or a Gaussian filter will be used to eliminate noise. With the elimination of noise, segmentation through the threshold will be performed in the middle of the analysis of the histogram. After segmentation, the plants will be counted.