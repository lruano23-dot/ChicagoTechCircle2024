# ChicagoTechCircle2024

This repo contains the functions I authored during my time as a Tech Consultant with Chicago Tech Circle and Argonne National Laboratory in Summer 2024.
There are 4 funtions in total consisting of two area detect functions, probe detection function, and coordinate difference function. These functions are 
mostly utilizing OpenCV and python to see shapes or patterns in given images. 

Area Detection Functions:
Detects the amount of tether left within a given image captured at the time. Using preprocessing, the area is calculated by coverting the image to black and white using a threshold function and counts the amount of black pixels present, divides it against the total amount of pixels present, then is turned into a percentage. There are 2 versions of this function. The non color version coverts the image to black and white and uses the process that was just explained. The colored version uses an otsu threshold in order to more accurately shade the image.

Probe Detection Function:
This function works by implementing a threshold on the image so that the edges of the probes can be seen and outlined. Once the outlines are placed the vertexes for each triangular probe can be found. The function then cycles through the 3 vertexs in order to find which ones are the innermost points in the picture and declares those as the tips of the probes.

Coordinate Difference Function:
This works by first calling the probe detection function and the square corners function in order to get the coordinates of interest. Once those are acquired each pair of coordinates of each tip probe is subtracted from it's corresponding square corner to see the differnce in location of the two.
