## Task3
This task is done using streamlit as i am able to add a slider to play with the window size & threshold values for most of the filters and see the results immediately in the browser.

## To run task3
Simply go to the folder and run run1.bat (Exp1.py) or run2.bat (Exp2.py) 
or 
run the code which will provide a command then copy it and run it in the terminal.

### Note: streamlit is needed to run this code (pip install streamlit)

#### Note: in Exp2 the user can select the seed points and the threshold value T. however since the image is gray scale, the region colors are not very visible.

#### The max color value is 255 (white) and the min color value is 0 (black) so as the no_of_splits increases the region colors become less visible.

#### Ex.
#### If 1 region is selected then the color of the region will be:
(255 - 0 * 255 / 1)
#### If 2 regions are selected then the color of the region will be:
(255 - 0 * 255 / 2) and (255 - 1 * 255 / 2)
#### If 3 regions are selected then the color of the region will be:
(255 - 0 * 255 / 3) and (255 - 1 * 255 / 3) and (255 - 2*255/3)