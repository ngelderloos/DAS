# Driver Awareness Sensor (DAS) #

## System Setup ##

### Windows 7 ###

* Install Python 2.7
	1. [http://www.anthonydebarros.com/2011/10/15/setting-up-python-in-windows-7/](http://www.anthonydebarros.com/2011/10/15/setting-up-python-in-windows-7/)
	2. [https://pypi.python.org/pypi/setuptools](https://pypi.python.org/pypi/setuptools)

	
* Install OpenCV
	1. [http://stackoverflow.com/questions/4709301/installing-opencv-on-windows-7-for-python-2-7](http://stackoverflow.com/questions/4709301/installing-opencv-on-windows-7-for-python-2-7)
	2. [http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv](http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv) (OpenCV-Python installer)

	
* Install NumPy
	1. [http://sourceforge.net/projects/numpy/files/NumPy/1.7.0/](http://sourceforge.net/projects/numpy/files/NumPy/1.7.0/)

	
* Didn't use, but may be useful.
	1. [http://opencvpython.blogspot.in/2012/05/install-opencv-in-windows-for-python.html](http://opencvpython.blogspot.in/2012/05/install-opencv-in-windows-for-python.html)

    
## Helpful Commands ##

Press the following keys when the main window (not the terminal) is active.

* c - Toggles the video capture mode on/off. When on, a video file will be saved in the ./captures/ folder.
* p - Toggles the output to the screen. When paused, the window does not update with current video, though processing still occurs, as well as video capture if it is on.
* w - Toggles the window lock. The window is initially locked in location to prevent unauthorized movement.
* ESC - Exits the program.