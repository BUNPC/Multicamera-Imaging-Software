# Multicamera Imaging Software
 For multi-channel SCOS

# Install and Quickstart
1. Run shell script ./Basler cameras/setup.sh to install the relevant Python modules.
2. Make sure Basler Pylon library is installed. Installing Basler's Camera Software Suite should be sufficient.
3. Create a version of the parameters file by copying ./Basler cameras/save raw/parameters.json and modifying the contents.
4. Run ./Basler cameras/save raw/index_keep_file_open.py to start the python script that saves data from multiple Basler cameras at once.

# Debug
Q: Not all cameras are running. There seem to be a few cameras that are not acquiring frames according to the terminal feedback.
A: Close the Python program instance. Open up Pylon viewer to ensure all the cameras are detected by the computer. Make sure the parameter file you use has the correct serial numbers. If using trigger ("use trigger" : true), then make sure the cameras are receiving TTL trigger signals.