import sys
from pypylon import pylon
from pypylon import genicam
import numpy as np
import os
import time
import h5py
from datetime import datetime
import concurrent.futures
import queue
import multiprocessing as mp
import re

def obtain_frame(q,q2,camera,num_frame,num_frame_per_file):
    starth = time.time()
    print('Obtain frame started')
    total_frame_ind = 0
    frame_ind = 0
    
    timestamps = [0]*num_frame

    while total_frame_ind < num_frame:
        if not camera.IsGrabbing():
            break

        if total_frame_ind % 100 == 1:
            print('Acquiring frame # ' + str(total_frame_ind))
        
        # first value is wait time before time out (in ms)
        grabResult = camera.RetrieveResult(
            30000, pylon.TimeoutHandling_ThrowException)
        
        # Access the chunk data attached to the result.
        # Before accessing the chunk data, you should check to see
        # if the chunk is readable. When it is readable, the buffer
        # contains the requested chunk data.
        timestamps[total_frame_ind] = grabResult.ChunkTimestamp.Value
        
        img = grabResult.GetArray()
        
        frame_ind = frame_ind + 1
        total_frame_ind = total_frame_ind + 1

        q.put(img)

        if frame_ind == num_frame_per_file:
            frame_ind = 0

            # query the image
            q.put(img)

    endh = time.time()
    print('Obtained all frames in ' + str(endh-starth) + " seconds.\n")

    # query the time stamps
    q2.put(timestamps)

def analyze(q,q2,camera_ind,num_frame,num_frame_per_file,save_folder,bit_depth,image_y,image_x):
    print('Analysis started')
    
    num_saved_frame = 0
    file_ind = 0
    frame_ind = 0
    total_frame_ind = 0

    if int(re.findall(r'\d+', bit_depth)[0]) > 8:
        mean_I = np.zeros((num_frame_per_file,),dtype=np.float64)
    else:
        mean_I = np.zeros((num_frame_per_file,),dtype=np.float64)
    
    while total_frame_ind < num_frame:

        if total_frame_ind % 100 == 1:
            print('Analyzing frame # ' + str(total_frame_ind))

        if total_frame_ind == 0:
            starth = time.time()
        
        #start = time.time()
        img = q.get()
        queue_size = q.qsize()
        #end = time.time()

        # make an array of images
        img = img.astype('float64')
        mean_I[frame_ind] = np.mean(img)

        frame_ind = frame_ind + 1
        total_frame_ind = total_frame_ind + 1

        if frame_ind >= num_frame_per_file:
            save_file = 'imgs_camera' + str(camera_ind) + '_file' + str(int(file_ind)) + '.h5'
            save_full = os.path.join(save_folder, save_file)

            start = time.time()
            hf = h5py.File(save_full, 'w')
            hf.create_dataset('mean_I', data=mean_I,maxshape=(num_frame_per_file,))
            hf.close()
            del hf
            end = time.time()

            num_saved_frame = num_saved_frame + len(mean_I)
            print('Saved camera #' + str(int(camera_ind)) + ' frame #' + str(int(num_saved_frame)) + ": " + str(end - start) + " seconds. " + str(queue_size) + " in queue.\n")

            file_ind = file_ind + 1
            frame_ind = 0

            if int(re.findall(r'\d+', bit_depth)[0]) > 8:
                mean_I = np.zeros((num_frame_per_file,),dtype=np.float64)
            else:
                mean_I = np.zeros((num_frame_per_file,),dtype=np.float64)
        
    endh = time.time()
    print('Saved all frames in ' + str(endh-starth) + " seconds.\n")

    while q2.qsize() == 0:
        pass

    # obtain time stamps of the frames
    timestamps = [q2.get() for _ in range(q2.qsize())]
    timestamps = timestamps[-1]

    # save the time stamps to a separate HDF5 file
    save_file = 'imgs_camera' + str(camera_ind) + '_timestamps.h5'
    save_full = os.path.join(save_folder, save_file)
    hf = h5py.File(save_full, 'w')
    hf.create_dataset('timestamps', data=timestamps) 
    hf.close()
    return file_ind

def acquire(camera_ind,use_trigger,bit_depth,gain,black_level,exp_time,frame_rate,image_y,image_x,num_frame,num_frame_per_file,save_folder):
    # process priority
    # p = psutil.Process(os.getpid())
    # p.nice(psutil.HIGH_PRIORITY_CLASS)

    # create a queue
    q = queue.Queue()
    q2 = queue.Queue()

    # Get the transport layer factory.
    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
    device = devices[camera_ind]

    camera = pylon.InstantCameraArray(1)
    for i, cam in enumerate(camera):
        cam.Attach(tlFactory.CreateDevice(device))

        # Print the model name of the camera.
        print("Using device " + cam.GetDeviceInfo().GetModelName() + ", S/N: " + cam.GetDeviceInfo().GetSerialNumber() + '\n')

    camera[0].Close()
    camera[0].Open()

    # A GenICam node map is required for accessing chunk data. That's why a small node map is required for each grab result.
    # Creating a node map can be time consuming, because node maps are created by parsing an XML description file.
    # The node maps are usually created dynamically when StartGrabbing() is called.
    # To avoid a delay caused by node map creation in StartGrabbing() you have the option to create
    # a static pool of node maps once before grabbing.
    camera[0].StaticChunkNodeMapPoolSize = camera[0].MaxNumBuffer.GetValue()

    # Enable chunks in general.
    if genicam.IsWritable(camera[0].ChunkModeActive):
        camera[0].ChunkModeActive = True
    else:
        raise pylon.RuntimeException("The camera doesn't support chunk features")
    
    # Enable time stamp chunks.
    camera[0].ChunkSelector = "Timestamp"
    camera[0].ChunkEnable = True

    # Exposure time
    camera[0].ExposureTime.SetValue(exp_time)

    # Size
    camera[0].Width.SetValue(image_x)
    camera[0].Height.SetValue(image_y)

    # Set bit depth
    camera[0].PixelFormat.SetValue(bit_depth)

    # Set gain
    camera[0].Gain.SetValue(gain)

    # Set black level
    camera[0].BlackLevel.SetValue(black_level)

    # Trigger mode
    if use_trigger:
        camera[0].TriggerMode.SetValue('On')
        camera[0].TriggerSource.SetValue('Line2')
        camera[0].TriggerActivation.SetValue('RisingEdge')
    else:
        camera[0].TriggerMode.SetValue('Off')
        camera[0].AcquisitionFrameRateEnable.SetValue(True)
        # Set frame rate (only applicable if using internal trigger)
        camera[0].AcquisitionFrameRate.SetValue(frame_rate)

    camera.StartGrabbing()
    time.sleep(0.1)
    start = time.time()

    # create a thread pool with 2 threads
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    # create a separate thread for obtaining frames from camera
    pool.submit(obtain_frame,q,q2,camera,num_frame,num_frame_per_file)

    # create a separate thread for querying the frames from the above thread and analyzing
    pool.submit(analyze,q,q2,camera_ind,num_frame,num_frame_per_file,save_folder,bit_depth,image_y,image_x)
    
    # wait for all tasks to complete
    pool.shutdown(wait=True)

    end = time.time()

    print('Total time taken (seconds) : ' + str(round((end - start))) + "\n")
    camera[0].Close()
    
    return True


if __name__ == "__main__":
    os.environ["PYLON_CAMEMU"] = "3"

    # bit depth
    #bit_depth = 'Mono8'
    bit_depth = 'Mono10p'
    #bit_depth = 'Mono12p'

    # gain
    gain = 16

    # black level
    black_level = 25

    # exposure time in microseconds
    exp_time = 1000

    # use trigger
    use_trigger = False

    # frame rate (Hz)
    frame_rate = 120

    # Number of images to be grabbed.
    num_frame_per_camera = 120*60*60

    # image size
    image_y = 1216
    image_x = 1936

    # Limits the amount of cameras used for grabbing.
    # It is important to manage the available bandwidth when grabbing with multiple cameras.
    num_camera = 1

    num_frame_total = num_frame_per_camera*num_camera

    # max number of frames per file
    num_frame_per_file = 120*60

    # array of save folders. One for each camera
    save_folders = ['C:\\temp\\20231003\\10bit16db25blacklevel1000us']

    # make folder if it does not exist
    for save_folder in save_folders:
        save_folder_exists = os.path.exists(save_folder)
        if not save_folder_exists:
            os.makedirs(save_folder)
            print("Save folder created.")

    # The exit code of the sample application.
    exitCode = 0

    try:
        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()

        # list of devices
        for device in devices:
            print(device.GetFriendlyName())

        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        time.sleep(0.1)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

        start = time.time()

        # make multiprocessing nodes
        p = [0]*num_camera
        for i in range(0,num_camera):
            save_folder = save_folders[i]
            print(save_folder)
            p[i] = mp.Process(target=acquire,args=(i,use_trigger,bit_depth,gain,black_level,exp_time,frame_rate,image_y,image_x,num_frame_per_camera,num_frame_per_file,save_folder,))

        # run the new process
        for i in range(0,num_camera):
            p[i].start()

        # join the new process to main process
        for i in range(0,num_camera):
            p[i].join()

        end = time.time()

        print('Total time taken (seconds) : ' + str(round((end - start))) + "\n")

    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", e.GetDescription())
        exitCode = 1

    # Comment the following two lines to disable waiting on exit.
    sys.exit(exitCode)
