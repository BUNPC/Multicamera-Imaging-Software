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
import PySimpleGUI as sg
import json

def obtain_frame(q,q2,camera_ind,camera,num_frame,num_frame_per_file):
    starth = time.time()
    print('Obtain frame started')
    total_frame_ind = 0
    frame_ind = 0
    start = time.time()
    timestamps = [0]*num_frame

    while total_frame_ind < num_frame:
        if not camera.IsGrabbing():
            break

        if total_frame_ind % 100 == 0:
            print('Acquiring camera # ' + str(camera_ind) + ' frame # ' + str(total_frame_ind) + ', ' + str(datetime.now()) + ', q = ' + str(q.qsize()) + "\n")
        
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

            #end = time.time()
            # print('Obtained frame #' + str(int(total_frame_ind)) + ': ' + str(end - start) + ' seconds.\n')
            #start = time.time()
    endh = time.time()
    print('Obtained all frames in ' + str(endh-starth) + " seconds.\n")

    # query the time stamps
    q2.put(timestamps)

def save_h5(q,q2,camera_ind,num_frame,num_frame_per_file,save_folder,bit_depth,image_y,image_x):
    print('Save H5 started')
    
    num_saved_frame = 0
    file_ind = 0
    frame_ind = 0
    total_frame_ind = 0

    #imgs = [0]*num_frame_per_file
    if int(re.findall(r'\d+', bit_depth)[0]) > 8:
        imgs = np.zeros((num_frame_per_file,image_y,image_x),dtype=np.uint16)
    else:
        imgs = np.zeros((num_frame_per_file,image_y,image_x),dtype=np.uint8)
    
    while total_frame_ind < num_frame:

        if total_frame_ind == 0:
            starth = time.time()
        
        img = q.get()
        queue_size = q.qsize()

        # make an array of images
        imgs[frame_ind,:,:] = img

        frame_ind = frame_ind + 1
        total_frame_ind = total_frame_ind + 1

        if frame_ind >= num_frame_per_file or total_frame_ind == num_frame:
            save_file = 'imgs_camera' + str(camera_ind) + '_file' + str(int(file_ind)) + '.h5'
            save_full = os.path.join(save_folder, save_file)

            start = time.time()
            hf = h5py.File(save_full, 'w')
            hf.create_dataset('imgs', data=imgs,chunks=(1,image_y,image_x), maxshape=(num_frame_per_file, image_y, image_x))
            hf.close()
            del hf
            end = time.time()

            num_saved_frame = num_saved_frame + len(imgs)
            print('Saved camera #' + str(int(camera_ind)) + ' frame #' + str(int(num_saved_frame)) + ": " + str(end - start) + " seconds. " + str(queue_size) + " in queue.\n")

            img_length = num_frame_per_file
            if num_frame - total_frame_ind < num_frame_per_file:
                img_length = num_frame - total_frame_ind

            if int(re.findall(r'\d+', bit_depth)[0]) > 8:
                imgs = np.zeros((img_length,image_y,image_x),dtype=np.uint16)
            else:
                imgs = np.zeros((img_length,image_y,image_x),dtype=np.uint8)

            file_ind = file_ind + 1
            frame_ind = 0
        
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
    return imgs, file_ind

def acquire(devices_sn,sn,camera_ind,use_trigger,bit_depth,gain,black_level,exp_time,frame_rate,image_y,image_x,num_frame,num_frame_per_file,save_folder):
    # process priority
    # p = psutil.Process(os.getpid())
    # p.nice(psutil.HIGH_PRIORITY_CLASS)

    # create a queue
    q = queue.Queue()
    q2 = queue.Queue()

    # find the right device index
    device_ind = [str(sn) == d for d in devices_sn]
    device_ind = [i for i, val in enumerate(device_ind) if val][0]

    try:
        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", e.GetDescription())
        exitCode = 1

    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    time.sleep(0.1)

    # choose device
    device = devices[device_ind]

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
    pool.submit(obtain_frame,q,q2,camera_ind,camera,num_frame,num_frame_per_file)

    # create a separate thread for querying the frames from the above thread and writing to H5 files
    pool.submit(save_h5,q,q2,camera_ind,num_frame,num_frame_per_file,save_folder,bit_depth,image_y,image_x)
    
    # wait for all tasks to complete
    pool.shutdown(wait=True)

    end = time.time()

    print('Total time taken (seconds) : ' + str(round((end - start))) + "\n")
    camera[0].Close()
    
    return True

def gui(num_camera):    
    sg.theme('DarkAmber')   # Add a touch of color
    layout = []
    layout += [sg.Text('Change # of cameras'), sg.In(key='camera_num'), sg.Button('Change')],
    layout += [sg.Text('Load parameters (.json)'), sg.In(size=(25,1), enable_events=True ,key='parameters'), sg.FileBrowse(), sg.Button('Fill')],
    
    for c_ind in range(num_camera):
        layout += [sg.Text('Camera ' + str(c_ind) + ':', font=("Helvetica", 12, "bold"))],
        layout += [sg.Text('SN'), sg.In(key='sn_' + str(c_ind))],
        layout += [sg.Text('Save folder'), sg.In(size=(25,1), enable_events=True ,key='folder_' + str(c_ind)), sg.FolderBrowse()],
        layout += [sg.Text('Frame number'), sg.In(key='frame_num_' + str(c_ind))],
        layout += [sg.Checkbox('Use trigger', key='trigger_' + str(c_ind))],
        layout += [sg.Text('Bit depth (Mono8, Mono10p, Mono12p)'), sg.In(key='bd_' + str(c_ind))],
        layout += [sg.Text('Frame rate (Hz)'), sg.In(key='fr_' + str(c_ind))],
        layout += [sg.Text('Exposure time (us)'), sg.In(key='et_' + str(c_ind))],
        layout += [sg.Text('Gain (dB)'), sg.In(key='gain_' + str(c_ind))],
        layout += [sg.Text('Black level'), sg.In(key='bl_' + str(c_ind))],
    layout += [[sg.Button('Run'), sg.Button('Cancel')]]

    layout = [
        [
            sg.Column(layout, scrollable=True,  vertical_scroll_only=True),
            sg.Column([])
        ]
    ]

    window = sg.Window('Camera parameters', layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED,'Run','Cancel'):
            window.close()
            break
        if event == 'Change':
            window.close()
            break
        if event == 'Fill':
            # Opening JSON file
            f = open(values['parameters'])

            # returns JSON object as 
            # a dictionary
            data = json.load(f)

            for c_ind in range(num_camera):
                values['sn_' + str(c_ind)] = data['camera ' + str(c_ind)]['sn']
                values['folder_' + str(c_ind)] = data['camera ' + str(c_ind)]['save folder']
                values['frame_num_' + str(c_ind)] = data['camera ' + str(c_ind)]['frame num']
                values['trigger_' + str(c_ind)] = data['camera ' + str(c_ind)]['use trigger']
                values['bd_' + str(c_ind)] = data['camera ' + str(c_ind)]['bit depth']
                values['fr_' + str(c_ind)] = data['camera ' + str(c_ind)]['frame rate']
                values['et_' + str(c_ind)] = data['camera ' + str(c_ind)]['exposure time']
                values['bl_' + str(c_ind)] = data['camera ' + str(c_ind)]['black level']
                values['gain_' + str(c_ind)] = data['camera ' + str(c_ind)]['gain']
            
            window.fill(values)
    
    if event == 'Cancel':
        exit()

    if event == 'Change':
        gui(int(values['camera_num']))

    return values

if __name__ == "__main__":
    os.environ["PYLON_CAMEMU"] = "3"

    # The exit code of the sample application.
    exitCode = 0

    try:
        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", e.GetDescription())
        exitCode = 1

    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    time.sleep(0.1)

    devices_sn = []
    for camera_ind in range(len(devices)):
        device = devices[camera_ind]

        camera = pylon.InstantCameraArray(1)
        for i, cam in enumerate(camera):
            cam.Attach(tlFactory.CreateDevice(device))
            devices_sn += [cam.GetDeviceInfo().GetSerialNumber()]
    
    # Limits the amount of cameras used for grabbing.
    # It is important to manage the available bandwidth when grabbing with multiple cameras.
    num_camera = 1

    values = gui(num_camera)

    sn = list(range(num_camera))
    num_frame_per_camera = list(range(num_camera))
    save_folders = list(range(num_camera))
    use_trigger = list(range(num_camera))
    bit_depth = list(range(num_camera))
    frame_rate = list(range(num_camera))
    exp_time = list(range(num_camera))
    gain = list(range(num_camera))
    black_level = list(range(num_camera))
    for c_ind in range(num_camera):
        sn[c_ind] = int(values['sn_' + str(c_ind)])
        num_frame_per_camera[c_ind] = int(values['frame_num_' + str(c_ind)])
        save_folders[c_ind] = values['folder_' + str(c_ind)]
        use_trigger[c_ind] = values['folder_' + str(c_ind)]
        bit_depth[c_ind] = values['bd_' + str(c_ind)]
        frame_rate[c_ind] = int(values['fr_' + str(c_ind)])
        exp_time[c_ind] = int(values['et_' + str(c_ind)])
        gain[c_ind] = int(values['gain_' + str(c_ind)])
        black_level[c_ind] = int(values['bl_' + str(c_ind)])

    # make folder if it does not exist
    for save_folder in save_folders:
        save_folder_exists = os.path.exists(save_folder)
        if not save_folder_exists:
            os.makedirs(save_folder)
            print("Save folder created.")
    
    # image size
    image_y = 1216
    image_x = 1936

    # max number of frames per file
    num_frame_per_file = 100

    try:
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
            p[i] = mp.Process(target=acquire,args=(devices_sn,sn[i],i,use_trigger[i],bit_depth[i],gain[i],black_level[i],exp_time[i],frame_rate[i],image_y,image_x,num_frame_per_camera[i],num_frame_per_file,save_folder,))

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
