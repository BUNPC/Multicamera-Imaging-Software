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
import psutil

def obtain_frame(q,q2,camera_ind,camera,num_frame,num_frame_per_file):
    starth = time.time()
    print('Obtain frame started for camera # ' + str(camera_ind))
    total_frame_ind = 0
    frame_ind = 0
    start = time.time()
    timestamps = [0]*num_frame

    while total_frame_ind < num_frame:
        if total_frame_ind == 0:
            t_start = time.time()

        if not camera.IsGrabbing():
            break

        if total_frame_ind % 100 == 0:
            print('Acquiring camera # ' + str(camera_ind) + ' frame # ' + str(total_frame_ind) + ', ' + str(time.time()-t_start) + ' seconds, q = ' + str(q.qsize()) + "\n")
        
        # first value is wait time before time out (in ms)
        grabResult = camera.RetrieveResult(
            300000, pylon.TimeoutHandling_ThrowException)
        
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

def save_h5(q,q2,camera_ind,num_frame,num_frame_per_file,num_frame_per_write,save_folder,bit_depth,image_y,image_x):

    print('Save H5 started for camera # ' + str(camera_ind))
    
    num_saved_frame = 0
    file_ind = 0
    frame_ind = 0
    total_frame_ind = 0

    starth = time.time()
    img_length = num_frame_per_file

    while total_frame_ind < num_frame:

        img = q.get()
        queue_size = q.qsize()

        if frame_ind == 0:
            start = time.time()
            
            img = np.expand_dims(img, axis=0)

            save_file = 'imgs_camera' + str(camera_ind) + '_file' + str(int(file_ind)) + '.h5'
            save_full = os.path.join(save_folder, save_file)
            hf = h5py.File(save_full, 'w')
            dset = hf.create_dataset('imgs', data=img,chunks=(1,image_y/2,image_x/2), maxshape=(img_length, image_y, image_x))
        else:
            dset.resize(dset.shape[0]+1, axis=0)
            dset[-1:,:,:] = img

        frame_ind = frame_ind + 1
        total_frame_ind = total_frame_ind + 1

        # last write of the file
        if frame_ind >= num_frame_per_file or total_frame_ind == num_frame:
            file_size = dset.shape[0]
            hf.close()
            del hf

            end = time.time()
            
            num_saved_frame = num_saved_frame + file_size
            print('Saved camera #' + str(int(camera_ind)) + ' frame #' + str(int(num_saved_frame)) + ": " + str(end - start) + " seconds. " + str(queue_size) + " in queue.\n")

            if num_frame - total_frame_ind < num_frame_per_file:
                img_length = num_frame - total_frame_ind
            
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
    return file_ind

def acquire(devices_sn,sn,camera_ind,cpu_core_inds,use_trigger,bit_depth,gain,black_level,exp_time,frame_rate,image_y,image_x,offset_y,offset_x,num_frame,num_frame_per_file,num_frame_per_write,save_folder):
    # process priority
    # p = psutil.Process(os.getpid())
    
    pid = os.getpid()
    p = psutil.Process(pid)
    p.nice(psutil.REALTIME_PRIORITY_CLASS)
    nice_value = p.nice()
    print(f"Child #{camera_ind}: {p}, nice value {nice_value}", flush=True)
    time.sleep(0.1)
    p.cpu_affinity(cpu_core_inds)
    print(f"Child #{camera_ind}: Set my affinity to {cpu_core_inds}, affinity now {p.cpu_affinity()}", flush=True)

    # create a queue
    q = queue.Queue()
    q2 = queue.Queue()

    # find the right device index
    device_ind = [str(sn) == d for d in devices_sn]
    device_ind = [i for i, val in enumerate(device_ind) if val][0]
    print('Camera # ' + str(camera_ind) + ' device index found')

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
    camera[0].OffsetX.SetValue(offset_x)
    camera[0].OffsetY.SetValue(offset_y)

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
        camera[0].LineSelector.SetValue('Line2')
        camera[0].LineMode.SetValue('Input')
        camera[0].LineSource.SetValue('ExposureActive')
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
    pool.submit(save_h5,q,q2,camera_ind,num_frame,num_frame_per_file,num_frame_per_write,save_folder,bit_depth,image_y,image_x)
    
    # wait for all tasks to complete
    pool.shutdown(wait=True)

    end = time.time()

    print('Total time taken (seconds) : ' + str(round((end - start))) + "\n")
    camera[0].Close()
    
    return True

def gui(camera_first,camera_last):   
    num_camera = camera_last - camera_first + 1

    # GUI template is created
    sg.theme('DarkAmber')   # Add a touch of color
    layout = []
    layout += [sg.Text('Camera index'), sg.In(size=(4,1), key='first_ind'), sg.Text('-'), sg.In(size=(4,1), key='last_ind'), sg.Button('Change camera range')],
    layout += [sg.Text('Load parameters (.json)'), sg.In(size=(25,1), enable_events=True ,key='parameter_file'), sg.FileBrowse(), sg.Button('Load parameters')],
    layout += [[sg.Button('Start acquisition'), sg.Button('Cancel')]]

    for c_ind in range(camera_first,camera_last+1):
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
        layout += [sg.Text('Height'), sg.In(key='image_y_' + str(c_ind))],    
        layout += [sg.Text('Width'), sg.In(key='image_x_' + str(c_ind))],    
        layout += [sg.Text('Offset Y'), sg.In(key='offset_y_' + str(c_ind))],    
        layout += [sg.Text('Offset X'), sg.In(key='offset_x_' + str(c_ind))],    
        layout += [sg.Text('CPU Core #'), sg.In(key='cpu_core_inds_' + str(c_ind))],    

    layout = [
        [
            sg.Column(layout, scrollable=True,  vertical_scroll_only=True),
            sg.Column([])
        ]
    ]

    window = sg.Window('Camera parameters', layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED,'Start acquisition','Cancel'):
            window.close()
            break
        if event == 'Change camera range':
            window.close()
            break
        if event == 'Load parameters':
            # get camera indexes
            c_inds = range(camera_first,camera_last+1)

            # Opening JSON file
            f = open(values['parameter_file'])

            # returns JSON object as 
            # a dictionary
            data = json.load(f)

            # get rid of extra backslash at the beginning
            save_folder_sub = data['save folder']
            if save_folder_sub[:1] == '\\':
                save_folder_sub = save_folder_sub[1:]
            if save_folder_sub[-1:] == '\\':
                save_folder_sub = save_folder_sub[:-1]

            # Write from json file onto the GUI
            for c_ind in c_inds:
                # make save folder
                save_folder = data['camera ' + str(c_ind)]['save drive'] + '\\' + save_folder_sub + '\\' + 'camera' + str(c_ind)

                values['sn_' + str(c_ind)] = data['camera ' + str(c_ind)]['sn']
                values['folder_' + str(c_ind)] = save_folder
                values['frame_num_' + str(c_ind)] = data['frame num']
                values['trigger_' + str(c_ind)] = data['camera ' + str(c_ind)]['use trigger']
                values['bd_' + str(c_ind)] = data['camera ' + str(c_ind)]['bit depth']
                values['fr_' + str(c_ind)] = data['camera ' + str(c_ind)]['frame rate']
                values['et_' + str(c_ind)] = data['exposure time']
                values['bl_' + str(c_ind)] = data['camera ' + str(c_ind)]['black level']
                values['gain_' + str(c_ind)] = data['camera ' + str(c_ind)]['gain']
                values['image_y_' + str(c_ind)] = data['camera ' + str(c_ind)]['image y']
                values['image_x_' + str(c_ind)] = data['camera ' + str(c_ind)]['image x']
                values['offset_y_' + str(c_ind)] = data['camera ' + str(c_ind)]['offset y']
                values['offset_x_' + str(c_ind)] = data['camera ' + str(c_ind)]['offset x']
                values['cpu_core_inds_' + str(c_ind)] = data['camera ' + str(c_ind)]['cores used']
            
            window.fill(values)
    
    if event == 'Cancel':
        exit()

    if event == 'Change camera range':
        values, num_camera, camera_first, camera_last = gui(int(values['first_ind']),int(values['last_ind']))

    return values, num_camera, camera_first, camera_last

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

    values, num_camera, camera_first, camera_last = gui(num_camera,num_camera)

    camera_ind = list(range(num_camera))
    sn = list(range(num_camera))
    num_frame_per_camera = list(range(num_camera))
    save_folders = list(range(num_camera))
    use_trigger = list(range(num_camera))
    bit_depth = list(range(num_camera))
    frame_rate = list(range(num_camera))
    exp_time = list(range(num_camera))
    gain = list(range(num_camera))
    black_level = list(range(num_camera))
    image_y = list(range(num_camera))
    image_x = list(range(num_camera))
    offset_y = list(range(num_camera))
    offset_x = list(range(num_camera))
    cpu_core_inds = list(range(num_camera))
    for c_ind in range(num_camera):
        camera_ind[c_ind] = c_ind + camera_first
        sn[c_ind] = int(values['sn_' + str(c_ind + camera_first)])
        num_frame_per_camera[c_ind] = int(values['frame_num_' + str(c_ind + camera_first)])
        save_folders[c_ind] = values['folder_' + str(c_ind + camera_first)]
        use_trigger[c_ind] = values['trigger_' + str(c_ind + camera_first)]
        bit_depth[c_ind] = values['bd_' + str(c_ind + camera_first)]
        frame_rate[c_ind] = int(values['fr_' + str(c_ind + camera_first)])
        exp_time[c_ind] = int(values['et_' + str(c_ind + camera_first)])
        gain[c_ind] = int(values['gain_' + str(c_ind + camera_first)])
        black_level[c_ind] = int(values['bl_' + str(c_ind + camera_first)])
        image_y[c_ind] = int(values['image_y_' + str(c_ind + camera_first)])
        image_x[c_ind] = int(values['image_x_' + str(c_ind + camera_first)])
        offset_y[c_ind] = int(values['offset_y_' + str(c_ind + camera_first)])
        offset_x[c_ind] = int(values['offset_x_' + str(c_ind + camera_first)])
        cpu_core_inds[c_ind] = [eval(i) for i in values['cpu_core_inds_' + str(c_ind + camera_first)][1:-1].split(', ')]

    # make folder if it does not exist
    for save_folder in save_folders:
        save_folder_exists = os.path.exists(save_folder)
        if not save_folder_exists:
            os.makedirs(save_folder)
            print("Save folder created.")

    # max number of frames per file
    num_frame_per_write = 100
    num_frame_per_file = 1000

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

        with mp.Pool() as pool:
            for i in range(num_camera):
                save_folder = save_folders[i]
                
                #cpu_core_inds = cpu_core_inds[cpu_core_inds < 24] # make sure max # of cores is not reached
                print(save_folder)
                pool.apply_async(acquire, (devices_sn,sn[i],camera_ind[i],cpu_core_inds[i],use_trigger[i],bit_depth[i],gain[i],black_level[i],exp_time[i],frame_rate[i],image_y[i],image_x[i],offset_y[i],offset_x[i],num_frame_per_camera[i],num_frame_per_file,num_frame_per_write,save_folder,))

            # Wait for children to finnish
            pool.close()
            pool.join()

        end = time.time()

        print('Total time taken (seconds) : ' + str(round((end - start))) + "\n")

    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", e.GetDescription())
        exitCode = 1

    # Comment the following two lines to disable waiting on exit.
    sys.exit(exitCode)
