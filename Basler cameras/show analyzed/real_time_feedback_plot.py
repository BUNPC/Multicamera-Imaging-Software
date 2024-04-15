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
import matplotlib.pyplot as plt
from tabulate import tabulate
from rich.live import Live
from rich.table import Table
from matplotlib.animation import FuncAnimation
import csv

def obtain_frame(q,camera_ind,camera,num_frame,num_frame_per_file):
    starth = time.time()
    print('Obtain frame started for camera # ' + str(camera_ind))
    total_frame_ind = 0
    frame_ind = 0
    start = time.time()

    while True:
        if not camera.IsGrabbing():
            break

        # if total_frame_ind % 100 == 0:
        #     print('Acquiring camera # ' + str(camera_ind) + ' frame # ' + str(total_frame_ind) + ', ' + str(datetime.now()) + ', q = ' + str(q.qsize()) + "\n")
        
        # first value is wait time before time out (in ms)
        grabResult = camera.RetrieveResult(
            120000, pylon.TimeoutHandling_ThrowException)
        
        # Access the chunk data attached to the result.
        # Before accessing the chunk data, you should check to see
        # if the chunk is readable. When it is readable, the buffer
        # contains the requested chunk data.
        
        img = grabResult.GetArray()

        mean_val = np.mean(img)

        # make dictionary
        dict = {
            "camera_ind": camera_ind,
            "frame_ind": total_frame_ind,
            "mean_val": mean_val
        }
        
        frame_ind = frame_ind + 1
        total_frame_ind = total_frame_ind + 1

        q.put(dict)

        if frame_ind == num_frame_per_file:
            frame_ind = 0
    endh = time.time()
    print('Obtained all frames in ' + str(endh-starth) + " seconds.\n")

def generate_table(source_num,camera_num,data,qsize) -> Table:
    """Make a new table."""
    table = Table()
    table.add_column("")
    for col in range(camera_num):
        table.add_column("Camera " + str(col+1))

    table.add_row(qsize)
    for row in range(source_num):
        value = data[row,:].tolist()
        value_str = [str(x) for x in value]
        value_str = ["Source " + str(row+1)] + value_str
        table.add_row(*value_str)
    return table

def update(i, lines, axs, q, camera_first_ind):
    # load data
    #print('Loading data')
    data = np.zeros((axs.shape[0],axs.shape[1],50))
    if os.path.exists('data_temp.csv'):
        with open('data_temp.csv', newline='') as csvfile:
            csvreader_object=csv.reader(csvfile)
            for row in csvreader_object:
                source_ind = int(row[0])
                camera_ind = int(row[1])
                data[source_ind,camera_ind,:] = row[2:]
    source_num = data.shape[0]
    camera_num = data.shape[1]

    # update data
    print('Queue size : ' + str(q.qsize()))
    updated = np.zeros((axs.shape[0],axs.shape[1]))
    while np.sum(updated) < axs.shape[0]*axs.shape[1]*40:
        val_update = q.get()
        frame_ind = val_update['frame_ind']
        source_ind = frame_ind % source_num
        camera_ind = val_update['camera_ind']
        val_single = val_update['mean_val']

        data[source_ind,camera_ind-camera_first_ind,0:49] = data[source_ind,camera_ind-camera_first_ind,1:50]
        data[source_ind,camera_ind-camera_first_ind,49] = val_single
        updated[source_ind,camera_ind-camera_first_ind] = updated[source_ind,camera_ind-camera_first_ind] + 1
    
    # write csv
    #print('Writing data')
    with open('data_temp.csv', 'w', newline='') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)
            
        # writing the data rows
        for source_ind in range(0,source_num):
            for camera_ind in range(0,camera_num):
                row = [str(source_ind)] + [str(camera_ind)] + list(map(str, np.squeeze(data[source_ind,camera_ind,:]).tolist()))
                csvwriter.writerow(row)

    #print('Updating plots')
    for source_ind in range(0,source_num):
        for camera_ind in range(0,camera_num):
            y = np.squeeze(data[source_ind,camera_ind,:])
            lines[source_ind*camera_num + camera_ind].set_ydata(y)
            y_min = np.min(y)
            y_max = np.max(y)
            axs[source_ind,camera_ind].set_ylim(y_min,y_max)

def pool_data(camera_num,source_num,frame_rate,q,camera_first_ind):
    print('Initializing data')
    if os.path.exists('data_temp.csv'):
        os.remove('data_temp.csv')

    # get data from other threads
    camera_1_frame = 0
    print('Updating table')
    fig, axs = plt.subplots(source_num, camera_num, figsize=(10, 6), layout='constrained')
    # plt.ion()
    # fig.show()
    #fig.canvas.show()
    lines = []
    for source_ind in range(0,source_num):
        lines_row = []
        for camera_ind in range(0,camera_num):
            y = np.zeros(50)
            lines_row = lines_row + axs[source_ind,camera_ind].plot([x / (frame_rate/source_num) for x in range(0,y.size)],y,linewidth=1)
            axs[source_ind,camera_ind].set_title('S' + str(source_ind+1) + 'D' + str(camera_ind + camera_first_ind),fontsize = 8)
            axs[source_ind,camera_ind].grid()
        lines = lines + lines_row

    anim = FuncAnimation(fig, update, frames=100, fargs=(lines, axs, q, camera_first_ind), interval=200)
    plt.show()

    # while camera_1_frame < camera_frame_num:
    #     val_update = q.get()
    #     # print('New frame obtained')
    #     frame_ind = val_update['frame_ind']
    #     source_ind = frame_ind % source_num
    #     camera_ind = val_update['camera_ind']
    #     val_single = val_update['mean_val']

    #     data[source_ind,camera_ind,0:49] = data[source_ind,camera_ind,1:50]
    #     data[source_ind,camera_ind,49] = val_single

    #     if source_ind == (source_num - 1) and camera_ind == (camera_num - 1):
    #     #     for source_ind in range(0,source_num):
    #     #         for camera_ind in range(0,camera_num):
    #     #             y = np.squeeze(data[source_ind,camera_ind,:])
    #     #             lines[source_ind+camera_ind*camera_num].set_ydata(y)
    #     #             axs[source_ind,camera_ind].set_ylim(np.min(y)-1, np.max(y)+1)
    #     #             # axs[source_ind,camera_ind].plot(range(0,y.size),y)
    #     #     print('Plot updated')
    #         print('Queue: ' + str(q.qsize()))
    #     #     fig.canvas.draw_idle()
    #     #     fig.show()
    #         #
        
    #     if camera_ind == 1:
    #         camera_1_frame = camera_1_frame + 1
        
    #     plt.pause(0.00001)

def acquire(devices_sn,sn,camera_ind,use_trigger,bit_depth,gain,black_level,exp_time,frame_rate,image_y,image_x,offset_y,offset_x,num_frame,num_frame_per_file,q):
    # process priority
    # p = psutil.Process(os.getpid())
    # p.nice(psutil.HIGH_PRIORITY_CLASS)

    # create a queue

    # find the right device index
    camera_num = len(devices_sn)
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
    pool.submit(obtain_frame,q,camera_ind,camera,num_frame,num_frame_per_file)

    #pool_data(camera_num,source_num,num_frame,q)
    
    # wait for all tasks to complete
    pool.shutdown(wait=True)

    end = time.time()

    print('Total time taken (seconds) : ' + str(round((end - start))) + "\n")
    camera[0].Close()
    
    return True

def gui(camera_first,camera_last):   
    num_camera = camera_last - camera_first + 1

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

            for c_ind in c_inds:
                # make save folder
                save_folder = data['camera ' + str(c_ind)]['save drive'] + '\\' + save_folder_sub + '\\' + 'camera' + str(c_ind)

                values['sn_' + str(c_ind)] = data['camera ' + str(c_ind)]['sn']
                values['folder_' + str(c_ind)] = save_folder
                values['frame_num_' + str(c_ind)] = data['camera ' + str(c_ind)]['frame num']
                values['trigger_' + str(c_ind)] = data['camera ' + str(c_ind)]['use trigger']
                values['bd_' + str(c_ind)] = data['camera ' + str(c_ind)]['bit depth']
                values['fr_' + str(c_ind)] = data['camera ' + str(c_ind)]['frame rate']
                values['et_' + str(c_ind)] = data['camera ' + str(c_ind)]['exposure time']
                values['bl_' + str(c_ind)] = data['camera ' + str(c_ind)]['black level']
                values['gain_' + str(c_ind)] = data['camera ' + str(c_ind)]['gain']
                values['image_y_' + str(c_ind)] = data['camera ' + str(c_ind)]['image y']
                values['image_x_' + str(c_ind)] = data['camera ' + str(c_ind)]['image x']
                values['offset_y_' + str(c_ind)] = data['camera ' + str(c_ind)]['offset y']
                values['offset_x_' + str(c_ind)] = data['camera ' + str(c_ind)]['offset x']
            
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

    source_num = 7

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

        # q = queue.Queue()
        q = mp.Queue()

        # make multiprocessing nodes
        p = [0]*(num_camera + 1)
        for i in range(0,num_camera):
            save_folder = save_folders[i]
            print(save_folder)
            p[i] = mp.Process(target=acquire,args=(devices_sn,sn[i],camera_ind[i],use_trigger[i],bit_depth[i],gain[i],black_level[i],exp_time[i],frame_rate[i],image_y[i],image_x[i],offset_y[i],offset_x[i],num_frame_per_camera[i],num_frame_per_file,q))

        # final node for table
        p[-1] = mp.Process(target=pool_data,args=(num_camera,source_num,frame_rate[0],q,camera_first))

        # run the new process
        for i in range(0,num_camera+1):
            p[i].start()

        # join the new process to main process
        for i in range(0,num_camera+1):
            p[i].join()

        end = time.time()

        print('Total time taken (seconds) : ' + str(round((end - start))) + "\n")

    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", e.GetDescription())
        exitCode = 1

    # Comment the following two lines to disable waiting on exit.
    sys.exit(exitCode)
