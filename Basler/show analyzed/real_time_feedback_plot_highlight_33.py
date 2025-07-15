import sys
from pypylon import pylon
from pypylon import genicam
import numpy as np
import os
import time
from datetime import datetime
import concurrent.futures
import multiprocessing as mp
import matplotlib.pyplot as plt
from tabulate import tabulate
from rich.table import Table
from matplotlib.animation import FuncAnimation
import csv

sys.path.append('.')

from lib import gui

def obtain_frame(q,camera_ind,camera,num_frame_per_file,exp_time_pattern):
    starth = time.time()
    print('Obtain frame started for camera # ' + str(camera_ind))
    total_frame_ind = 0
    frame_ind = 0

    while True:
        if not camera.IsGrabbing():
            break

        # if total_frame_ind % 100 == 0:
        #     print('Acquiring camera # ' + str(camera_ind) + ' frame # ' + str(total_frame_ind) + ', ' + str(datetime.now()) + ', q = ' + str(q.qsize()) + "\n")
        
        exp_ind = total_frame_ind % len(exp_time_pattern)
        camera[0].ExposureTime.SetValue(exp_time_pattern[exp_ind])

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

def update(i, lines, axs, q, camera_first_ind, csv_file_name):
    # load data
    #print('Loading data')
    if len(axs.shape) == 1:
        data = np.zeros((axs.shape[0],1,50))
    else:
        data = np.zeros((axs.shape[0],axs.shape[1],50))
    if os.path.exists(csv_file_name):
        with open(csv_file_name, newline='') as csvfile:
            csvreader_object=csv.reader(csvfile)
            for row in csvreader_object:
                source_ind = int(row[0])
                camera_ind = int(row[1])
                data[source_ind,camera_ind,:] = row[2:]
    source_num = data.shape[0]
    camera_num = data.shape[1]

    # update data
    print('Queue size : ' + str(q.qsize()))
    if len(axs.shape) == 1:
        updated = np.zeros((axs.shape[0],1))
        frame_buffer_num = axs.shape[0]*40
    else:
        updated = np.zeros((axs.shape[0],axs.shape[1]))
        frame_buffer_num = axs.shape[0]*axs.shape[1]*40
    
    while np.sum(updated) < frame_buffer_num:
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
    with open(csv_file_name, 'w', newline='') as csvfile:  
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
            y = y - 24.3
            lines[source_ind*camera_num + camera_ind].set_ydata(y)
            y_min = np.mean(y)*0.98
            y_max = np.mean(y)*1.02
            y_min = np.min([y_min, np.min(y)])
            y_max = np.max([y_max, np.max(y)])
            #contrast = np.std(y)/(np.mean(y)+0.000001)
            if y_max == y_min:
                y_max = y_min + 0.01 
            if len(axs.shape) == 1:
                axs[source_ind].set_ylim(y_min,y_max)
            else:
                axs[source_ind,camera_ind].set_ylim(y_min,y_max)
            #axs[source_ind].annotate(str(np.std(y)/(np.mean(y)+0.00001)), xy=(1, np.mean(y)*0.99), xycoords='axes fraction')
            #axs[source_ind,camera_ind].annotate(str(np.std(y)/(np.mean(y)+0.00001)), xy=(1, np.mean(y)*0.99), xycoords='axes fraction')
            #axs[source_ind,camera_ind].text(1, np.mean(y)*0.99, str(np.std(y)/(np.mean(y)+0.00001)))
            # Define a list to store text objects
            #texts = []

            # Clear previous text annotations
            #for txt in texts:
            #    txt.remove()

            # Add new text annotation
            #text = axs[source_ind,camera_ind].annotate(str(np.std(y)/(np.mean(y)+0.00001)), xy=(1, np.mean(y)*0.99), xycoords='axes fraction')

            # Store the new text object
            #texts.append(text)


            

def pool_data(camera_num,source_num,frame_rate,q,camera_first_ind):
    print('Initializing data')
    csv_file_name = 'data_temp' + str(camera_first_ind) + '.csv'
    if os.path.exists(csv_file_name):
        os.remove(csv_file_name)


    # List of source-detector pairs with 1-based indices
    green_pairs_1based = [
        (5, 1), (6, 1),
        (7, 2), (8, 2),
        (9, 3), (10, 3),
        (7, 4), (8, 4),
        (3, 5), (4, 5), (5, 5), (6, 5),
        (1, 6), (2, 6), (9, 6), (10, 6),
        (7, 7), (8, 7),
        (1, 8), (2, 8), (11, 8), (12, 8),
        (3, 9), (4, 9), (13, 9), (14, 9),
        (7, 10), (8, 10),
        (5, 11), (6, 11), (13, 11), (14, 11),
        (9, 12), (10, 12), (11, 12), (12, 12),
        (7, 13), (8, 13),
        (5, 14), (6, 14),
        (7, 15), (8, 15),
        (9, 16), (10, 16)
    ]

    # Convert to zero-based indices by subtracting 1 from source and detector
    green_pairs = [(s-1, d-1) for s, d in green_pairs_1based]

    # Initialize colors matrix
    colors = [['black' for _ in range(17)] for _ in range(14)]

    # Set green for specified pairs
    for s_idx, d_idx in green_pairs:
        colors[s_idx][d_idx] = 'green'

    # get data from other threads
    camera_1_frame = 0
    print('Updating table')
    fig_w = max(10, camera_num * 1.75)
    fig_h = max(6, source_num * 0.625)
    fig, axs = plt.subplots(source_num, camera_num, figsize=(fig_w, fig_h), layout='constrained')
    # plt.ion()
    # fig.show()
    #fig.canvas.show()
    lines = []
    for source_ind in range(0,source_num):
        lines_row = []
        for camera_ind in range(0,camera_num):
            y = np.zeros(50)
            if camera_num == 1:
                lines_row = lines_row + axs[source_ind].plot([x / (frame_rate/source_num) for x in range(0,y.size)],y,linewidth=1,)
                axs[source_ind].set_title('S' + str(source_ind+1) + 'D' + str(camera_ind + camera_first_ind),fontsize = 8)
                axs[source_ind].grid()
                #axs[source_ind].text(1, np.mean(y)*0.99, str(np.std(y)/(np.mean(y)+0.00001)))
                #axs[source_ind].text(1, np.mean(y)*0.99, str(np.std(y)/(np.mean(y)+0.00001)))

            else:
                lines_row = lines_row + axs[source_ind,camera_ind].plot([x / (frame_rate/source_num) for x in range(0,y.size)],y,linewidth=1,color = colors[source_ind][camera_ind + camera_first_ind - 1])
                axs[source_ind,camera_ind].set_title('S' + str(source_ind+1) + 'D' + str(camera_ind + camera_first_ind),fontsize = 8)
                axs[source_ind,camera_ind].grid()
                #axs[source_ind].text(1, np.mean(y)*0.99, str(np.std(y)/(np.mean(y)+0.00001)))
                #axs[source_ind].text(1, np.mean(y)*0.99, str(np.std(y)/(np.mean(y)+0.00001)))

        lines = lines + lines_row

    anim = FuncAnimation(fig, update, frames=100, fargs=(lines, axs, q, camera_first_ind, csv_file_name), interval=200)
    
    # Maximize the figure window (TkAgg backend on Windows)
    manager = plt.get_current_fig_manager()
    try:
        manager.window.state('zoomed')
    except Exception:
        pass  # if it fails, just continue silently

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

def acquire(devices_sn,sn,camera_ind,use_trigger,bit_depth,gain,black_level,exp_time_pattern,frame_rate,image_y,image_x,offset_y,offset_x,num_frame,num_frame_per_file,q):
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
    camera[0].ExposureTime.SetValue(exp_time_pattern[0])

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
    pool.submit(obtain_frame,q,camera_ind,camera,num_frame_per_file,exp_time_pattern)

    #pool_data(camera_num,source_num,num_frame,q)
    
    # wait for all tasks to complete
    pool.shutdown(wait=True)

    end = time.time()

    print('Total time taken (seconds) : ' + str(round((end - start))) + "\n")
    camera[0].Close()
    
    return True

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

    #source_num = 7
    source_num = 14

    values, num_camera, camera_first, camera_last = gui.camera_select_gui(num_camera,num_camera)
    camera_ind, sn, num_frame_per_camera, num_frame_per_file, save_folders, use_trigger, bit_depth, frame_rate, exp_time_patterns, gain, black_level, image_y, image_x, offset_y, offset_x, cpu_core_inds = gui.parse_gui_output(values, num_camera, camera_first)

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
            p[i] = mp.Process(target=acquire,args=(devices_sn,sn[i],camera_ind[i],use_trigger[i],bit_depth[i],gain[i],black_level[i],exp_time_patterns[i],frame_rate[i],image_y[i],image_x[i],offset_y[i],offset_x[i],num_frame_per_camera[i],num_frame_per_file,q))

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
