import FreeSimpleGUI as sg

def camera_select_gui(camera_first,camera_last):   
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
        layout += [sg.Text('Frame number per file'), sg.In(key='frame_num_file_' + str(c_ind))],
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
                values['frame_num_file_' + str(c_ind)] = data['frame num per file']
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
                values['cpu_core_inds_' + str(c_ind)] = data['camera ' + str(c_ind)]['cores used']
            
            window.fill(values)
    
    if event == 'Cancel':
        exit()

    if event == 'Change camera range':
        values, num_camera, camera_first, camera_last = gui(int(values['first_ind']),int(values['last_ind']))

    return values, num_camera, camera_first, camera_last