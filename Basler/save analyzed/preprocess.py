import h5py
import numpy as np
from pathlib import Path
import os
import glob
import time
from natsort import natsorted
import scipy.io as sio

def reshape_window(imgs, window_size_y, window_size_x):
    """
    Reshapes images into rectangular windows
    
    Args:
        imgs: Input image array
        window_size_y: Window size in y dimension
        window_size_x: Window size in x dimension
    
    Returns:
        tuple: (reshaped images array, window indices)
    """
    import numpy as np
    
    # Cut imgs to multiple of window sizes
    img_size_y = (imgs.shape[0] // window_size_y) * window_size_y
    img_size_x = (imgs.shape[1] // window_size_x) * window_size_x
    imgs = imgs[:img_size_y, :img_size_x, ...]
    window_num = (img_size_y // window_size_y) * (img_size_x // window_size_x)
    
    # Window indices
    row_ind, col_ind = np.meshgrid(np.arange(1, img_size_y + 1), 
                                  np.arange(1, img_size_x + 1))
    row_ind = row_ind.T
    col_ind = col_ind.T
    
    window_ind_temp = np.ceil(row_ind/window_size_y) + \
                     (np.ceil(col_ind/window_size_x) - 1) * (img_size_y/window_size_y)
    
    # Reshape by window
    window_ind = np.argsort(window_ind_temp.ravel())
    imgs = imgs.reshape(-1, imgs.shape[2] if imgs.ndim > 2 else 1)
    imgs = imgs[window_ind, :]
    imgs = imgs.reshape(window_size_y * window_size_x, window_num, -1)
    
    if imgs.shape[-1] == 1:
        imgs = imgs.squeeze(axis=-1)
        
    return imgs

def get_frames(img_files, frame_ind, frame_per_file):
    """
    Get frames from h5 files based on which frames to get and frames per file
    
    Args:
        img_files: List of h5 files containing image data
        frame_ind: Array of frame indices to retrieve
        frame_per_file: Number of frames per file
    
    Returns:
        numpy array containing the requested frames
    """
    
    # Calculate which files contain the requested frames
    frame_file_ind = np.ceil(frame_ind / frame_per_file).astype(int)
    
    # Calculate frame index within each file (1-based indexing)
    file_frame_ind = np.mod(frame_ind - 1, frame_per_file) + 1
    
    # Get unique file indices
    file_ind = np.unique(frame_file_ind)
    if file_ind.shape[0] > 1:
        file_ind = file_ind.reshape(1, -1)[0]
    
    imgs = []
    for f in file_ind:
        # Get frame indices for current file
        mask = frame_file_ind == f
        start_ind = np.min(file_frame_ind[mask])
        end_ind = np.max(file_frame_ind[mask])
        
        # Read frames from h5 file
        with h5py.File(os.path.join(img_files[f-1]), 'r') as h5f:
            imgs_file = h5f['imgs'][start_ind-1:end_ind, :, :]
            
        # Select only requested frames
        frame_indices = file_frame_ind[mask] - start_ind
        imgs.append(imgs_file[frame_indices, :, :])
    
    # Concatenate all frames
    if imgs:
        imgs = np.concatenate(imgs, axis=2)
    else:
        imgs = np.array([])
        
    return imgs

def preprocess_h5(img_files, dark_files, ts_file, window_y, window_x, hot_pix_fcn, cycle_frame_num, save_folder, dark_save_folder, file_ind=None, temporal=False, section_frame_num=1000):
    flag = False
    frame_num_total = 0
    
    # Calculate total frames
    if file_ind is None:
        file_ind = range(len(img_files))
        
    for k in file_ind:
        with h5py.File(img_files[k], 'r') as f:
            frame_num_total += f['imgs'].shape[2]
    
    frame_num = 0
    frame_step = 100
    
    # Handle timestamps
    consider_skipped_frame = False if not ts_file else True
    
    if consider_skipped_frame:
        with h5py.File(ts_file, 'r') as f:
            ts = f['timestamps'][:]
        ts_diff = np.diff(ts)
        ts_diff_scale = np.round(ts_diff / np.mean(ts_diff))
    
    # Process dark files
    save_file = 'dark_preprocessed.mat'
    os.makedirs(dark_save_folder, exist_ok=True)
    
    dark_var_pix = None
    dark_mean_pix = None
    
    for dark_ind, dark_file in enumerate(dark_files):
        with h5py.File(os.path.join(dark_file), 'r') as f:
            file_frame_num = f['imgs'].shape[0]
            
            for section_ind in range(1, file_frame_num // frame_step + 1):
                start_loc = [(section_ind-1)*frame_step, 0, 0]
                dark_imgs = f['imgs'][start_loc[0]:start_loc[0]+frame_step, start_loc[1]:, start_loc[2]:]
                dark_imgs = np.transpose(dark_imgs, (2, 1, 0))
                
                frame_num += frame_step
                
                if section_ind == 1 and dark_ind == 0:
                    dark_var_pix = np.var(dark_imgs.astype(float), axis=2) * frame_step
                    dark_mean_pix = np.mean(dark_imgs.astype(float), axis=2) * frame_step
                else:
                    dark_var_pix += np.var(dark_imgs.astype(float), axis=2) * frame_step
                    dark_mean_pix += np.mean(dark_imgs.astype(float), axis=2) * frame_step
    
    dark_var_pix /= frame_num
    dark_mean_pix /= frame_num
    
    # Create coordinate meshgrid
    Y, X = np.meshgrid(range(dark_var_pix.shape[0]), range(dark_var_pix.shape[1]))
    X = X.T
    Y = Y.T
    
    x_coor_start = reshape_window(X, window_y, window_x)[0,:]
    y_coor_start = reshape_window(Y, window_y, window_x)[0,:]
    
    # Window dark var and mean
    dark_var_pix = reshape_window(dark_var_pix, window_y, window_x)
    dark_mean_pix = reshape_window(dark_mean_pix, window_y, window_x)
    window_num = dark_var_pix.shape[1]
    
    # Find hot pixels
    hot_pix = hot_pix_fcn(dark_mean_pix, dark_var_pix)
    
    # Calculate dark var window
    dark_var_windowed = np.zeros(window_num)
    dark_mean_windowed = np.zeros(window_num)
    
    for window_ind in range(window_num):
        dark_var_windowed[window_ind] = np.mean(dark_var_pix[~hot_pix[:,window_ind], window_ind])
        dark_mean_windowed[window_ind] = np.mean(dark_mean_pix[~hot_pix[:,window_ind], window_ind])
    
    # Save dark data
    sio.savemat(os.path.join(dark_save_folder, save_file), {
        'dark_var_windowed': dark_var_windowed,
        'dark_var_pix': dark_var_pix,
        'dark_mean_windowed': dark_mean_windowed,
        'dark_mean_pix': dark_mean_pix,
        'x_coor_start': x_coor_start,
        'y_coor_start': y_coor_start
    })
    
    if consider_skipped_frame:
        # Calculate true frame indices
        true_frame_ind = np.zeros_like(ts)
        true_frame_ind[0] = 1
        for ts_diff_ind in range(len(ts_diff_scale)):
            true_frame_ind[ts_diff_ind + 1] = true_frame_ind[ts_diff_ind] + ts_diff_scale[ts_diff_ind]
            
        # Calculate save frame indices
        save_frame_ind = np.zeros_like(ts)
        save_frame_ind[0] = 1
        frame_cum = 1
        
        for ts_diff_ind in range(len(ts_diff_scale)):
            if ts_diff_scale[ts_diff_ind] > cycle_frame_num and ts_diff_ind == 0:
                ts_diff_scale[ts_diff_ind] = 1
                save_frame_ind[frame_cum:frame_cum+int(ts_diff_scale[ts_diff_ind])] = \
                    save_frame_ind[frame_cum-1] + np.arange(1, ts_diff_scale[ts_diff_ind]+1)
                save_frame_ind += 1
                flag = True
            else:
                save_frame_ind[frame_cum:frame_cum+int(ts_diff_scale[ts_diff_ind])] = \
                    save_frame_ind[frame_cum-1] + np.arange(1, ts_diff_scale[ts_diff_ind]+1)
            frame_cum += int(ts_diff_scale[ts_diff_ind])
    else:
        save_frame_ind = np.arange(1, frame_num_total + 1)
        true_frame_ind = save_frame_ind.copy()
    
    # Source ind of raw frames
    source_ind_all = np.mod(np.arange(len(save_frame_ind)), cycle_frame_num)
    source_ind_all = source_ind_all + 1

    raw_frame_source_ind = np.mod(true_frame_ind - 1, cycle_frame_num) 
    raw_frame_source_ind = raw_frame_source_ind + 1

    # Process image sections
    section_num = int(np.ceil(len(save_frame_ind) / section_frame_num))
    os.makedirs(save_folder, exist_ok=True)
    
    for s_ind in range(1, section_num + 1):
        print(f'Section # {s_ind}')
        save_file = f'section{s_ind}_preprocessed.mat'
        
        section_frame_ind = np.arange((s_ind-1)*section_frame_num, min(s_ind*section_frame_num, len(save_frame_ind)))
        save_section_frame_ind = save_frame_ind[section_frame_ind]
        source_ind = np.mod(section_frame_ind, cycle_frame_num) + 1
        
        imgs_section = get_frames(img_files, save_section_frame_ind, file_frame_num)
        imgs_section = np.transpose(imgs_section, (2, 1, 0))
        imgs_section = reshape_window(imgs_section, window_y, window_x)
        
        imgs_section_offset = imgs_section.astype(float) - dark_mean_pix[:, :, np.newaxis]
        
        # Calculate statistics
        img_mean = np.zeros((window_y*window_x, window_num, cycle_frame_num))
        img_mean_frame_num = np.zeros(cycle_frame_num)
        
        for s in range(cycle_frame_num):
            mask = source_ind == s + 1
            img_mean[:,:,s] = np.mean(imgs_section_offset[:,:,mask], axis=2)
            img_mean_frame_num[s] = np.sum(mask)
        
        var_windowed = np.zeros((window_num, imgs_section_offset.shape[2]))
        mean_windowed = np.zeros((window_num, imgs_section_offset.shape[2]))
        
        for window_ind in range(window_num):
            var_windowed[window_ind,:] = np.var(imgs_section_offset[~hot_pix[:,window_ind], window_ind, :], axis=0)
            mean_windowed[window_ind,:] = np.mean(imgs_section_offset[~hot_pix[:,window_ind], window_ind, :], axis=0)
        
        # Save results
        save_dict = {
            'source_ind': source_ind,
            'mean_windowed': mean_windowed,
            'var_windowed': var_windowed,
            'img_mean': img_mean,
            'img_mean_frame_num': img_mean_frame_num,
            'hot_pix': hot_pix,
            'x_coor_start': x_coor_start,
            'y_coor_start': y_coor_start,
            'save_frame_ind': save_frame_ind,
            'raw_frame_source_ind': raw_frame_source_ind
        }
        
        if consider_skipped_frame:
            save_dict['ts'] = ts
            
        sio.savemat(os.path.join(save_folder, save_file), save_dict)

# Drive locations mapping
drive_loc = {
    i: 'D:' if i in [1, 2, 9, 10] else
       'E:' if i in [3, 4, 11, 12] else
       'F:' if i in [5, 6, 13, 14, 17] else
       'G:' if i in [7, 8, 15, 16] else None
    for i in range(1, 18)
}

bingus = ['run1', 'run2', 'run3']

def process_run(camera, run):
    print(str(camera))
    super_folder = Path(drive_loc[camera]) / '20241120 stroop'
    
    # Get image files
    img_pattern = os.path.join(super_folder, f"{bingus[run-1]}", f"camera{camera}", "*file*.h5")
    img_files = natsorted(glob.glob(img_pattern))
    
    # Get timestamp file
    ts_pattern = os.path.join(super_folder, f"{bingus[run-1]}", f"camera{camera}", "*timestamps.h5")
    ts_file = glob.glob(ts_pattern)[0]
    
    # Get dark files
    dark_pattern = os.path.join(super_folder, "dark", f"camera{camera}", "*file*.h5")
    dark_files = glob.glob(dark_pattern)
    
    window_y = 7
    window_x = 7
    
    def hot_pix_fcn(mu, var):
        return (mu >= 24.4 + (0.375*2.2)*5) | (var >= ((0.375*2.2*10)**2 + 1/12))
    
    cycle_frame_num = 7
    
    # Create save folders
    save_folder = os.path.join(super_folder, "preprocessed-python", f"run{run}", f"camera{camera}")
    dark_save_folder = os.path.join(save_folder, "dark")
    
    # Process data
    start_time = time.time()
    preprocess_h5(
        img_files=img_files,
        dark_files=dark_files,
        ts_file=ts_file,
        window_y=window_y,
        window_x=window_x,
        hot_pix_fcn=hot_pix_fcn,
        cycle_frame_num=cycle_frame_num,
        save_folder=save_folder,
        dark_save_folder=dark_save_folder,
        section_frame_num=500
    )
    print(f"Processing time: {time.time() - start_time:.2f} seconds")

def main():
    camera = 16
    for run in range(1, 3):
        process_run(camera, run)

if __name__ == "__main__":
    main()