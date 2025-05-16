import argparse
import datetime
import logging
import h5py
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np

from temporaldata import Data, ArrayDict, Interval, IrregularTimeSeries

## this is from the brainsets package, not the local dir
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)

from brainsets.taxonomy import RecordingTech, Task
from brainsets import serialize_fn_map

#from local dir
from monkey_N_utils import monkey_N_subject

logging.basicConfig(level=logging.INFO)


def extract_units(task_data):
    #Extract units from the data dictionary.
    
    # num_units = np.shape(task_data["binned_sbp"])[1]
    num_units = np.max(task_data["spike_channels"]) + 1
    
    ids = np.array(["channel"+str(i) for i in range(num_units)])
    chan_number = np.array([i for i in range(num_units)])
    chan_type = np.zeros(num_units,dtype= np.uint8)

    #count spikes per channel
    counts = np.zeros(num_units,dtype= np.uint32)
    
    for unit in task_data["spike_channels"]:
        counts[unit] += 1 
        
    units = ArrayDict(
        id=ids,
        unit_number = chan_number,
        type = chan_type,
        count = counts
        )

    return units

def extract_spikes(task_data):
    
    timestamps = task_data["spike_times"] / 1000.0
    unit_index = task_data["spike_channels"]
    sorted_pairs = sorted(zip(timestamps,unit_index))
    sorted_tuple_tstep, sorted_tuple_uidx =zip(*sorted_pairs)
    sorted_timestamps = np.array(sorted_tuple_tstep,dtype= np.float64)
    sorted_unit_index = np.array(sorted_tuple_uidx)
    
    spikes=IrregularTimeSeries(
        timestamps=sorted_timestamps,
        unit_index=sorted_unit_index,
        domain="auto"
    )

    ### for simulating spikes from spiking band power
    # sbp = task_data["sbp"] #binned_sbp
    
    # # zero out the top 10% of sbp as outliers
    # ninety_percentile = np.percentile(sbp,90,axis = 0)
    # sbp[sbp>ninety_percentile] = np.nan #set top ten percent to nan so they dont influence normalization
    
    # # normalize 
    # minmax_transformer = MinMaxScaler().fit(sbp)
    # sbp = minmax_transformer.transform(sbp)
    
    # # set nan to zeros -> no contribution to spikes
    # sbp = np.nan_to_num(sbp)

    # num_units = sbp.shape[1]
    # timestamps = np.array(task_data['time'],dtype= np.float64) / 1000.0
    
    # unit_idxs = np.array([i for i in range(num_units)])
    # spiking_unit_list = []
    # spiking_timestamps = []

    # for t_step_idx in range(1,len(timestamps)):#assume no spikes at first time bin
    #     sbp[t_step_idx,:] += sbp[t_step_idx-1,:]
    #     spike_occured_mask= sbp[t_step_idx,:] >= 1
    #     sbp[t_step_idx,:][spike_occured_mask] -= 1
    #     spiking_unit_list.extend(unit_idxs[spike_occured_mask])
    #     spiking_timestamps.extend([timestamps[t_step_idx]]*spike_occured_mask.sum())
    
    # #no need to sort, we grab spikes in order
    # np_spiking_timestamps = np.array(spiking_timestamps, dtype= np.float64).squeeze()
    # np_spiking_unit_list = np.array(spiking_unit_list)
    
    # spikes=IrregularTimeSeries(
    #     timestamps = np_spiking_timestamps,
    #     unit_index= np_spiking_unit_list,
    #     domain="auto"
    # )
    # units = extract_units(task_data,np_spiking_unit_list)
    
    return spikes

def extract_behavior(task_data): #updated
    
    timestamps = np.array(task_data['bin_times'],dtype= np.float64) / 1000.0
    cursor_pos = task_data['binned_pos']
    cursor_vel = task_data['binned_vel']
    cursor_pos_and_vel=  np.concatenate((cursor_pos,cursor_vel), axis = 1)
    
    cursor = IrregularTimeSeries( 
        timestamps=timestamps,
        pos_and_vel=cursor_pos_and_vel,
        pos=cursor_pos,
        vel=cursor_vel,
        domain="auto",
    )
    
    mean = np.mean(cursor_pos_and_vel[:int(len(task_data['bin_times'])*0.7)],axis = 0)
    std = np.std(cursor_pos_and_vel[:int(len(task_data['bin_times'])*0.7)],axis = 0)

    return cursor, mean, std

def interval_splitter(task_data):
    
    bin_times = task_data["bin_times"] / 1000.0
    num_bins = len(bin_times)
    
    train_interval = Interval(bin_times[0],bin_times[int(num_bins*.7)])
    valid_interval = Interval(bin_times[int(num_bins*.7)],bin_times[int(num_bins*.8)])
    test_interval = Interval(bin_times[int(num_bins*.8)],bin_times[-1])
    
    return train_interval, valid_interval, test_interval 

def main():
    #open raw data file manifest
    dir = "./data/"
    data_folder = "raw"
    suffix = "_preprocess.pkl"

    means = np.zeros(4)
    stds = np.zeros(4)

    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    args = parser.parse_args()
    
    # Perform actions with the processed line
    input_file = os.path.join(dir, data_folder, args.input_file + suffix)
    
    with open(os.path.join(dir, 'manifest.txt'), 'r') as file:

        # intiantiate a DatasetBuilder which provides utilities for processing data
        brainset_description = BrainsetDescription(
            id="LINK_Dataset_2022_2023",
            origin_version="preprocessing_092024_no7822nofalcon",
            derived_version="1.0.0",
            source="LINK_Dataset",
            description= "179 session files labeled 'YYYY-MM-DD_plotpreprocess.pkl'"
            "Each contains preprocessed neural and behavioral data from one day, along with metadata and trial data"
            "Each contains 400 trials of data for one target style, center-out (CO) or random (RD), or 400 trials "
            "for each of the two target styles if both target styles are present on the same day, under two seperate dictionaries"
            "The trials may contain trials from more than a single run, but only if the best run of the day does "
            "not contain enough trials and there are other runs with the same target style"
            "HOW TO OPEN: data_CO, data_RD = pickle.load(filepath)",
        )

        # open file
        day_data = None
        with open(input_file, 'rb') as f:
            day_data= pickle.load(f) 
            
        recording_date = args.input_file.replace("-","") #remove hyphens, results in str of form YYYYMMDD
        subject = monkey_N_subject()

        # extract experiment metadata
        device_id = f"{subject.id}_{recording_date}"
        
        # register device
        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.UTAH_ARRAY,
        )
        
        ## loop over day data. for each valid taske assign both tcfr and sbp features
        valid_day_data = [i for i in range(2) if day_data[i] is not None]
        for idx in valid_day_data:
            task_data = day_data[idx]
            task = (
                "center_out_reaching" if task_data["target_style"] == "CO" else "random_target_reaching"
            ) 
            
            if task_data["spike_channels"].size > 0:
            
                # register session
                session_id = f"{device_id}_{task}"
                session_description = SessionDescription(
                    id=session_id,
                    recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                    task=Task.REACHING,
                )

                # extract neural data (units, sbp, tcfr)
                # units, ie channels. consider units.select_by_mask(np.array([True, False])) for bad channels
                units = extract_units(task_data)
                
                #sbp and tcfr
                neural_features = extract_spikes(task_data)
                total_spikes += len(task_data["spike_channels"])
                
                # extract behavior
                cursor, cursor_mean, cursor_std = extract_behavior(task_data)
                
                # Stats!
                # means += cursor_mean
                # stds += cursor_std

                data = Data(
                    brainset=brainset_description,
                    subject=subject,
                    session=session_description,
                    device=device_description,
                    # neural activity
                    spikes = neural_features,
                    units=units,
                    # behavior
                    cursor=cursor,
                    # domain
                    domain=cursor.domain, 
                )
            
                train_interval, valid_interval, test_interval = interval_splitter(task_data)
        
                data.set_train_domain(train_interval)
                data.set_valid_domain(valid_interval)
                data.set_test_domain(test_interval)
                

                # save data to disk
                path = os.path.join(args.output_dir, f"{session_id}.h5")
                
                with h5py.File(path, "w") as file:
                    data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

if __name__ == "__main__":
    main()
