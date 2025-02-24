import argparse
import datetime
import logging
import h5py
import os
import pickle

from tqdm import tqdm
import numpy as np
# from pynwb import NWBHDF5IO
# from scipy.ndimage import binary_dilation, binary_erosion

from temporaldata import Data, ArrayDict, RegularTimeSeries, Interval

## this is from the brainsets package, not the local dir
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)

# from brainsets.utils.dandi_utils import ( 
#     extract_spikes_from_nwbfile,
#     extract_subject_from_nwb,
# )

from brainsets.taxonomy import RecordingTech, Task
from brainsets import serialize_fn_map

#from local dir
from monkey_N_utils import monkey_N_subject

logging.basicConfig(level=logging.INFO)

def extract_domain(task_data):
    #as i assume the data is cleaned and binned, let each time bin occupy 1 unit of time
    total_time_bins = np.shape(task_data["sbp"])[0]
    starts = np.array([float(i) for i in range(total_time_bins - 1)])
    ends = np.array([float(i)  for i in range(1,total_time_bins)])
    domain = Interval(starts,ends)
    return domain

def extract_units(task_data):
    #Extract units from the data dictionary.
    
    num_units = np.shape(task_data["sbp"])[1]
    
    ids = np.array(["channel"+str(i) for i in range(num_units)])

    units = ArrayDict(
        unit_id=ids,
        )

    return units

def extract_neural_features(task_data):
    
    neural_features = RegularTimeSeries(
        sbp = task_data["sbp"],
        tcfr = task_data["tcfr"],
        sampling_rate=1., # as the data is already binned, the sample rate of the data if 1 bin. 
        #maybe set to bin rate. the model probably learns the timescale, arbitrary shifting the time scale when incorporating our data with other data of other timescale could be an issue
        domain = "auto"
    )
    return neural_features

def extract_behavior(task_data):
    
    cursor = RegularTimeSeries(
        finger_kinematics = task_data['finger_kinematics'],
        sampling_rate=1., # as the data is already binned, the sample rate of the data if 1 bin
        domain = "auto"
    )
    return cursor


def main():
    #open raw data file manifest
    dir = "./data/raw"
    data_folder = "hisham_big_dataset"
    suffix = "_preprocess.pkl"
    
    #below creates a manifest free of bad days
    #moved dataset to the data/raw folder
    #find hisham_big_dataset/ -type f -name "*.pkl" | sed "s|^hisham_big_dataset/||"  | sed "s|^/*||" | sed 's/_preprocess.pkl$//'> manifest.txt
    #grep -F -v -f hisham_big_dataset/bad_days.txt manifest.txt > filtered_manifest.txt
    #mv filtered_manifest.txt manifest.txt 
    
    pbar = tqdm(total=385, desc="Processing datasets")#hardcoded from manifest
    
    with open(os.path.join(dir, 'manifest.txt'), 'r') as file:
    # Iterate over each line in the file
        for line in file:
            # Strip any trailing whitespace (e.g., newline characters)
            manifest_entry = line.strip()
            
            # Perform actions with the processed line
            input_file = os.path.join(dir, data_folder, manifest_entry + suffix)
    
            # use argparse to get arguments from the command line
            parser = argparse.ArgumentParser()
            
            parser.add_argument("--output_dir", type=str, default=".\data\processed\hisham_big_dataset")

            args = parser.parse_args()

            # intiantiate a DatasetBuilder which provides utilities for processing data
            brainset_description = BrainsetDescription(
                id="napier_dataset",
                origin_version="preprocessing_092024_no7822nofalcon",
                derived_version="1.0.0",
                source="Z:\Student Folders\Hisham_Temmar\big_dataset\2_autotrimming_and_preprocessing\preprocessing_092024_no7822nofalcon",
                description= "439 session files labeled 'YYYY-MM-DD_plotpreprocess.pkl'"
                "Each contains preprocessed neural and behavioral data from one day, along with metadata and trial data"
                "Each contains 400 trials of data for one target style, center-out (CO) or random (RD), or 400 trials "
                "for each of the two target styles if both target styles are present on the same day, under two seperate dictionaries"
                "The trials may contain trials from more than a single run, but only if the best run of the day does "
                "not contain enough trials and there are other runs with the same target style"
                "HOW TO OPEN: data_CO, data_RD = pickle.load(filepath)",
            )

            # logging.info(f"Processing file: {input_file}")

            # open file
            day_data = None
            with open(input_file, 'rb') as f:
                day_data= pickle.load(f) 
                ## each day has at least one, but sometimes both, of CO and RD. The plan 
                ## for days with both is to split them up into different sessions, as the 
                ## inherited setup includes task type in session id. they will still have
                ## the same device ID
                
            # subject metadata. since this is known to by our dear monkey N, no extracting necessary
            subject = monkey_N_subject()

            # extract experiment metadata
            recording_date = manifest_entry.replace("-","") #remove hyphens, results in str of form YYYYMMDD
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
                ) ## check day_data[idx]["target_style"] is actually "CO" for center out
            
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
                neural_features = extract_neural_features(task_data)
                
                # extract behavior
                cursor = extract_behavior(task_data)
                
                domain = extract_domain(task_data)
                
                # register session. assume data has already been cleaned and only valid, succesful trials are left. 
                # as such and data is binned, removed extract trials and cursor outliers
                # consider normalizing data
                data = Data(
                    brainset=brainset_description,
                    subject=subject,
                    session=session_description,
                    device=device_description,
                    # neural activity
                    neural_features = neural_features,
                    units=units,
                    # behavior
                    cursor=cursor,
                    # domain
                    domain=domain, 
                )

            # split trials into train, validation and test
            # for now, keep straight 70/10/20 split, in the future consider randomly slitting the 
            # first 80% of the data by trial into train and val
            
            train_interval, valid_interval, test_interval = data.domain.split([0.7, 0.1, 0.2])
       
            data.set_train_domain(train_interval)
            data.set_valid_domain(valid_interval)
            data.set_test_domain(test_interval)

            # save data to disk
            path = os.path.join(args.output_dir, f"{session_id}.h5")
            
            # with h5py.File(path, "w") as file:
            #     data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
            pbar.update(1)
        pbar.close()

if __name__ == "__main__":
    main()
