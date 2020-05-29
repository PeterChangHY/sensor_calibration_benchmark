# Sensor_calibration_benchmark
  This repository will provide the necessary instructions and tools to generate sensor_calibration_benchmark.
  This benchmark should be 
  1. general -> not only for certain sensor_calibration
  2. extendable -> we can reuse the data for other application and add/modify labels when we need to
  



# Workflow for adding a new label data
1. Find the interesting bag and slice it to smallest bite without degenerating the problem.
2. Carefully do manul/automatic check and put the key and value with timestamp in the bag.csv.
3. Provide the proof of accuracy of lobeled data and ask other folks to double check it.
4. Upload the bag and related csv and share with Sensor calibration team.

# Workflow of adding/modifing new label on current data
1. Carefully do manul/automatic check and adding/modify the key and value with timestamp in the bag.csv.
2. Provide the proof of accuracy of newly lobeled data and ask other folks to double check it.
3. Upload the new .csv and share with Sensor calibration team.

## Output/groud truth format:
 using Comma Tab Values format(.tsv) without any header. 
 
 It should comes with three columns {**UnixTime**, **Key** ,**Value**}
 
 **UnixTime**: using [Unix Epoch Clock](https://en.wikipedia.org/wiki/Unix_time) format. 
 
 **Key**: any plain string e.g. SENSOR_CALIB_EXTRINSIC_PARAM_IMU_OFF
 
 For sensor calibration check, we will follow the name in [drive/monitor](https://github.com/PlusAI/drive/blob/master/protos/monitor/status_report_msg.proto)
 
 *please make sure the key is properly naming, everyone agrees its meaning before really adding it and add description in [key_description](./doc/key_description.md)
 
 **Value**: three types of value
 1. a scalar, using a single float number
 
 2. a boolean, we use 'True' or 'False'(case insensitive)

 3. a matrix(row major), we use a *comma Separate Values*, first two value are number of row and number of col. 
 ex. value = '2, 2, 0.01, 0.02, 0.03, 0.04' will represent
      | 2 by 2 Matrix || 
      |--|:-:|
      | 0.01| 0.02|      
      | 0.03| 0.04|

## Folder structure
Every bag should comes with a .csv file with the same basename and both are in the same directory

Using *vehicle name* as the first sub folders.
```
sensor_calibraton_benchmark 
│
└───paccar-k001dm
    └─── A.bag
│        │─── A.bag
│        │─── A.bag.tsv
│   
└───petebilf-sif
│   │─── B.bag
│   │─── B.bag.tsv
```

## How to upload sensor_calibration_benchmark

If you want to share the data

put your data at 

`@data1:/mnt/vault0/peter/sensor_calibration_benchmark/`

If you will need to use the benchmark in replay simulator or other places.

put you data at 

`@data1:/mnt/vault0/simulator_bags/sensor_calibration_benchmark/` 

and it will automatically upload to 

`/work/data/benchmark-bags/sensor_calibration_benchmark/`

ref: [data-sync](https://github.com/PlusAI/data-sync/blob/master/conf/sync.yaml#L52)



## How to check the data

### imu-vehicle-calibration

check imu's velocity direction v.s. orientation yaw [code](./scripts/find-diff-imu-yaw-to-velocity-yaw.py)
```bash
$ python find-diff-imu-yaw-to-velocity-yaw.py --bags your.bag --yaw_th_in_rad 0.0085
```

### lane-cam-to-imu
1. check golden lane using [calibration_viewer](./scripts/calibration_viewer/README.md)
2. check lane-boundary-connection


## Generate a report

### How to setup envrionment for reporter

```bash
cd scripts
./setup.sh
source .venv/bin/activate
```

```base
python reporter/reporter.py --baseline_dir ~/work/sensor_calibration_benchmark/test/baseline_tsv_dir/ --groundtruth_dir ~/work/sensor_calibration_benchmark/test/gt_tsv_dir/ --target_dir ~/work/sensor_calibration_benchmark/test/target_tsv_dir/ --key SENSOR_CALIB_EXTRINSIC_PARAM_IMU_OFF --report_type sensor_calib_checker --output_dir ~/work/sensor_calibration_benchmark/test/output_2
```



 
 
 

 
 
 
 
 
