# Sensor_calibration_benchmark
  This repository will include the instruction of how to generate sensor_calibration_benchmark.

# Workflow for adding a new label data
1. find the interesting $data and slice it to smallest bite without degenerating the problem.
2. Carefully do manul/automatical check and put the key and value with timestamp in the $bag.csv.
3. Provide the proof of accuracy of lobeled data and ask other folks to double check it.
4. Upload the bag and related csv to the data1 and share with Sensor calibration team.

# Workflow of adding/modifing new label on current data
1. Carefully do manul/automatical check and adding/modify the key and value with timestamp in the $bag.csv.
2. Provide the proof of accuracy of newly lobeled data and ask other folks to double check it.
3. Update the new .csv on the data1 and share with Sensor calibration team.

## Output/groud truth format:
 using Comma Separate Values format(.csv) without any header. 
 
 It should comes with three columns {**Time**, **Key**, **value**}
 
 **Time**: using [Unix Epoch Clock](https://en.wikipedia.org/wiki/Unix_time) format. 
 
 **Key**: any plain string ex. SENSOR_CALIB_EXTRINSIC_PARAM_IMU_OFF
 
 For sensor calibration check, we will follow the name in [drive/monitor](https://github.com/PlusAI/drive/blob/master/protos/monitor/status_report_msg.proto)
 
 *please make sure the key is properly naming, everyone agrees its meaning before really adding it and add description in [key_description](./doc/key_description.md)
 
 **Value**: three types of value
 1. a scalar, using a float number
 
 2. a boolean, we use 1.0 as true, 0.0 as false

 3. a sqaure matrix(row major), we use a *Space Separate Values*, 
 ex. value = '0.01 0.02 0.03 0.04' will represent
      | 2 by 2 Matrix || 
      |--|:-:|
      | 0.01| 0.02|      
      | 0.03| 0.04|

## Folder structure
every bag should comes with a .csv file with the same basename of the bag in the same directory
we use *vehicle name* as the first sub folders.
```
sensor_calibraton_benchmark 
│
└───paccar-k001dm
│   │─── A.bag
│   │─── A.csv
│   
└───petebilf-sif
│   │─── B.bag
│   │─── B.csv
```

## How to upload sensor_calibration_benchmark

put your data at @data1:/mnt/vault0/peter/sensor_calibration_benchmark/  

If you will need to use the benchmark in replay simulator or other places.

put you data at @data1:/mnt/vault0/simulator_bags/sensor_calibration_benchmark/ and it will automatically upload to /work/data/benchmark-bags/sensor_calibration_benchmark/ in all CI machines



## How to check the data

### imu-vehicle-calibration

check imu's velocity direction v.s. orientation yaw [code](./scripts/find-diff-imu-yaw-to-velocity-yaw.py)
```bash
$ python find-diff-imu-yaw-to-velocity-yaw.py --bags your.bag --yaw_th_in_rad 0.0085
```

### lane-cam-to-imu
1. check golden lane using [calibration_viewer](./scripts/calibration_viewer/README.md)
2. check lane-boundary-connection




 
 
 

 
 
 
 
 
