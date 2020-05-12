# Sensor_calibration_benchmark
  This repository will include the instruction of how to generate sensor_calibration_benchmark.

# Workflow for adding a new bag-ground-truth
1. find the interesting $data and slice it to smallest bite without degenerating the problem.
2. Carefully do manul/automatical check and put the key and value with timestamp in the $bag.csv.
3. Provide the proof of accuracy of lobeled data and ask other folks to double check it.
4. Upload to the data1 and share with Sensor calibration team.

# Workflow for adding new ground truch on current data
1. Carefully do manul/automatical check and adding/modify the key and value with timestamp in the $bag.csv.
2. Provide the proof of accuracy of newly lobeled data and ask other folks to double check it.
3. Upload to the data1 and share with Sensor calibration team.

## how to generate ground truth data

### imu-vehicle-calibration

check imu's velocity direction v.s. orientation yaw [code](./scripts/find-diff-imu-yaw-to-velocity-yaw.py)
```bash
$ python find-diff-imu-yaw-to-velocity-yaw.py --bags your.bag --yaw_th_in_rad 0.0085
```

### lane-cam-to-imu
1. check golden lane using [calibration_viewer](./scripts/calibration_viewer/README.md)
2. check lane-boundary-connection

## Output/groud truth format:
 using Comma Separate Values format(.csv) without any header. 
 
 It should comes with three columns {**Time**, **Key**, **value**}
 
 **Time**: using [Unix Epoch Clock](https://en.wikipedia.org/wiki/Unix_time) format. 
 
 **Key**: any plain string ex. SENSOR_CALIB_EXTRINSIC_PARAM_IMU_OFF
 
 For sensor calibrtion check we will follow the name in [drive/monitor](https://github.com/PlusAI/drive/blob/master/protos/monitor/status_report_msg.proto)
 
 *please make sure the key is properly naming, everyone agrees its meaning before really adding it and add description in [here][./doc/key_description.md]
 
 **Value**: we have Three values of value here
 1. a float number
 
 2. a boolean value , we use 1.0 as true, 0.0 as false

 3 a sqaure matrix(row major), we use a *Space Separate Values*, 
 ex. value = '0.01 0.02 0.03 0.04' will represent
      | 2 by 2 Matrix || 
      |--|:-:|
      | 0.01| 0.02|      
      | 0.03| 0.03|

## How to upload sensor_calibration_benchmark

TODO


 
 
 

 
 
 
 
 
