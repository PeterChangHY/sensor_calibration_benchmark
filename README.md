# Sensor_calibration_benchmark
  This repository will include the instruction of how to  sensor_calibration_benchmark.
  
## how to generate ground truth data

### imu-vehicle-calibration

check imu's velocity direction v.s. orientation yaw [code](./scripts/find-diff-imu-yaw-to-velocity-yaw.py)
```bash
$ python find-diff-imu-yaw-to-velocity-yaw.py --bags your.bag --yaw_th_in_rad 0.0085
```

### lane-cam-to-imu
1. check golden lane using [calibration_viewer](./scripts/calibration_viewer/README.md)
2. check lane-boundary-connection

## Ground truth format:
 using Comma Separate Values format(.csv) without any header. 
 
 It should comes with three columns {**Time**, **Key**, **value**}
 
 **Time**: using [Unix Epoch Clock](https://en.wikipedia.org/wiki/Unix_time) format. 
 
 **Key**: any plain string ex. SENSOR_CALIB_EXTRINSIC_PARAM_IMU_OFF
 
 **Value**: we have two types of value here
 1. a float number( Note: for boolean value, we use 1.0 as true, 0.0 as false)

 2. a sqaure matrix(row major), we use a *Space Separate Values*, 
 ex. value = '0.01 0.02 0.03 0.04' will represent
      | 2 by 2 Matrix || 
      |--|:-:|
      | 0.01| 0.02|      
      | 0.03| 0.03|

## How to upload sensor_calibration_benchmark




 
 
 

 
 
 
 
 
