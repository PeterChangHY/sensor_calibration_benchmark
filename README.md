# Sensor_calibration_benchmark
  This repository will include the instruction of how to  sensor_calibration_benchmark.
  
## how to generate ground truth data

### imu-vehicle-calibration

check imu's velocity direction v.s. orientation [here](./scripts/find-diff-imu-yaw-to-velocity-yaw.py)

`$ python find-diff-imu-yaw-to-velocity-yaw.py --bag your.bag --output_dir . --sample_rate 0.1 --yaw_th_in_rad 0.005`

### lane-cam-to-imu
1. check golden lane
2. check lane-boundary-connection

## Ground truth format:
 using Comma Separate Values format(.csv) without any header. It should comes with three collumns {Time, Key, value}
 Time: using [Unix Epoch Clock](https://en.wikipedia.org/wiki/Unix_time) format. 
 Key: a plain string
 Value: we have two types of value here
 1. a scalar in flaot ( Note: for boolean value, we use 1.0 as true, 0.0 as false)

 2. a sqaure matrix(row major), we use a *Space Separate Values*, 
 ex. value = 0.01 0.02 0.03 0.04 will represent
      | 2 by 2 Matrix || 
      |--|:-:|
      | 0.01| 0.02|      
      | 0.03| 0.03|

## How to upload sensor_calibration_benchmark
 
1. to data1:



 
 
 

 
 
 
 
 
