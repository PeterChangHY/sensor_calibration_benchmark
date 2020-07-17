
Calibration_viewer

```bash
$ cd scripts
$ python -m calibration_viewer/calibration_viewer --bag your.bag --cam_topic left_camera right_camera --cam_calibs front_left_camera.yml front_right_camera.yaml --output_dir ~/Work/test/calibration_viewer/ --show_golden_lane
```

top down view
------------------
top-down view view ( lidar and radar)
![calibration_view_birdeye](img/calibration_viewer_birdeye.gif)

stitched_image
----------------
stitched image
![calibration_view_stitch_img](img/calibration_viewer_stitched.gif)

stitched cylindrical image
![stitched_cylindrical_view](img/stitched_cylindrical_view.gif)

Golden_lane
-----------
golden lane should match the ego lane boundary when we drove in the middle and along with lane direction

![golden_lane](img/golden_lane.png)

