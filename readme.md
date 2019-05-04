## Structure from Motion

Given a sequence of images, estimates the 3D structure of the scene and the relative motion between each image.

### Compile

```
$ mkdir build && cd build
```

```
$ cmake .. 
```

``` 
$ make -j
```

Use ```ccmake``` to further tweak options.

### Usage
```
$ usage: GLOG_logtostderr=1 ./sfm downsample fx fy dataset
                   
  downsample: (int) scaling factor for images (increases performance for values greater than 1)
  fx        : (double) focal length in 'px' calculated as image_width(px)*focal_length(mm)/sensor_width(mm)
  fy        : (double) focal length in 'px' calculated as image_height(px)*focal_length(mm)/sensor_height(mm)
  dataset   : (string) path to dataset directory
  
```

### Dependency
- OpenCV 3.3
- Eigen 3.3
- Boost 1.69
- Ceres 1.14
- PCL 1.7
