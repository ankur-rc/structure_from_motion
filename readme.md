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
$ make -j10
```

Use ```ccmake``` to further tweak options.

### Run
```
$ GLOG_logtostderr=1 ./sfm
```

### Dependency
- OpenCV 3.3
- Eigen 3.3
- Boost 1.69
- Ceres 1.14
- PCL 1.7
