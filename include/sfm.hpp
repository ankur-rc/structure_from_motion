#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <glog/logging.h>

#include "camera_pose.hpp"

class SFM
{
public:
    cv::Mat K;                       // camera intrinsics
    std::vector<CameraPose> poses;   // camera poses
    std::vector<Landmark> landmarks; // 3d points

private:
    const int IMAGE_DOWNSAMPLE;  // downsample the image to speed up processing
    const double FOCAL_LENGTH_X; // focal length in pixels, after downsampling, guess from jpeg EXIF data
    const double FOCAL_LENGTH_Y; // focal length in pixels, after downsampling, guess from jpeg EXIF data
    const int MIN_LANDMARK_SEEN; // minimum number of camera views a 3d point (landmark) has to be seen to be used
    const std::string IMAGE_DIR; // Dataset directory

private:
    void initialise_intrinsics();

public:
    SFM(uint downsample,
        double fx, double fy,
        uint landmark_visbility,
        std::string image_directory)
        : IMAGE_DOWNSAMPLE(downsample),
          FOCAL_LENGTH_X(fx),
          FOCAL_LENGTH_Y(fy),
          MIN_LANDMARK_SEEN(landmark_visbility),
          IMAGE_DIR(image_directory) {}

    void find_keypoints();

    void match_keypoints();

    void triangulate_points();

    void visualise_pointcloud(std::string name);

    void bundle_adjust();
};
