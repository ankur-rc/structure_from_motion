#pragma once

#include <iostream>
#include <vector>
#include <map>

#include <opencv2/core.hpp>

struct CameraPose
{

    cv::Mat T;                           // transform [R|t]
    cv::Mat P;                           // projection K*T
    cv::Mat image;                       // downsampled image
    std::vector<cv::KeyPoint> keypoints; // list of keypoints
    cv::Mat descriptors;                 // matrix of keypoint descriptors

    std::map<size_t, std::map<size_t, size_t>> keypoint_match; // maps current camera's keypoint id to other cameras' keypoint id
    std::map<size_t, size_t> keypoint_landmark;                // maps a keypoint id in current camera to a landmark id

    bool is_keypoint_exists(size_t keypoint_id, size_t camera_index)
    {
        return keypoint_match[keypoint_id].count(camera_index) > 0;
    }

    const size_t &get_keypoint_match(size_t keypoint_id, size_t camera_index)
    {
        return keypoint_match[keypoint_id][camera_index];
    }

    bool is_keypoint_landmark(size_t keypoint_id)
    {
        return keypoint_landmark.count(keypoint_id) > 0;
    }

    size_t &get_keypoint_landmark(size_t keypoint_id)
    {
        return keypoint_landmark[keypoint_id];
    }
};

struct Landmark
{
    cv::Point3d pt3d;
    cv::Vec3b color;
    size_t id;
    uint visible = 0;
};