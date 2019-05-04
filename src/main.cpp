#include <iostream>
#include <string>

#include <glog/logging.h>

#include "sfm.hpp"

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "Starting..." << std::endl;

    if (argc < 5)
    {
        LOG(ERROR) << R"(
            usage: GLOG_logtostderr=1 ./sfm downsample fx fy dataset
                   
                   downsample: (int) scaling factor for images (increases performance for values greater than 1)
                   fx        : (double) focal length in 'px' calculated as image_width(px)*focal_length(mm)/sensor_width(mm)
                   fy        : (double) focal length in 'px' calculated as image_height(px)*focal_length(mm)/sensor_height(mm)
                   dataset   : (string) path to dataset directory)";

        return 1;
    }

    int downsample = std::stoi(argv[1]);
    double fx = std::stod(argv[2]);
    double fy = std::stod(argv[3]);
    std::string dataset = argv[4];

    // int downsample = 4;
    // double fx, fy;
    // fx = (4368 * 24 / 36);
    // fy = (2912 * 24 / 24);
    // std::string dataset = "/media/ankurrc/new_volume/sfm/datasets/building";
    // fx = (5472 * 10.4 / 13.20) / downsample;
    // fy = (3648 * 10.4 / 8.80) / downsample;
    // std::string dataset = "/media/ankurrc/new_volume/sfm/datasets/desk";

    SFM sfm(downsample, fx, fy, 3, dataset);
    sfm.find_keypoints();
    sfm.match_keypoints();
    sfm.triangulate_points();
    sfm.visualise_pointcloud("Triangulated Points: Before bundle adjustment");
    sfm.bundle_adjust();
    sfm.visualise_pointcloud("Triangulated Points: After bundle adjustment");

    LOG(INFO) << "...done." << std::endl;
    return 0;
}