#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <boost/filesystem.hpp>

#include <pcl/visualization/cloud_viewer.h>

#include <glog/logging.h>

#include "camera_pose.hpp"
#include "sfm.hpp"
#include "bundle_adjust.hpp"

void SFM::initialise_intrinsics()
{
    double cx = poses[0].image.size().width / 2;
    double cy = poses[0].image.size().height / 2;

    K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = FOCAL_LENGTH_X / IMAGE_DOWNSAMPLE;
    K.at<double>(1, 1) = FOCAL_LENGTH_Y / IMAGE_DOWNSAMPLE;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 2) = cy;

    LOG(INFO) << "Camera intrinsics are: " << std::endl
              << K << std::endl;
}

void SFM::visualise_pointcloud(std::string name = "Point Cloud")
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (const auto &landmark : landmarks)
    {
        if (landmark.visible >= MIN_LANDMARK_SEEN)
        {
            pcl::PointXYZRGB pt;
            pt.x = landmark.pt3d.x;
            pt.y = landmark.pt3d.y;
            pt.z = landmark.pt3d.z;
            pt.b = landmark.color[0];
            pt.g = landmark.color[1];
            pt.r = landmark.color[2];
            cloud->push_back(pt);
        }
    }

    pcl::visualization::CloudViewer viewer(name);
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {
    }
}

void SFM::find_keypoints()
{

    using namespace boost::filesystem;
    using namespace cv;
    using namespace std;

    path image_directory = path(IMAGE_DIR);
    if (!is_directory(image_directory))
    {
        LOG(FATAL) << "Please specify a path to a directory!" << endl;
    }

    Ptr<cv::AKAZE> feature_extractor = cv::AKAZE::create();

    for (const auto &file : directory_iterator(image_directory))
    {
        if (!is_regular_file(file))
            continue;

        Mat img = imread(file.path().string());
        assert(!img.empty());

        resize(img, img, img.size() / IMAGE_DOWNSAMPLE);

        CameraPose pose;
        pose.image = img;

        cvtColor(img, img, COLOR_BGR2GRAY);

        vector<KeyPoint> kps;
        Mat descriptors;
        feature_extractor->detectAndCompute(img, noArray(), kps, descriptors);

        pose.keypoints = move(kps);
        pose.descriptors = move(descriptors);

        poses.emplace_back(pose);
    }

    LOG(INFO) << "Found " << poses.size() << " cameras." << endl;
}

void SFM::match_keypoints()
{
    using namespace cv;
    using namespace std;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    const char *window_name = "Keypoint matches";
    namedWindow(window_name, WINDOW_NORMAL);

    for (size_t i = 0; i < poses.size(); i++)
    {
        CameraPose &src_pose = poses[i];
        for (size_t j = i + 1; j < poses.size(); j++)
        {
            CameraPose &dst_pose = poses[j];

            vector<Point2f> src_2d, dst_2d;  // temporary holder for 2d image points; will be refined gradually
            vector<vector<DMatch>> matches;  // holder for keypoint match pairs
            vector<size_t> src_kps, dst_kps; // holder for keypoints; will be refined

            matcher->knnMatch(src_pose.descriptors, dst_pose.descriptors, matches, 2);
            for (const auto &match : matches)
            {
                // check goodness; if criteria met add to map of keypoint matches
                if (match[0].distance < 0.7 * match[1].distance)
                {
                    // store the corresponding 2d points
                    src_2d.emplace_back(src_pose.keypoints[match[0].queryIdx].pt);
                    dst_2d.emplace_back(dst_pose.keypoints[match[0].trainIdx].pt);

                    // store the corresponding keypoints
                    src_kps.emplace_back(match[0].queryIdx);
                    dst_kps.emplace_back(match[0].trainIdx);
                }
            }

            vector<unsigned char> outlier_mask; // mask bad indices
            // refine further; check if points meet the fundamental matrix constraint
            findFundamentalMat(src_2d, dst_2d, CV_FM_RANSAC, 3.0, 0.9, outlier_mask);

            Mat canvas = src_pose.image.clone();
            canvas.push_back(dst_pose.image.clone());

            for (size_t k = 0; k < outlier_mask.size(); k++)
            {
                if (outlier_mask[k])
                {
                    src_pose.keypoint_match[src_kps[k]][j] = dst_kps[k];
                    dst_pose.keypoint_match[dst_kps[k]][i] = src_kps[k];

                    line(canvas, src_2d[k], dst_2d[k] + Point2f(0, src_pose.image.rows), Scalar(0, 0, 255));
                }
            }

            LOG(INFO) << "Matches between Camera " << i << " --> Camera " << j << " := "
                      << sum(outlier_mask)[0] << "/" << src_pose.keypoints.size() << endl;

            imshow(window_name, canvas);
            waitKey(1);
        }
    }

    destroyAllWindows();
}

void SFM::triangulate_points()
{
    using namespace cv;
    using namespace std;

    initialise_intrinsics();

    // fix the first camera as the origin
    poses[0].T = Mat::eye(4, 4, CV_64F);
    poses[0].P = K * Mat::eye(3, 4, CV_64F);

    const char *window_name = "Keypoint matches after triangulation";
    namedWindow(window_name, WINDOW_NORMAL);

    // random number seeding
    // random_device rd;      // obtain a random number from hardware
    // mt19937 rnd_eng(rd()); // seed the generator

    for (size_t i = 0; i < poses.size() - 1; i++)
    {
        CameraPose &prev = poses[i];
        CameraPose &curr = poses[i + 1];

        // accumulate matching keypoints
        vector<Point2f> src_2d, dst_2d;
        vector<size_t> kp_used;

        for (size_t j = 0; j < prev.keypoints.size(); j++)
        {
            if (prev.is_keypoint_exists(j, i + 1))
            {
                size_t curr_kp_idx = prev.get_keypoint_match(j, i + 1);
                src_2d.emplace_back(prev.keypoints[j].pt);
                dst_2d.emplace_back(curr.keypoints[curr_kp_idx].pt);

                kp_used.emplace_back(j);
            }
        }

        if (kp_used.size() < 7)
        {
            LOG(FATAL) << "Image " << i << " and image " << i + 1 << " have only " << kp_used.size()
                       << " matching points. Need atleast 7 to compute the pose between them. Exiting..." << endl;
        }

        // NOTE: pose from dst to src: C(n-1)->C(n)... (transformation bw the previous and the current camera)
        Mat outlier_mask;
        Mat E, local_R, local_t;
        E = findEssentialMat(dst_2d, src_2d, K, RANSAC, 0.999, 1.0, outlier_mask);

        recoverPose(E, dst_2d, src_2d, K, local_R, local_t, outlier_mask);

        // camera center transformation
        Mat local_T = Mat::eye(4, 4, CV_64F);
        local_R.copyTo(local_T(Range(0, 3), Range(0, 3)));
        local_t.copyTo(local_T(Range(0, 3), Range(3, 4)));

        // total transformation: C0->C1->C2... (transformation bw the first and the current camera)
        curr.T = prev.T * local_T;

        // projection matrix
        Mat R = curr.T(Range(0, 3), Range(0, 3));
        Mat t = curr.T(Range(0, 3), Range(3, 4));
        Mat P(3, 4, CV_64F);

        P(Range(0, 3), Range(0, 3)) = R.t();
        P(Range(0, 3), Range(3, 4)) = -R.t() * t;
        P = K * P; // projection matrix

        curr.P = P;

        // trinagulate points
        Mat landmarks_4d;
        triangulatePoints(prev.P, curr.P, src_2d, dst_2d, landmarks_4d);

        Mat canvas = prev.image.clone();
        canvas.push_back(curr.image.clone());

        // attempt to rescale transformation if current landmark is visible in previous frames
        if (i > 0)
        {
            double scale = 0;
            int count = 0;

            Point3d prev_camera;

            prev_camera.x = prev.T.at<double>(0, 3);
            prev_camera.y = prev.T.at<double>(1, 3);
            prev_camera.z = prev.T.at<double>(2, 3);

            vector<Point3d> new_pts;
            vector<Point3d> existing_pts;

            for (size_t j = 0; j < kp_used.size(); j++)
            {
                size_t k = kp_used[j];
                if (outlier_mask.at<unsigned char>(j) && prev.is_keypoint_exists(k, i + 1) && prev.is_keypoint_landmark(k))
                {
                    Point3d landmark_3d;

                    landmark_3d.x = landmarks_4d.at<float>(0, j) / landmarks_4d.at<float>(3, j);
                    landmark_3d.y = landmarks_4d.at<float>(1, j) / landmarks_4d.at<float>(3, j);
                    landmark_3d.z = landmarks_4d.at<float>(2, j) / landmarks_4d.at<float>(3, j);

                    size_t idx = prev.get_keypoint_landmark(k);
                    Point3d avg_landmark;
                    avg_landmark.x = landmarks[idx].pt3d.x / static_cast<uint>(landmarks[idx].visible - 1);
                    avg_landmark.y = landmarks[idx].pt3d.y / static_cast<uint>(landmarks[idx].visible - 1);
                    avg_landmark.z = landmarks[idx].pt3d.z / static_cast<uint>(landmarks[idx].visible - 1);

                    new_pts.push_back(landmark_3d);
                    existing_pts.push_back(avg_landmark);
                }
            }

            // ratio of distance for all possible point pairing
            size_t samples = new_pts.size() - 1;
            if (samples > 0)
            {
                // uniform_int_distribution<> dist_j(0, samples);
                // uniform_int_distribution<> dist_k(0, samples);

                for (size_t j = 0; j < samples; j++)
                {
                    // int j_idx = dist_j(rnd_eng);
                    for (size_t k = j + 1; k < samples; k++)
                    {
                        // int k_idx = dist_k(rnd_eng);
                        double s = norm(existing_pts[j] - existing_pts[k]) / norm(new_pts[j] - new_pts[k]);

                        scale += s;
                        count++;
                    }
                }
            }

            // DLOG(INFO) << "samples: " << samples << "\tcount: " << count << "\tscale: " << scale << endl;

            if (count > 0)
            {
                scale /= count;

                LOG(INFO) << "image " << (i + 1) << " ==> " << i << " scale=" << scale << " count=" << count << endl;

                // apply scale and re-calculate T and P matrix
                local_t *= scale;

                // local tansform
                Mat T = Mat::eye(4, 4, CV_64F);
                local_R.copyTo(T(Range(0, 3), Range(0, 3)));
                local_t.copyTo(T(Range(0, 3), Range(3, 4)));

                // accumulate transform
                curr.T = prev.T * T;

                // make projection ,matrix
                R = curr.T(Range(0, 3), Range(0, 3));
                t = curr.T(Range(0, 3), Range(3, 4));

                Mat P(3, 4, CV_64F);
                P(Range(0, 3), Range(0, 3)) = R.t();
                P(Range(0, 3), Range(3, 4)) = -R.t() * t;
                P = K * P;

                curr.P = P;
                triangulatePoints(prev.P, curr.P, src_2d, dst_2d, landmarks_4d);
            }
        }

        // Find good triangulated points
        for (size_t j = 0; j < kp_used.size(); j++)
        {
            if (outlier_mask.at<unsigned char>(j))
            {
                size_t k = kp_used[j];
                size_t kp_match = prev.get_keypoint_match(k, i + 1);

                Point3d landmark_3d;

                landmark_3d.x = landmarks_4d.at<float>(0, j) / landmarks_4d.at<float>(3, j);
                landmark_3d.y = landmarks_4d.at<float>(1, j) / landmarks_4d.at<float>(3, j);
                landmark_3d.z = landmarks_4d.at<float>(2, j) / landmarks_4d.at<float>(3, j);

                if (prev.is_keypoint_landmark(k))
                {
                    // Found a match with an existing landmark
                    curr.get_keypoint_landmark(kp_match) = prev.get_keypoint_landmark(k);

                    landmarks[prev.get_keypoint_landmark(k)].pt3d += landmark_3d;
                    landmarks[prev.get_keypoint_landmark(k)].color = curr.image.at<Vec3b>(curr.keypoints[kp_match].pt);
                    landmarks[curr.get_keypoint_landmark(kp_match)].visible++; // same as "landmarks[prev.get_keypoint_landmark(k)].seen++;"
                }
                else
                {
                    // Add new 3d point
                    Landmark landmark;

                    landmark.pt3d = landmark_3d;
                    landmark.color = curr.image.at<Vec3b>(curr.keypoints[kp_match].pt);
                    // landmark.color += prev.image.at<Point3i>(prev.keypoints[k].pt);
                    landmark.visible = 2;

                    landmarks.emplace_back(landmark);

                    prev.get_keypoint_landmark(k) = landmarks.size() - 1;
                    curr.get_keypoint_landmark(kp_match) = landmarks.size() - 1;
                }

                line(canvas, src_2d[j], dst_2d[j] + Point2f(0, prev.image.rows), Scalar(0, 0, 255));
                // imshow(window_name, canvas);
                // waitKey(1);
            }
        }
    }

    destroyAllWindows();

    // Avg out the values for the landmarks
    for (auto &landmark : landmarks)
    {
        if (landmark.visible > 2)
        {
            landmark.pt3d.x /= (landmark.visible - 1);
            landmark.pt3d.y /= (landmark.visible - 1);
            landmark.pt3d.z /= (landmark.visible - 1);
        }
    }
}

void SFM::bundle_adjust()
{
    using namespace ceres;

    Problem problem;

    for (auto &pose : poses)
    {
        auto &T = pose.T;
        assert(T.isContinuous());

        for (const auto &map : pose.keypoint_landmark)
        {
            const auto kp_id = map.first;
            const auto landmark_id = map.second;

            auto &landmark = landmarks[landmark_id];
            if (landmark.visible < MIN_LANDMARK_SEEN)
                continue;

            const auto kp = pose.keypoints[kp_id];

            // prepare datapoints
            const auto obs_x = kp.pt.x;
            const auto obs_y = kp.pt.y;
            const auto focal = K.at<double>(0, 0);
            const auto cx = K.at<double>(0, 2);
            const auto cy = K.at<double>(1, 2);

            CostFunction *cost_function =
                SnavelyReprojectionError::Create(obs_x, obs_y, focal, cx, cy);
            problem.AddResidualBlock(cost_function,
                                     NULL /* squared loss */,
                                     T.ptr<double>(),
                                     &(landmark.pt3d.x),
                                     &(landmark.pt3d.y),
                                     &(landmark.pt3d.z));
        }
    }
    Solver::Options options;
    options.linear_solver_type = DENSE_SCHUR;
    options.max_num_iterations = 50;
    options.use_explicit_schur_complement = true;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);
    LOG(INFO) << summary.FullReport() << "\n";
}

void SFM::generate_for_pmvs2()
{
    /* WIP*/
    auto K_ = K.clone();
    K_.at<double>(0, 0) *= IMAGE_DOWNSAMPLE;
    K_.at<double>(1, 1) *= IMAGE_DOWNSAMPLE;
    K_.at<double>(0, 2) *= IMAGE_DOWNSAMPLE;
    K_.at<double>(1, 2) *= IMAGE_DOWNSAMPLE;
}