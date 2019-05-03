#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct SnavelyReprojectionError
{
    SnavelyReprojectionError(const double observed_x,
                             const double observed_y,
                             const double focal,
                             const double cx,
                             const double cy)
        : observed_x(observed_x), observed_y(observed_y), focal(focal), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T *const _T,
                    const T *const landmark_x,
                    const T *const landmark_y,
                    const T *const landmark_z,
                    T *residuals) const
    {
        // R_inverse = R.transpose
        T R_inv[9] = {_T[0], _T[4], _T[8], _T[1], _T[5], _T[9], _T[2], _T[6], _T[10]};

        // [X', Y', Z'] = [X-t_x, Y-t_y, Z-t_z]
        T pt3d[3] = {landmark_x[0] - _T[3], landmark_y[0] - _T[7], landmark_z[0] - _T[11]};
        // [X'', Y'', Z''] = R_inverse.[X', Y', Z']
        T p[3] = {R_inv[0] * pt3d[0] + R_inv[1] * pt3d[1] + R_inv[2] * pt3d[2],
                  R_inv[3] * pt3d[0] + R_inv[4] * pt3d[1] + R_inv[5] * pt3d[2],
                  R_inv[6] * pt3d[0] + R_inv[7] * pt3d[1] + R_inv[8] * pt3d[2]};

        // Normalize homogeneous coordinates
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Compute final projected point position.
        T predicted_x = focal * xp + cx;
        T predicted_y = focal * yp + cy;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y,
                                       const double focal,
                                       const double cx,
                                       const double cy)
    {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 16, 1, 1, 1>(
            new SnavelyReprojectionError(observed_x, observed_y, focal, cx, cy)));
    }

    double observed_x;
    double observed_y;
    double focal, cx, cy;
};