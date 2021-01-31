#ifndef INCLUDE_FREETURE_MATCHER_MATCHER_H_
#define INCLUDE_FREETURE_MATCHER_MATCHER_H_

#include <open3d/pipelines/registration/Registration.h>
#include <ros/ros.h>

#include <vector>

#include "freeture_matcher/common.h"

namespace voxblox {
class Matcher {
 public:
  struct Config {
    Config() {}
    bool verbose = false;
    float max_corr_distance = 1.5;
    float k_dist = 0.9;
    int n_max_ransac = 10000;
    float max_ransac_valid = 0.9;

    friend inline std::ostream& operator<<(std::ostream& s, const Config& v) {
      s << std::endl
        << "Matcher  using Config:" << std::endl
        << "  verbose: " << v.verbose << std::endl
        << "  max_corr_distance: " << v.max_corr_distance << std::endl
        << "  k_dist: " << v.k_dist << std::endl
        << "  n_max_ransac: " << v.k_dist << std::endl
        << "  max_ransac_valid: " << v.max_ransac_valid << std::endl
        << "-------------------------------------------" << std::endl;
      return (s);
    }
  };

  static Config getConfigFromRosParam(const ros::NodeHandle& nh_private) {
    Config config;
    nh_private.param("verbose", config.verbose, config.verbose);
    nh_private.param<float>("max_corr_distance", config.max_corr_distance,
                            config.max_corr_distance);
    nh_private.param("k_dist", config.k_dist, config.k_dist);
    nh_private.param("n_max_ransac", config.n_max_ransac, config.n_max_ransac);
    nh_private.param("max_ransac_valid", config.max_ransac_valid,
                     config.max_ransac_valid);
    return config;
  }

  Matcher() {}
  virtual ~Matcher() = default;

  bool matchPointClouds(
      const PointcloudV& point_cloud_q, const FeatureMatrix& features_q,
      const PointcloudV& point_cloud_t, const FeatureMatrix& features_t,
      open3d::pipelines::registration::RegistrationResult* result) {
    if (features_q.empty() || features_t.empty()) return false;
    int n = features_q.size(), dim = features_q[0].size();
    Eigen::MatrixXd feature_matrix_q(dim, n);
    for (int i = 0; i < features_q.size(); i++)
      feature_matrix_q.col(i) = features_q[i];
    O3dFeature o3d_feature_q;
    o3d_feature_q.data_ = feature_matrix_q;

    n = features_t.size();
    dim = features_t[0].size();
    Eigen::MatrixXd feature_matrix_t(dim, n);
    for (int i = 0; i < features_t.size(); i++)
      feature_matrix_t.col(i) = features_t[i];
    O3dFeature o3d_feature_t;
    o3d_feature_t.data_ = (feature_matrix_t);

    std::vector<std::reference_wrapper<
        const open3d::pipelines::registration::CorrespondenceChecker>>
        checkers;
    checkers.emplace_back(
        open3d::pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(
            config_.k_dist));
    checkers.emplace_back(
        open3d::pipelines::registration::CorrespondenceCheckerBasedOnDistance(
            config_.max_corr_distance));

    *result = open3d::pipelines::registration::
        RegistrationRANSACBasedOnFeatureMatching(
            point_cloud_q, point_cloud_t, o3d_feature_q, o3d_feature_t, false,
            config_.max_corr_distance,
            open3d::pipelines::registration::
                TransformationEstimationPointToPoint(false),
            3, checkers,
            open3d::pipelines::registration::RANSACConvergenceCriteria(
                config_.n_max_ransac, config_.max_ransac_valid));
    LOG(INFO) << "registration result: fitness: " << result->fitness_
              << ", inliner_rmse: " << result->inlier_rmse_;

    return true;
  }

  const Config config_;
};
}  // namespace voxblox

#endif  // INCLUDE_FREETURE_MATCHER_MATCHER_H_
