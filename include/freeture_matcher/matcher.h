#ifndef INCLUDE_FREETURE_MATCHER_MATCHER_H_
#define INCLUDE_FREETURE_MATCHER_MATCHER_H_

#include <Open3D/Registration/Registration.h>
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

  explicit Matcher(const ros::NodeHandle& nh_private)
      : config_(getConfigFromRosParam(nh_private)) {}
  virtual ~Matcher() = default;

  bool featureMatrixToO3dFeature(const FeatureMatrix& features,
                                 O3dFeature* o3d_features) {
    if (features.empty()) return false;
    int n = features.size(), dim = features[0].size();
    Eigen::MatrixXd feature_matrix_q(dim, n);
    for (int i = 0; i < features.size(); i++)
      feature_matrix_q.col(i) = features[i];
    o3d_features->data_ = feature_matrix_q;
    return true;
  }

  bool matchPointClouds(const PointcloudV& point_cloud_q,
                        const O3dFeature& features_q,
                        const PointcloudV& point_cloud_t,
                        const O3dFeature& features_t,
                        RegistrationResult* result) {
    std::vector<std::reference_wrapper<
        const open3d::registration::CorrespondenceChecker>>
        checkers;
    auto checker_edge_length =
        open3d::registration::CorrespondenceCheckerBasedOnEdgeLength(
            config_.k_dist);
    auto checker_distance =
        open3d::registration::CorrespondenceCheckerBasedOnDistance(
            config_.max_corr_distance);
    checkers.push_back(checker_edge_length);
    checkers.push_back(checker_distance);

    *result = open3d::registration::RegistrationRANSACBasedOnFeatureMatching(
        point_cloud_q, point_cloud_t, features_q, features_t,
        config_.max_corr_distance,
        open3d::registration::TransformationEstimationPointToPoint(false), 3,
        checkers,
        open3d::registration::RANSACConvergenceCriteria(
            config_.n_max_ransac, config_.max_ransac_valid));
    LOG(INFO) << "registration result: fitness: " << result->fitness_
              << ", inliner_rmse: " << result->inlier_rmse_;

    return true;
  }

  const Config config_;
};
}  // namespace voxblox

#endif  // INCLUDE_FREETURE_MATCHER_MATCHER_H_
