#ifndef INCLUDE_FREETURE_MATCHER_MATCHER_H_
#define INCLUDE_FREETURE_MATCHER_MATCHER_H_

#include "std_srvs/Empty.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <DBoW3/DBoW3.h>
#pragma GCC diagnostic pop
#include <Open3D/Registration/Registration.h>
#include <ros/ros.h>
#include <opencv2/core/eigen.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "freeture_matcher/common.h"
#include "freeture_matcher/visualization.h"

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
    bool o3d_visualize = false;
    std::string voc_file = "";
    bool train_voc = false;
    int bow_voc_k = 10;
    int bow_voc_l = 6;

    friend inline std::ostream& operator<<(std::ostream& s, const Config& v) {
      s << std::endl
        << "Matcher  using Config:" << std::endl
        << "  verbose: " << v.verbose << std::endl
        << "  max_corr_distance: " << v.max_corr_distance << std::endl
        << "  k_dist: " << v.k_dist << std::endl
        << "  n_max_ransac: " << v.k_dist << std::endl
        << "  max_ransac_valid: " << v.max_ransac_valid << std::endl
        << "  o3d_visualize: " << v.o3d_visualize << std::endl
        << "  voc_file: " << v.voc_file << std::endl
        << "  train_voc: " << v.train_voc << std::endl
        << "  bow_voc_k: " << v.bow_voc_k << std::endl
        << "  bow_voc_l: " << v.bow_voc_l << std::endl
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
    nh_private.param("o3d_visualize", config.o3d_visualize,
                     config.o3d_visualize);
    nh_private.param("voc_file", config.voc_file, config.voc_file);
    nh_private.param("train_voc", config.train_voc, config.train_voc);
    nh_private.param("bow_voc_k", config.bow_voc_k, config.bow_voc_k);
    nh_private.param("bow_voc_l", config.bow_voc_l, config.bow_voc_l);
    return config;
  }

  explicit Matcher(const ros::NodeHandle& nh_private)
      : config_(getConfigFromRosParam(nh_private)), nh_(nh_private) {
    LOG(INFO) << config_;
    if (config_.voc_file.size() > 0 && !config_.train_voc) {
      std::ifstream f_check(config_.voc_file.c_str());
      if (!f_check.good()) {
        LOG(WARNING) << "Feature database configured to enable dbow search, "
                        "but no valid voc file provided. Dbow disabled.";
      } else {
        voc_.reset(new DBoW3::Vocabulary(config_.voc_file));
        db_.reset(new DBoW3::Database());
        db_->setVocabulary(*voc_, false, 0);
      }
    }
    if (config_.train_voc)
      train_voc_srv_ =
          nh_.advertiseService("train_voc", &Matcher::trainVocCallback, this);
  }

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

  void addSubmap(int submap_id, const PointcloudV& keypoints,
                 const O3dFeature& features, const Transformation& T_G_S,
                 const open3d::geometry::TriangleMesh& mesh) {
    submap_db_.emplace(submap_id, MinSubmap(keypoints, features, T_G_S, mesh));
  }

  void matchWithDatabase(const PointcloudV& keypoints_q,
                         const O3dFeature features_q) {
    if (config_.train_voc) return;
    std::vector<RegistrationResult> results;
    RegistrationResult best_result;
    int best_match_id = -1;
    for (int i = 0; i < submap_db_.size(); i++) {
      auto const& submap = submap_db_[i];

      RegistrationResult result;
      matchPointClouds(keypoints_q, features_q, submap.keypoints,
                       submap.features, &result);
      if (resultBetter(result, best_result)) {
        best_result = result;
        best_match_id = i;
      }
    }
    LOG(INFO) << "best_match_id: " << best_match_id;

    // Visualization
    if (best_match_id >= 0 && config_.o3d_visualize) {
      visualize_registration(keypoints_q, submap_db_[best_match_id].keypoints,
                             best_result.transformation_);
    }
  }

  // TODO(mikexyl): there seems to be a compare funtion in open3d
  bool resultBetter(const RegistrationResult& a, const RegistrationResult& b) {
    if (a.fitness_ > b.fitness_)
      return true;
    else if (a.fitness_ == b.fitness_ && a.inlier_rmse_ < b.inlier_rmse_)
      return true;
    return false;
  }

 private:
  const Config config_;

  ros::NodeHandle nh_;

  std::shared_ptr<DBoW3::Database> db_;
  std::shared_ptr<DBoW3::Vocabulary> voc_;

  std::map<int, MinSubmap> submap_db_;

  ros::ServiceServer train_voc_srv_;

  bool trainVocCallback(std_srvs::Empty::Request& request,      // NOLINT
                        std_srvs::Empty::Response& response) {  // NOLINT
    const int k = config_.bow_voc_k;
    const int l = config_.bow_voc_l;
    constexpr DBoW3::WeightingType weight = DBoW3::TF_IDF;
    constexpr DBoW3::ScoringType score = DBoW3::L1_NORM;

    DBoW3::Vocabulary voc(k, l, weight, score);

    std::vector<cv::Mat> feature_cv_vec;
    uint64_t total_num = 0;
    for (auto const& submap_kv : submap_db_) {
      auto const& feature = submap_kv.second.features.data_;
      cv::Mat feature_cv(cv::Size(feature.cols(), feature.rows()), CV_64F);
      cv::eigen2cv(feature, feature_cv);
      feature_cv_vec.emplace_back(feature_cv);
    }
    for (const cv::Mat& feature_mat : feature_cv_vec) {
      total_num += feature_mat.rows;
    }
    LOG(INFO) << "Voc input feature mat size: " << total_num << "x"
              << feature_cv_vec.at(0).cols;
    LOG(INFO) << "Creating a " << k << "^" << l << " vocabulary...";

    auto t_start = std::chrono::high_resolution_clock::now();
    voc.create(feature_cv_vec);
    auto t_end = std::chrono::high_resolution_clock::now();

    voc.save(config_.voc_file);

    LOG(INFO) << "Created Vocabulary:";
    LOG(INFO) << "  Vocabulary information: " << std::endl
              << "  " << voc << std::endl;
    LOG(INFO) << "  saved to " << config_.voc_file;
    LOG(INFO) << "  time="
              << double(std::chrono::duration_cast<std::chrono::milliseconds>(
                            t_end - t_start)
                            .count())
              << " msecs";
    LOG(INFO) << "  nblocks=" << voc.size();
    LOG(INFO) << "-------------------------";

    return true;
  }
};
}  // namespace voxblox

#endif  // INCLUDE_FREETURE_MATCHER_MATCHER_H_
