#include <Open3D/Open3D.h>
#include <glog/logging.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <voxblox_msgs/Layer.h>
#include <voxblox_msgs/LayerWithTrajectory.h>
#include <voxblox_ros/conversions.h>

#include <future>
#include <utility>
#include <vector>

#include "freeture_matcher/freeture_extractor.h"
#include "freeture_matcher/matcher.h"

namespace voxblox {
class FreetureNode {
 public:
  FreetureNode(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
      : extractor_(ros::NodeHandle(nh_private, "extractor")),
        matcher_(ros::NodeHandle(nh_private, "matcher")),
        nh_(nh),
        nh_private_(nh_private),
        o3d_visualize_(false),
        layer_with_traj_(true) {
    nh_private.param("layer_with_traj", layer_with_traj_, layer_with_traj_);
    nh_private.param("o3d_visualize", o3d_visualize_, o3d_visualize_);
    if (layer_with_traj_)
      tsdf_map_sub_ = nh_private_.subscribe(
          "tsdf_map_in", 10, &FreetureNode::layerWithTrajCallback, this);
    else
      tsdf_map_sub_ = nh_private_.subscribe("tsdf_map_in", 10,
                                            &FreetureNode::layerCallback, this);

    keypoints_pub_ = nh_private_.advertise<visualization_msgs::MarkerArray>(
        "keypoints", 10, true);
  }
  virtual ~FreetureNode() = default;

  void layerCallback(const voxblox_msgs::Layer& layer_msg) {
    Layer<TsdfVoxel> tsdf_layer(layer_msg.voxel_size,
                                layer_msg.voxels_per_side);
    if (!deserializeMsgToLayer(layer_msg, &tsdf_layer)) {
      ROS_WARN("Received a submap msg with an invalid TSDF");
    } else {
      ROS_INFO("Received a valid tsdf map");
    }

    if (match_async_handle_.valid() &&
        match_async_handle_.wait_for(std::chrono::milliseconds(10)) !=
            std::future_status::ready) {
      ROS_WARN("Previous matching not yet finished, waiting");
      match_async_handle_.wait();
    }

    match_async_handle_ = std::async(
        std::launch::async, &FreetureNode::matchWithDatabase, this, tsdf_layer);
  }

  void layerWithTrajCallback(
      const voxblox_msgs::LayerWithTrajectory& layer_msg) {
    layerCallback(layer_msg.layer);
  }

  // TODO(mikexyl): there seems to be a compare funtion in open3d
  bool resultBetter(const RegistrationResult& a, const RegistrationResult& b) {
    if (a.fitness_ > b.fitness_)
      return true;
    else if (a.fitness_ == b.fitness_ && a.inlier_rmse_ < b.inlier_rmse_)
      return true;
    return false;
  }

  void matchWithDatabase(const Layer<TsdfVoxel>& tsdf_layer) {
    extractor_.extractFreetures(tsdf_layer);
    auto const& keypoints_q = extractor_.getKeypoints();
    auto const& features_q = extractor_.getFeatures();
    O3dFeature o3d_feature_q, o3d_feature_t;
    if (!matcher_.featureMatrixToO3dFeature(features_q, &o3d_feature_q)) return;
    std::vector<RegistrationResult> results;
    RegistrationResult best_result;
    int best_match_id = -1;
    for (int i = 0; i < submap_db_.size(); i++) {
      auto const& kp_desc = submap_db_[i];

      RegistrationResult result;
      matcher_.matchPointClouds(keypoints_q, o3d_feature_q, kp_desc.first,
                                o3d_feature_t, &result);
      if (resultBetter(result, best_result)) {
        best_result = result;
        best_match_id = i;
      }
    }
    LOG(INFO) << "best_match_id: " << best_match_id;
    if (best_match_id >= 0 && o3d_visualize_) {
      visualize_registration(keypoints_q, submap_db_[best_match_id].first,
                             best_result.transformation_);
    }

    submap_db_.emplace_back(keypoints_q, o3d_feature_q);
  }

  void publishKeypoints(const PointcloudV& keypoints) {
    visualization_msgs::MarkerArray marker_array_msg;
    for (auto const& kp : keypoints) {
      visualization_msgs::Marker marker;
    }
  }

  void visualize_registration(const O3dPointCloud& source,
                              const O3dPointCloud& target,
                              const Eigen::Matrix4d& transformation) {
    std::shared_ptr<open3d::geometry::PointCloud> source_transformed_ptr(
        new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> target_ptr(
        new open3d::geometry::PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(transformation);
    open3d::visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                          "Registration result");
  }

  bool layer_with_traj_;
  bool o3d_visualize_;

  FreetureExtractor extractor_;
  Matcher matcher_;

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber tsdf_map_sub_;
  ros::Publisher keypoints_pub_;

  std::vector<std::pair<PointcloudV, O3dFeature>> submap_db_;

  std::future<void> match_async_handle_;

  std::string world_frame_;
};
}  // namespace voxblox

int main(int argc, char** argv) {
  // Start logging
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  // Register with ROS master
  ros::init(argc, argv, "freeture_matcher");

  // Create node handles
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  voxblox::FreetureNode node(nh, nh_private);

  // Spin
  ros::spin();

  // Exit normally
  return 0;
}
