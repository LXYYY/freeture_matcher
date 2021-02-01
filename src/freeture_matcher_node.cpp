#include <glog/logging.h>
#include <ros/ros.h>
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
        nh_private_(nh_private) {
    nh_private.param("layer_with_traj", layer_with_traj_, layer_with_traj_);
    if (layer_with_traj_)
      tsdf_map_sub_ = nh_private_.subscribe(
          "tsdf_map_in", 10, &FreetureNode::layerWithTrajCallback, this);
    else
      tsdf_map_sub_ = nh_private_.subscribe("tsdf_map_in", 10,
                                            &FreetureNode::layerCallback, this);
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

  void matchWithDatabase(const Layer<TsdfVoxel>& tsdf_layer) {
    extractor_.extractFreetures(tsdf_layer);
    auto const& keypoints_q = extractor_.getKeypoints();
    auto const& features_q = extractor_.getFeatures();
    for (auto const& kp_desc : submap_db_) {
      RegistrationResult result;
      matcher_.matchPointClouds(keypoints_q, features_q, kp_desc.first,
                                kp_desc.second, &result);
    }

    submap_db_.emplace_back(keypoints_q, features_q);
  }

  bool layer_with_traj_;

  FreetureExtractor extractor_;
  Matcher matcher_;

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber tsdf_map_sub_;

  std::vector<std::pair<PointcloudV, FeatureMatrix>> submap_db_;

  std::future<void> match_async_handle_;
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
