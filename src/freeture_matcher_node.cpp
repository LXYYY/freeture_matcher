#include <Open3D/Open3D.h>
#include <glog/logging.h>
#include <minkindr_conversions/kindr_msg.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <voxblox/integrator/merge_integration.h>
#include <voxblox_msgs/Layer.h>
#include <voxblox_msgs/LayerWithTrajectory.h>
#include <voxblox_ros/conversions.h>

#include <future>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "freeture_matcher/freeture_extractor.h"
#include "freeture_matcher/matcher.h"
#include "freeture_matcher/visualization.h"

namespace voxblox {
class FreetureNode {
 public:
  FreetureNode(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
      : extractor_(ros::NodeHandle(nh_private, "extractor")),
        matcher_(ros::NodeHandle(nh_private, "matcher")),
        nh_(nh),
        nh_private_(nh_private),
        o3d_visualize_(false),
        layer_with_traj_(true),
        last_submap_id_(0) {
    nh_private.param("layer_with_traj", layer_with_traj_, layer_with_traj_);
    nh_private.param("o3d_visualize", o3d_visualize_, o3d_visualize_);
    nh_private.param("publish_keypoints", publish_keypoints_,
                     publish_keypoints_);
    if (layer_with_traj_)
      tsdf_map_sub_ = nh_private_.subscribe(
          "tsdf_map_in", 10, &FreetureNode::layerWithTrajCallback, this);
    else
      tsdf_map_sub_ = nh_private_.subscribe("tsdf_map_in", 10,
                                            &FreetureNode::layerCallback, this);

    if (publish_keypoints_)
      keypoints_pub_ = nh_private_.advertise<visualization_msgs::Marker>(
          "keypoints", 10, true);
  }
  virtual ~FreetureNode() = default;

  void layerCallback(const voxblox_msgs::Layer& layer_msg) {
    submapProcess(layer_msg);
  }

  void layerWithTrajCallback(
      const voxblox_msgs::LayerWithTrajectory& layer_msg) {
    if (layer_msg.trajectory.poses.empty()) {
      ROS_WARN_STREAM("Received a submap without trajectory");
    }

    kindr::minimal::QuatTransformationTemplate<double> T_G_S_D;
    tf::poseMsgToKindr(
        layer_msg.trajectory
            .poses[static_cast<size_t>(layer_msg.trajectory.poses.size() / 2)]
            .pose,
        &T_G_S_D);
    submapProcess(layer_msg.layer, T_G_S_D.cast<FloatingPoint>());
  }

  void submapProcess(const voxblox_msgs::Layer& layer_msg,
                     Transformation T_G_S = Transformation()) {
    Layer<TsdfVoxel> tsdf_layer(layer_msg.voxel_size,
                                layer_msg.voxels_per_side);
    if (!deserializeMsgToLayer(layer_msg, &tsdf_layer)) {
      ROS_WARN("Received a submap msg with an invalid TSDF");
    } else {
      ROS_INFO("Received a valid tsdf map");
    }

    Layer<TsdfVoxel> tsdf_layer_G(layer_msg.voxel_size,
                                  layer_msg.voxels_per_side);
    transformLayer(tsdf_layer, T_G_S.inverse(), &tsdf_layer_G);
    tsdf_layer.removeAllBlocks();

    if (match_async_handle_.valid() &&
        match_async_handle_.wait_for(std::chrono::milliseconds(10)) !=
            std::future_status::ready) {
      ROS_WARN("Previous matching not yet finished, waiting");
      match_async_handle_.wait();
    }

    match_async_handle_ =
        std::async(std::launch::async, &FreetureNode::matchWithDatabase, this,
                   tsdf_layer_G, T_G_S);
  }

  // TODO(mikexyl): there seems to be a compare funtion in open3d
  bool resultBetter(const RegistrationResult& a, const RegistrationResult& b) {
    if (a.fitness_ > b.fitness_)
      return true;
    else if (a.fitness_ == b.fitness_ && a.inlier_rmse_ < b.inlier_rmse_)
      return true;
    return false;
  }

  void matchWithDatabase(const Layer<TsdfVoxel>& tsdf_layer,
                         const Transformation& T_G_S) {
    extractor_.extractFreetures(tsdf_layer);
    auto const& keypoints_q = extractor_.getKeypoints();
    auto const& features_q = extractor_.getFeatures();
    O3dFeature o3d_feature_q;
    if (!matcher_.featureMatrixToO3dFeature(features_q, &o3d_feature_q)) return;
    std::vector<RegistrationResult> results;
    RegistrationResult best_result;
    int best_match_id = -1;
    for (int i = 0; i < last_submap_id_; i++) {
      auto const& submap = submap_db_[i];

      RegistrationResult result;
      matcher_.matchPointClouds(keypoints_q, o3d_feature_q, submap.keypoints,
                                submap.features, &result);
      if (resultBetter(result, best_result)) {
        best_result = result;
        best_match_id = i;
      }
    }
    LOG(INFO) << "best_match_id: " << best_match_id;

    // Visualization
    if (best_match_id >= 0 && o3d_visualize_) {
      visualize_registration(keypoints_q, submap_db_[best_match_id].keypoints,
                             best_result.transformation_);
    }
    if (publish_keypoints_) publishKeypoints(keypoints_q);
    open3d::geometry::TriangleMesh mesh;
    o3dMeshFromTsdfLayer(tsdf_layer, 1, &mesh);
    submap_db_.emplace(last_submap_id_++,
                       MinSubmap(keypoints_q, o3d_feature_q, T_G_S, mesh));
  }

  void publishKeypoints(const PointcloudV& keypoints) {
    visualization_msgs::Marker kp_marker;
    kp_marker.type = visualization_msgs::Marker::POINTS;
    kp_marker.scale.x = 0.1;
    kp_marker.scale.y = 0.1;
    for (auto const& kp : keypoints) {
      geometry_msgs::Point kp_msg;
      kp_msg.x = kp.x();
      kp_msg.y = kp.y();
      kp_msg.z = kp.z();
      kp_marker.points.emplace_back(kp_msg);
    }
    keypoints_pub_.publish(kp_marker);
  }

  bool layer_with_traj_;
  bool o3d_visualize_;
  bool publish_keypoints_;

  FreetureExtractor extractor_;
  Matcher matcher_;

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber tsdf_map_sub_;
  ros::Publisher keypoints_pub_;

  std::map<int, MinSubmap> submap_db_;

  std::future<void> match_async_handle_;

  std::string world_frame_;

  int last_submap_id_;
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
