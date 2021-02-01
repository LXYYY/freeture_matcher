#ifndef INCLUDE_FREETURE_MATCHER_FREETURE_EXTRACTOR_H_
#define INCLUDE_FREETURE_MATCHER_FREETURE_EXTRACTOR_H_

#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Registration/Feature.h>
#include <ros/ros.h>
#include <voxblox/integrator/esdf_integrator.h>
#include <voxblox_ros/ros_params.h>

#include <math.h>
#include <functional>
#include <limits>
#include <vector>

#include "freeture_matcher/common.h"

namespace voxblox {
class FreetureExtractor {
 public:
  struct Config : EsdfIntegrator::Config {
    bool verbose = false;
    int radius_feature = 15;
    int n_div = 10;
    int radius_local_extremum = 15;
  };

  friend inline std::ostream& operator<<(std::ostream& s, const Config& v) {
    s << std::endl
      << "FreetureExtractor using Config:" << std::endl
      << "  verbose: " << v.verbose << std::endl
      << "  radius_feature: " << v.radius_feature << std::endl
      << "  n_div: " << v.n_div << std::endl
      << "  radius_local_extremum: " << v.radius_local_extremum << std::endl
      << "-------------------------------------------" << std::endl;
    return (s);
  }

  Config getConfigFromRosParam(const ros::NodeHandle& nh_private) {
    Config config;
    nh_private.param("verbose", config.verbose, config.verbose);
    nh_private.param("radius_feature", config.radius_feature,
                     config.radius_feature);
    nh_private.param("n_div", config.n_div, config.n_div);
    nh_private.param("radius_local_extremum", config.radius_local_extremum,
                     config.radius_local_extremum);
    return config;
  }

  typedef Eigen::Matrix<float, 1, 3> Kernel3;
  typedef Eigen::Matrix3f HessianMatrix;
  typedef Eigen::Matrix<float, 1, Eigen::Dynamic> KernelX;

  explicit FreetureExtractor(const ros::NodeHandle& nh_private)
      : FreetureExtractor(getConfigFromRosParam(nh_private)) {}

  explicit FreetureExtractor(const Config& config) : config_(config) {
    LOG(INFO) << config_;
    float sigma2 =
        static_cast<float>(config_.radius_feature * config_.radius_feature);

    CHECK_EQ(config_.radius_feature % 2, 1);
    int r_f = config_.radius_feature;
    std::vector<int> gauss;
    for (int i = -r_f; i <= r_f; i++)
      for (int j = -(r_f - abs(i)); j <= (r_f - abs(i)); j++)
        for (int k = -(r_f - abs(i) - abs(j)); k < (r_f - abs(i) - abs(j));
             k++) {
          kFeatureOffset.emplace_back(i, j, k);
          int distance = abs(i) + abs(j) + abs(k);
          gauss.emplace_back(gaussian(distance, sigma2));
        }
    kFeatureGaussianKernel.resize(gauss.size());
    for (size_t i = 0; i < gauss.size(); i++)
      kFeatureGaussianKernel(i) = gauss[i];
    kFeatureGaussianKernel /= kFeatureGaussianKernel.sum();

    kAngleDivision = M_PI / config_.n_div;

    // TODO(mikexyl): verify
    kSolidAngle.resize(2 * config_.n_div * config_.n_div + 2);
    kSolidAngle.setZero();
    for (int i = 0; i < kSolidAngle.size() - 2; i++) {
      int azi_id;
      int pol_id = std::remquo(i, config_.n_div, &azi_id);
      float azi = azi_id * kAngleDivision - M_PI;
      float pol = pol_id * kAngleDivision - M_PI / 2;
      kSolidAngle[i] = kAngleDivision * (cos(azi) - cos(azi + kAngleDivision));
    }
    CHECK_EQ(kSolidAngle[kSolidAngle.size() - 1], 0.0);
    CHECK_EQ(kSolidAngle[kSolidAngle.size() - 2], 0.0);
    kSolidAngle[kSolidAngle.size() - 1] = 1.0;
    kSolidAngle[kSolidAngle.size() - 2] = 1.0;

    int r_local = config_.radius_feature;
    for (int i = -r_local; i <= r_local; i++)
      for (int j = -(r_local - abs(i)); j <= (r_local - abs(i)); j++)
        for (int k = -(r_local - abs(i) - abs(j));
             k < (r_local - abs(i) - abs(j)); k++) {
          kLocalMaxOffset.emplace_back(i, j, k);
        }
  }

  static float gaussian(float distance, float sigma2) {
    return exp(-((distance * distance) / (2 * sigma2))) / (2 * M_PI * sigma2);
  }

  virtual ~FreetureExtractor() = default;

  void extractFreetures(const Layer<TsdfVoxel>& tsdf_layer);

  void computeGaussianDistance(const Layer<TsdfVoxel>& tsdf_layer,
                               Layer<DistVoxel>* dist_gauss_layer);

  void computeHessian(const Layer<TsdfVoxel>& tsdf_layer,
                      const Layer<GradVoxel>& grad_layer) {
    Layer<DetVoxel> det_layer(tsdf_layer.voxel_size(),
                              tsdf_layer.voxels_per_side());
    BlockIndexList grad_blocks;
    grad_layer.getAllAllocatedBlocks(&grad_blocks);
    for (auto const& block_index : grad_blocks) {
      auto grad_block = grad_layer.getBlockPtrByIndex(block_index);
      if (!grad_block) continue;

      Block<DetVoxel>::Ptr det_block =
          det_layer.allocateBlockPtrByIndex(block_index);
      det_block->set_updated(true);
      const size_t num_voxels_per_block = grad_block->num_voxels();
      for (size_t lin_index = 0u; lin_index < num_voxels_per_block;
           lin_index++) {
        DetVoxel& det_voxel = det_block->getVoxelByLinearIndex(lin_index);
        auto const& grad_voxel = grad_block->getVoxelByLinearIndex(lin_index);
        if (!grad_voxel.valid) {
          det_voxel.valid = false;
          continue;
        }

        VoxelIndex voxel_index =
            grad_block->computeVoxelIndexFromLinearIndex(lin_index);
        GlobalIndex global_index = getGlobalVoxelIndexFromBlockAndVoxelIndex(
            block_index, voxel_index, tsdf_layer.voxels_per_side());

        Neighborhood<Connectivity::kSix>::IndexMatrix index_matrix;
        Neighborhood<Connectivity::kSix>::getFromGlobalIndex(global_index,
                                                             &index_matrix);

        HessianMatrix h;
        for (unsigned int idx = 0; idx < 6; ++idx) {
          const GlobalIndex& neighbor_index =
              global_index + index_matrix.col(idx);

          const GradVoxel* neighbor_voxel =
              grad_layer.getVoxelPtrByGlobalIndex(neighbor_index);
          if (!neighbor_voxel) {
            det_voxel.valid = false;
            break;
          } else {
            int dim = idx / 2, dir = idx % 2;
            for (int i = 0; i < 3; i++) {
              h(dim, i) = neighbor_voxel->value(i) * kSobelKernel(dir * 2);
            }
          }
        }
        // compute determinant
        if (det_voxel.valid) det_voxel.value = h.determinant();
      }
    }

    detectKeypointsAndComputeDescriptors(&det_layer, grad_layer, tsdf_layer);
  }

  void detectKeypointsAndComputeDescriptors(
      Layer<DetVoxel>* det_layer, const Layer<GradVoxel>& grad_layer,
      const Layer<TsdfVoxel>& tsdf_layer) {
    Layer<SkVoxel> sk_layer(det_layer->voxel_size(),
                            det_layer->voxels_per_side());
    LongIndexSet keypoint_ids;
    BlockIndexList grad_blocks;
    det_layer->getAllAllocatedBlocks(&grad_blocks);
    for (auto const& block_index : grad_blocks) {
      auto det_block = det_layer->getBlockPtrByIndex(block_index);
      if (!det_block) continue;

      sk_layer.allocateBlockPtrByIndex(block_index);
      const size_t num_voxels_per_block = det_block->num_voxels();
      for (size_t lin_index = 0u; lin_index < num_voxels_per_block;
           lin_index++) {
        auto const& det_voxel = det_block->getVoxelByLinearIndex(lin_index);
        if (!det_voxel.valid) {
          continue;
        }

        VoxelIndex voxel_index =
            det_block->computeVoxelIndexFromLinearIndex(lin_index);
        GlobalIndex global_index = getGlobalVoxelIndexFromBlockAndVoxelIndex(
            block_index, voxel_index, det_layer->voxels_per_side());

        HessianMatrix h;
        DetVoxel::ValueT maximum = std::numeric_limits<DetVoxel::ValueT>::min(),
                         minimum = std::numeric_limits<DetVoxel::ValueT>::max();
        GlobalIndex max_index, min_index;
        GlobalIndexVector max_indices, min_indices;
        std::vector<DetVoxel::ValueT> max_values, min_values;
        bool found_any = false;
        for (unsigned int idx = 0; idx < kLocalMaxOffset.size(); ++idx) {
          GlobalIndex neighbor_index = global_index + kLocalMaxOffset[idx];

          DetVoxel* neighbor_voxel =
              det_layer->getVoxelPtrByGlobalIndex(neighbor_index);
          if (!neighbor_voxel) continue;

          if (neighbor_voxel->maximum) {
            max_indices.emplace_back(neighbor_index);
            max_values.emplace_back(neighbor_voxel->value);
          }
          if (neighbor_voxel->minimum) {
            min_indices.emplace_back(neighbor_index);
            min_values.emplace_back(neighbor_voxel->value);
          }

          if (neighbor_voxel->valid) {
            neighbor_voxel->valid = false;
            CHECK_GT(neighbor_voxel->value,
                     std::numeric_limits<DetVoxel::ValueT>::min());
            if (neighbor_voxel->value > maximum) {
              maximum = neighbor_voxel->value;
              max_index = neighbor_index;
              found_any = true;
            }
            if (neighbor_voxel->value < minimum) {
              minimum = neighbor_voxel->value;
              min_index = neighbor_index;
              found_any = true;
            }
          }
        }

        for (int max_i = 0; max_i < max_indices.size(); max_i++)
          if (max_values[max_i] < maximum)
            keypoint_ids.erase(max_indices[max_i]);

        for (int min_i = 0; min_i < min_indices.size(); min_i++)
          if (min_values[min_i] > minimum)
            keypoint_ids.erase(min_indices[min_i]);

        if (found_any) {
          DetVoxel* max_voxel = det_layer->getVoxelPtrByGlobalIndex(max_index);
          DetVoxel* min_voxel = det_layer->getVoxelPtrByGlobalIndex(min_index);
          max_voxel->maximum = true;
          min_voxel->minimum = true;
          keypoint_ids.emplace(max_index);
          keypoint_ids.emplace(min_index);
        }
      }
    }

    LOG_IF(INFO, config_.verbose)
        << "detected keypoints " << keypoint_ids.size();

    convolveVoxel<GradVoxel, SkVoxel, KernelX>(
        grad_layer, &sk_layer, keypoint_ids, kFeatureGaussianKernel,
        kFeatureOffset,
        std::bind(&FreetureExtractor::convSk, this, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3)),
        std::bind(&FreetureExtractor::postEigen, this, std::placeholders::_1,
                  std::placeholders::_2);

    LOG_IF(INFO, config_.verbose) << "finished compute first 2*n^2 descriptor";

    convolveVoxel<TsdfVoxel, SkVoxel, KernelX>(
        tsdf_layer, &sk_layer, keypoint_ids, kFeatureGaussianKernel,
        kFeatureOffset,
        std::bind(&FreetureExtractor::convAugDescriptor, this,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3)),
        std::bind(&FreetureExtractor::postAugDescriptor, this,
                  std::placeholders::_1, std::placeholders::_2);

    LOG_IF(INFO, config_.verbose)
        << "finished compute dist and class descriptor";
  }

  template <typename VoxelT>
  GlobalIndexVector getNeighborVoxels(const GlobalIndex& voxel_index,
                                      GlobalIndexVector offsets) {
    GlobalIndexVector neighbor_indices;
    for (unsigned int idx = 0; idx < offsets.size(); idx++)
      neighbor_indices.emplace_back(voxel_index + offsets[idx]);
  }

  template <typename VoxelT>
  bool getNextBlock(const Layer<VoxelT>& layer,
                    const BlockIndexList& block_indices, int* i,
                    typename Block<VoxelT>::Ptr* block) {
    *block = layer.getBlockPtrByIndex(block_indices[(*i)++]);
    return static_cast<bool>(block);
  }

  void computeGradient(const Layer<DistVoxel>& in_layer,
                       Layer<GradVoxel>* grad_layer) {
    BlockIndexList in_blocks;
    in_layer.getAllAllocatedBlocks(&in_blocks);
    for (auto const& block_index : in_blocks) {
      auto in_block = in_layer.getBlockPtrByIndex(block_index);
      if (!in_block) continue;

      Block<GradVoxel>::Ptr grad_block =
          grad_layer->allocateBlockPtrByIndex(block_index);
      grad_block->set_updated(true);
      const size_t num_voxels_per_block = in_block->num_voxels();
      for (size_t lin_index = 0u; lin_index < num_voxels_per_block;
           lin_index++) {
        GradVoxel& grad_voxel = grad_block->getVoxelByLinearIndex(lin_index);

        auto const& tsdf_voxel = in_block->getVoxelByLinearIndex(lin_index);
        if (!tsdf_voxel.valid) {
          grad_voxel.valid = false;
          continue;
        }

        VoxelIndex voxel_index =
            in_block->computeVoxelIndexFromLinearIndex(lin_index);
        GlobalIndex global_index = getGlobalVoxelIndexFromBlockAndVoxelIndex(
            block_index, voxel_index, in_layer.voxels_per_side());

        Neighborhood<Connectivity::kSix>::IndexMatrix index_matrix;
        Neighborhood<Connectivity::kSix>::getFromGlobalIndex(global_index,
                                                             &index_matrix);
        for (unsigned int idx = 0; idx < 6; ++idx) {
          const GlobalIndex& neighbor_index =
              global_index + index_matrix.col(idx);

          const DistVoxel* neighbor_voxel =
              in_layer.getVoxelPtrByGlobalIndex(neighbor_index);
          if (!neighbor_voxel) {
            grad_voxel.valid = false;
            break;
          } else {
            int dim = idx / 2, dir = idx % 2;
            grad_voxel.value(dim) =
                neighbor_voxel->value * kSobelKernel(dir * 2);
          }
        }
      }
    }
  }

  void convolveLayer(const Layer<DistVoxel>& in_layer,
                     Layer<DistVoxel>* out_layer, Kernel3 kernel,
                     GlobalIndexVector offsets);

  void convolveLayerTsdf(const Layer<TsdfVoxel>& in_layer,
                         Layer<DistVoxel>* out_layer, Kernel3 kernel,
                         GlobalIndexVector offsets);

  template <class InVoxelT, class OutVoxelT>
  using ConvFunc = std::function<void(const InVoxelT&, OutVoxelT*, float)>;

  template <class InVoxelT, class OutVoxelT>
  using PostProcFunc = std::function<void(OutVoxelT*, GlobalIndex index)>;

  template <typename InVoxelT, typename OutVoxelT, typename KernelT,
            typename GlobalIndicesT>
  void convolveVoxel(const Layer<InVoxelT>& in_layer,
                     Layer<OutVoxelT>* out_layer,
                     const GlobalIndicesT& global_indices, KernelT kernel,
                     GlobalIndexVector offsets,
                     ConvFunc<InVoxelT, OutVoxelT> conv_func,
                     PostProcFunc<InVoxelT, OutVoxelT> post_func =
                         PostProcFunc<InVoxelT, OutVoxelT>(),
                     bool ignore_invalid = false);

  void convValueAdd(const DistVoxel& in, DistVoxel* out, float kernel_value) {
    out->value += in.value * kernel_value;
  }

  void convValueAddTsdf(const TsdfVoxel& in, DistVoxel* out,
                        float kernel_value) {
    out->value += in.distance * kernel_value;
  }

  void convAugDescriptor(const TsdfVoxel& in, SkVoxel* out,
                         float kernel_value) {
    out->n_b_dist += kernel_value;
    out->b_dist += in.distance * kernel_value;
  }

  void postAugDescriptor(SkVoxel* in, GlobalIndex) {
    features_[in->descriptor_id][features_[in->descriptor_id].size() - 2] =
        in->b_dist / in->n_b_dist;
  }

  void convSk(const GradVoxel& in, SkVoxel* out, float kernel_value) {
    SkVoxel::ValueT neighbor_value;
    auto const& weighted_grad = in.value * kernel_value;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        neighbor_value(i, j) = weighted_grad[i] * weighted_grad[j];
      }
    out->value += neighbor_value;
    out->gk.emplace_back(in.value * kernel_value);
  }

  void postEigen(SkVoxel* in, GlobalIndex global_index) {
    // self adjoint eigen solver gives eigen value and vector in
    // ASCENDING order of eigen values
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(
        in->value.block(0, 0, 3, 3));
    auto eigen_values = es.eigenvalues();
    auto v = es.eigenvectors();

    std::vector<Eigen::Vector3f> a[2];
    float gkvi = 0, gkvi_abs = 0;
    for (int i = 0; i < 3; i += 2) {
      for (auto const& gk : in->gk) {
        gkvi += (gk.dot(v.col(i)));
        gkvi_abs += fabs(gk.dot(v.col(i)));
      }

      float s = gkvi / gkvi_abs;
      if (s > kKAxis) {
        a[i / 2].emplace_back(v.col(i));
      } else if (s <= kKAxis && s >= -kKAxis) {
        a[i / 2].emplace_back(v.col(i));
        a[i / 2].emplace_back(-v.col(i));
      } else {
        a[i / 2].emplace_back(-v.col(i));
      }
    }

    std::vector<Eigen::Matrix3f> lrfs;

    for (auto const& v1 : a[0])
      for (auto const& v2 : a[1]) {
        auto const& v3 = v1.cross(v2);
        Eigen::Matrix3f lrf;
        lrf.col(0) = v1;
        lrf.col(1) = v2;
        lrf.col(2) = v3;
        lrfs.emplace_back(lrf);
      }

    Feature descriptor(2 * config_.n_div * config_.n_div + 2);
    for (auto const& lrf : lrfs) {
      // TODO(mikexyl): verify
      auto const& R_f_s = lrf.inverse();
      for (auto const& gsk : in->gk) {
        auto const& gfk = R_f_s * gsk;
        auto const& mag = gfk.norm();
        auto const& azi = std::atan2(gfk[1], gfk[0]) + M_PI;
        auto const& pol = std::atan2(gfk[2], gfk.leftCols(2).norm()) + M_PI / 2;

        int azi_id;
        float azi_rem = std::remquo(azi, kAngleDivision, &azi_id);
        int pol_id;
        float pol_rem = std::remquo(pol, kAngleDivision, &pol_id);

        descSoftBinning(azi_id, azi_rem, pol_id, pol_rem, mag, &descriptor);
      }
    }

    // normalize descriptor
    descriptor.cwiseProduct((in->gk.size() * kSolidAngle).cwiseInverse());

    descriptor[descriptor.size() - 1] =
        (eigen_values.array() > 0).sum() * kAlphaClass;

    keypoints_.emplace_back(global_index.cast<double>());
    in->descriptor_id = features_.size();
    features_.emplace_back(descriptor);
  }

  void descSoftBinning(int azi_id, float azi_rem, int pol_id, float pol_rem,
                       float mag, Feature* descriptor) {
    auto const& n_div = config_.n_div;
    (*descriptor)[azi_id * n_div + pol_id] +=
        mag * (kAngleDivision - azi_rem) / kAngleDivision *
        (kAngleDivision - pol_rem) / kAngleDivision;
    (*descriptor)[(azi_id + 1) * n_div + pol_id] +=
        mag * azi_rem / kAngleDivision * (kAngleDivision - pol_rem) /
        kAngleDivision;
    (*descriptor)[azi_id * n_div + pol_id + 1] +=
        mag * (kAngleDivision - azi_rem) / kAngleDivision * pol_rem /
        kAngleDivision;
    (*descriptor)[(azi_id + 1) * n_div + pol_id + 1] +=
        mag * azi_rem / kAngleDivision * pol_rem / kAngleDivision;
  }

  auto const& getKeypoints() const { return keypoints_; }
  auto const& getFeatures() const { return features_; }

  PointcloudV keypoints_;
  FeatureMatrix features_;

  Config config_;

  KernelX kFeatureGaussianKernel;
  GlobalIndexVector kFeatureOffset;
  float kAngleDivision;
  Eigen::VectorXd kSolidAngle;

  GlobalIndexVector kLocalMaxOffset;

  static const Kernel3 kGaussKernel;
  static const Kernel3 kSobelKernel;
  static const Eigen::Matrix<float, 1, 1> kIdenKernel;

  static const GlobalIndexVector kOffsetX;
  static const GlobalIndexVector kOffsetY;
  static const GlobalIndexVector kOffsetZ;

  static constexpr float kKAxis = 0.5;
  static constexpr float kAlphaDist = 1e-7;
  static constexpr float kAlphaClass = 1e-5;
};
}  // namespace voxblox

#endif  // INCLUDE_FREETURE_MATCHER_FREETURE_EXTRACTOR_H_
