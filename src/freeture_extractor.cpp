#include "freeture_matcher/freeture_extractor.h"

namespace voxblox {

void FreetureExtractor::extractFreetures(const Layer<TsdfVoxel>& tsdf_layer) {
  keypoints_.clear();
  features_.clear();
  Layer<DistVoxel> dist_gauss_layer(tsdf_layer.voxel_size(),
                                    tsdf_layer.voxels_per_side());
  computeGaussianDistance(tsdf_layer, &dist_gauss_layer);

  Layer<GradVoxel> grad_layer(tsdf_layer.voxel_size(),
                              tsdf_layer.voxels_per_side());
  computeGradient(dist_gauss_layer, &grad_layer);
  computeHessian(tsdf_layer, grad_layer);
}

void FreetureExtractor::computeGaussianDistance(
    const Layer<TsdfVoxel>& tsdf_layer, Layer<DistVoxel>* dist_gauss_layer) {
  Layer<DistVoxel> gauss_x_layer(tsdf_layer.voxel_size(),
                                 tsdf_layer.voxels_per_side());
  convolveLayerTsdf(tsdf_layer, &gauss_x_layer, kGaussKernel, kOffsetX);
  Layer<DistVoxel> gauss_y_layer(tsdf_layer.voxel_size(),
                                 tsdf_layer.voxels_per_side());
  convolveLayer(gauss_x_layer, &gauss_y_layer, kGaussKernel, kOffsetY);
  convolveLayer(gauss_y_layer, dist_gauss_layer, kGaussKernel, kOffsetZ);
}

void FreetureExtractor::convolveLayer(const Layer<DistVoxel>& in_layer,
                                      Layer<DistVoxel>* out_layer,
                                      Kernel3 kernel,
                                      GlobalIndexVector offsets) {
  BlockIndexList tsdf_blocks;
  in_layer.getAllAllocatedBlocks(&tsdf_blocks);
  GlobalIndexVector global_indices;
  for (auto const& block_index : tsdf_blocks) {
    auto tsdf_block = in_layer.getBlockPtrByIndex(block_index);
    if (!tsdf_block) continue;

    typename Block<DistVoxel>::Ptr gauss_block =
        out_layer->allocateBlockPtrByIndex(block_index);
    gauss_block->set_updated(true);
    const size_t num_voxels_per_block = tsdf_block->num_voxels();
    for (size_t lin_index = 0u; lin_index < num_voxels_per_block; lin_index++) {
      VoxelIndex voxel_index =
          tsdf_block->computeVoxelIndexFromLinearIndex(lin_index);
      GlobalIndex global_index = getGlobalVoxelIndexFromBlockAndVoxelIndex(
          block_index, voxel_index, in_layer.voxels_per_side());

      global_indices.emplace_back(global_index);
    }
  }

  convolveVoxel<DistVoxel, DistVoxel, Kernel3>(
      in_layer, out_layer, global_indices, kernel, offsets,
      std::bind(&FreetureExtractor::convValueAdd, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3));
}

void FreetureExtractor::convolveLayerTsdf(const Layer<TsdfVoxel>& in_layer,
                                          Layer<DistVoxel>* out_layer,
                                          Kernel3 kernel,
                                          GlobalIndexVector offsets) {
  BlockIndexList tsdf_blocks;
  in_layer.getAllAllocatedBlocks(&tsdf_blocks);
  GlobalIndexVector global_indices;
  for (auto const& block_index : tsdf_blocks) {
    auto tsdf_block = in_layer.getBlockPtrByIndex(block_index);
    if (!tsdf_block) continue;

    Block<DistVoxel>::Ptr gauss_block =
        out_layer->allocateBlockPtrByIndex(block_index);
    gauss_block->set_updated(true);
    const size_t num_voxels_per_block = tsdf_block->num_voxels();
    for (size_t lin_index = 0u; lin_index < num_voxels_per_block; lin_index++) {
      DistVoxel& dist_gauss_voxel =
          gauss_block->getVoxelByLinearIndex(lin_index);

      auto const& tsdf_voxel = tsdf_block->getVoxelByLinearIndex(lin_index);
      if (tsdf_voxel.weight < config_.min_weight) {
        dist_gauss_voxel.valid = false;
        continue;
      }

      VoxelIndex voxel_index =
          tsdf_block->computeVoxelIndexFromLinearIndex(lin_index);
      GlobalIndex global_index = getGlobalVoxelIndexFromBlockAndVoxelIndex(
          block_index, voxel_index, in_layer.voxels_per_side());
      global_indices.emplace_back(global_index);
    }
  }
  convolveVoxel<TsdfVoxel, DistVoxel, Kernel3>(
      in_layer, out_layer, global_indices, kernel, offsets,
      std::bind(&FreetureExtractor::convValueAddTsdf, this,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3));
}

template <typename InVoxelT, typename OutVoxelT, typename KernelT,
          typename GlobalIndicesT>
void FreetureExtractor::convolveVoxel(
    const Layer<InVoxelT>& in_layer, Layer<OutVoxelT>* out_layer,
    const GlobalIndicesT& global_indices, KernelT kernel,
    GlobalIndexVector offsets, ConvFunc<InVoxelT, OutVoxelT> conv_func,
    PostProcFunc<InVoxelT, OutVoxelT> post_func, bool ignore_invalid) {
  for (auto const& global_index : global_indices) {
    auto out_voxel = out_layer->getVoxelPtrByGlobalIndex(global_index);
    if (!ignore_invalid && !out_voxel->valid) continue;

    setToZero(&out_voxel->value);
    CHECK_EQ(kernel.cols(), offsets.size());
    for (unsigned int idx = 0; idx < offsets.size(); ++idx) {
      const GlobalIndex& neighbor_index = global_index + offsets[idx];

      const InVoxelT* neighbor_voxel =
          in_layer.getVoxelPtrByGlobalIndex(neighbor_index);
      // TODO(mikexyl): two kind of invalid, either neighbor_voxel doesn't
      // exist, or valid is false
      if (!neighbor_voxel) {
        if (!ignore_invalid) {
          out_voxel->valid = false;
          break;
        }
      } else {
        CHECK_EQ(kernel[idx], kernel[idx]) << kernel.cols() << " " << idx;
        conv_func(*neighbor_voxel, out_voxel, kernel(idx));
      }
    }

    if (post_func && out_voxel->valid) post_func(out_voxel, global_index);
  }
}

const FreetureExtractor::Kernel3 FreetureExtractor::kGaussKernel = [] {
  Kernel3 gauss_matrix;
  gauss_matrix << 0.8825, 1.0000, 0.8825;
  gauss_matrix /= gauss_matrix.sum();
  return gauss_matrix;
}();

const FreetureExtractor::Kernel3 FreetureExtractor::kSobelKernel = [] {
  Kernel3 sobel_kernel;
  sobel_kernel << 1.0, 0, -1.0;
  return sobel_kernel;
}();

const GlobalIndexVector FreetureExtractor::kOffsetX = [] {
  GlobalIndexVector direction;
  direction.emplace_back(1, 0, 0);
  direction.emplace_back(0, 0, 0);
  direction.emplace_back(-1, 0, 0);
  return direction;
}();

const GlobalIndexVector FreetureExtractor::kOffsetY = [] {
  GlobalIndexVector direction;
  direction.emplace_back(0, 1, 0);
  direction.emplace_back(0, 0, 0);
  direction.emplace_back(0, -1, 0);
  return direction;
}();

const GlobalIndexVector FreetureExtractor::kOffsetZ = [] {
  GlobalIndexVector direction;
  direction.emplace_back(0, 0, 1);
  direction.emplace_back(0, 0, 0);
  direction.emplace_back(0, 0, -1);
  return direction;
}();

const Eigen::Matrix<float, 1, 1> FreetureExtractor::kIdenKernel = [] {
  Eigen::Matrix<float, 1, 1> kernel;
  kernel << 1.0;
  return kernel;
}();

}  // namespace voxblox
