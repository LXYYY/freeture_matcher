#ifndef INCLUDE_FREETURE_MATCHER_COMMON_H_
#define INCLUDE_FREETURE_MATCHER_COMMON_H_

#include <minkindr_conversions/kindr_msg.h>
#include <ros/ros.h>
#include <voxblox/core/common.h>
#include <voxblox/core/voxel.h>

#include <memory>
#include <set>
#include <vector>

namespace voxblox {

template <typename ValueType>
struct TensorVoxelType {
  typedef ValueType ValueT;
  bool valid = true;
  ValueT value;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
typedef TensorVoxelType<Eigen::Vector3f> GradVoxel;
typedef TensorVoxelType<float> DistVoxel;
struct DetVoxel : TensorVoxelType<float> {
  bool maximum = false, minimum = false;
};
struct SkVoxel : TensorVoxelType<Eigen::Matrix<float, 3, 4>> {
  std::vector<Eigen::Vector3f> gk;
  size_t descriptor_id;
  float b_dist = 0, b_class = 0;
  float n_b_dist = 0;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct TensorVoxel {
  explicit TensorVoxel(const TsdfVoxel& in_voxel) : in_voxel_(in_voxel) {}
  typedef float ValueT;
  TsdfVoxel in_voxel_;
  const ValueT& getValue() const { return in_voxel_.distance; }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename T>
inline void setToZero(T* in) {
  *in = static_cast<T>(0);
}

template <typename T, int Row, int Col>
inline void setToZero(Eigen::Matrix<T, Row, Col>* in) {
  in->setZero();
}

template <>
inline void setToZero<Eigen::Matrix3f>(Eigen::Matrix3f* in) {
  in->setZero();
}

typedef Eigen::Vector3d PointV;
typedef std::vector<PointV> PointcloudV;
using Feature = Eigen::VectorXd;
using FeatureMatrix = AlignedVector<Eigen::VectorXd>;
}  // namespace voxblox

#endif  // INCLUDE_FREETURE_MATCHER_COMMON_H_
