#ifndef INCLUDE_FREETURE_MATCHER_VISUALIZATION_H_
#define INCLUDE_FREETURE_MATCHER_VISUALIZATION_H_

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/DrawGeometry.h>
#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/interpolator/interpolator.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/mesh/mesh_layer.h>

#include <memory>
#include <vector>

#include "freeture_matcher/common.h"

namespace voxblox {

inline void visualize_registration(const O3dPointCloud& source,
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

inline bool o3dMeshFromTsdfLayer(const Layer<TsdfVoxel>& tsdf_layer,
                                 float min_weight,
                                 open3d::geometry::TriangleMesh* o3d_mesh) {
  CHECK(o3d_mesh != nullptr);
  o3d_mesh->Clear();
  MeshLayer mesh_layer(tsdf_layer.block_size());
  MeshIntegratorConfig mesh_integrator_config;
  mesh_integrator_config.use_color = false;
  mesh_integrator_config.min_weight = static_cast<float>(min_weight);

  voxblox::MeshIntegrator<voxblox::TsdfVoxel> mesh_integrator(
      mesh_integrator_config, tsdf_layer, &mesh_layer);
  mesh_integrator.generateMesh(false, false);

  // Convert it into a connected mesh
  voxblox::Point origin{0, 0, 0};
  voxblox::Mesh connected_mesh(tsdf_layer.block_size(), origin);
  mesh_layer.getConnectedMesh(&connected_mesh, 0.5 * tsdf_layer.voxel_size());

  CHECK_EQ(connected_mesh.indices.size() % 3, 0);

  std::vector<Eigen::Vector3d> vertices;
  std::vector<Eigen::Vector3i> indices;
  for (int i = 0; i < connected_mesh.indices.size(); i = i + 3) {
    indices.emplace_back(connected_mesh.indices[i],
                         connected_mesh.indices[i + 1],
                         connected_mesh.indices[i + 2]);
  }
  for (auto const& point : connected_mesh.vertices)
    vertices.emplace_back(point.cast<double>());
  *o3d_mesh = open3d::geometry::TriangleMesh(vertices, indices);

  return true;
}

}  // namespace voxblox

#endif  // INCLUDE_FREETURE_MATCHER_VISUALIZATION_H_
