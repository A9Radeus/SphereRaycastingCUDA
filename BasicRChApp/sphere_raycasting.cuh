#pragma once
#include <vector>
#include <random>

#include "HostUtils.h"
#include "DevUtils.cuh"
#include "maths3D.h"

namespace rch {

struct CuSphere {
  CuVec3 pos = {0, 0, 0};
  float radius = 0;

  CuSphere(const Vec3& pos_, float radius_) : pos(pos_), radius(radius_) {}
  CuSphere(Vec3&& pos_, float radius_)
      : pos(std::move(pos_)), radius(radius_) {}

  CuSphere(const CuSphere& src) = default;
  CuSphere(CuSphere&& src) = default;
  CuSphere& operator=(const CuSphere& src) = default;
  CuSphere& operator=(CuSphere&& src) = default;

  static std::vector<CuSphere> genRandSpheres(std::mt19937& rne,
                                              std::size_t sphereNum,
                                              float posMax = 100,
                                              float radiusMax = 20) {
    std::uniform_real_distribution<float> radiusDistri(0.001f, radiusMax);
    std::uniform_real_distribution<float> posDistri(-posMax, posMax);

    std::vector<CuSphere> spheres;
    spheres.reserve(sphereNum);
    for (auto i = 0; i < sphereNum; i++) {
      Vec3 pos = {posDistri(rne), posDistri(rne), posDistri(rne)};
      float r = radiusDistri(rne);
      spheres.push_back(CuSphere(pos, r));
    }

    return spheres;
  }
};

struct CuLightSrc {
  CuVec3 pos = {0, 0, 0};
  CuVec3 col = {1, 1, 1};

  CuLightSrc(const Vec3& pos_, const Vec3& col_) : pos(pos_), col(col_) {}
  CuLightSrc(Vec3&& pos_, Vec3&& col_)
      : pos(std::move(pos_)), col(std::move(col_)) {}

  //CuLightSrc(const CuLightSrc& src) = default;
  //CuLightSrc(CuLightSrc&& src) = default;
  //CuLightSrc& operator=(const CuLightSrc& src) = default;
  //CuLightSrc& operator=(CuLightSrc&& src) = default;

  static std::vector<CuLightSrc> genRandLightSrcs(std::mt19937& rne,
                                                  std::size_t lightsNum,
                                                  float posMax = 100) {
    std::uniform_real_distribution<float> colDistri(0.01f, 1.f);
    std::uniform_real_distribution<float> posDistri(-posMax, posMax);

    std::vector<CuLightSrc> lights;
    lights.reserve(lightsNum);
    for (auto i = 0; i < lightsNum; i++) {
      Vec3 pos = {posDistri(rne), posDistri(rne), posDistri(rne)};
      Vec3 col = {colDistri(rne), colDistri(rne), colDistri(rne)};
      lights.push_back(CuLightSrc(pos, col));
    }

    return lights;
  }

};

void sphere_raycast_to_tex2d(const CuSphere* spheresArr, std::size_t spheresNum,
                             const CuLightSrc* lightsArr, std::size_t lightsNum,
                             const cuda::InteropTexture2D& itex,
                             const XMMATRIX& xmInvVpMtx);

__device__ bool raySphereIntersection(const CuVec3& rayOrigin,
                                      const CuVec3& rayDirection,
                                      const CuVec3& spherePos, float radius,
                                      float* sol1, float* sol2);

__device__ bool raySphereIntersection(const CuVec3& rayOrigin,
                                      const CuVec3& rayDirection,
                                      const CuVec3& spherePos, float radius,
                                      float* sol);

__device__ CuVec3 calcBlinnPhongReflection(CuVec3 point, CuVec3 norm,
                                           CuVec3 toEye,
                                           const CuLightSrc* lights,
                                           std::size_t lightNum);

__device__ CuVec3 calcPhongReflection(CuVec3 point, CuVec3 norm, CuVec3 toEye,
                                      const CuLightSrc* lights,
                                      std::size_t lightNum);

}  // namespace rch