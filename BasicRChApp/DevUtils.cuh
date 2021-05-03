#pragma once
#include "cuda.h"
#include "cuda_d3d11_interop.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Maths3D.h"

/*
 * PTODO: Use rch::cuda namespace? Rather not since this is a device code by definition
 */

namespace rch {

#pragma region Common Structures

struct CuVec3 {
  float x, y, z;

  __device__ CuVec3() : x(0), y(0), z(0){};
  __device__ CuVec3(float x1, float y1, float z1) : x(x1), y(y1), z(z1){};
  __device__ CuVec3(float comVal) : x(comVal), y(comVal), z(comVal){};

  //__device__ CuVec3(CuVec3&&) noexcept = default;
  //__device__ CuVec3& operator=(CuVec3&& rv) noexcept = default;
  //__device__ CuVec3(const CuVec3& orig) = default;
  //__device__ CuVec3& operator=(const CuVec3& orig) = default;

  __device__ CuVec3(const Vec3& src) {
    this->x = src.x;
    this->y = src.y;
    this->z = src.z;
  };

  __device__ CuVec3 operator=(const Vec3& src) {
    CuVec3 res;
    res.x = src.x;
    res.y = src.y;
    res.z = src.z;
    return res;
  };

  __device__ CuVec3& operator+=(const CuVec3& v2) {
    this->x = this->x + v2.x;
    this->y = this->y + v2.y;
    this->z = this->z + v2.z;
    return *this;
  };

  __device__ friend CuVec3 operator+(CuVec3 v1, const CuVec3& v2) {
    CuVec3 res;
    res.x = v1.x + v2.x;
    res.y = v1.y + v2.y;
    res.z = v1.z + v2.z;
    return res;
  };

  __device__ CuVec3& operator-=(CuVec3& v2) {
    this->x = this->x - v2.x;
    this->y = this->y - v2.y;
    this->z = this->z - v2.z;
    return *this;
  };

  __device__ friend CuVec3 operator-(CuVec3 v1, const CuVec3& v2) {
    CuVec3 res;
    res.x = v1.x - v2.x;
    res.y = v1.y - v2.y;
    res.z = v1.z - v2.z;
    return res;
  };

  __device__ CuVec3& operator*=(float scalar) {
    this->x = this->x * scalar;
    this->y = this->y * scalar;
    this->z = this->z * scalar;
    return *this;
  };

  __device__ friend CuVec3 operator*(const float scalar, CuVec3 v1) {
    CuVec3 res;
    res.x = v1.x * scalar;
    res.y = v1.y * scalar;
    res.z = v1.z * scalar;
    return res;
  };

  __device__ friend CuVec3 operator*(CuVec3 v1, const float scalar) {
    CuVec3 res;
    res.x = v1.x * scalar;
    res.y = v1.y * scalar;
    res.z = v1.z * scalar;
    return res;
  };

  __device__ friend CuVec3 operator/(CuVec3 v1, const float scalar) {
    if (scalar == 0) {
      return 0; // PTODO
    }
    CuVec3 res;
    res.x = v1.x / scalar;
    res.y = v1.y / scalar;
    res.z = v1.z / scalar;
    return res;
  };

  __device__ friend bool operator==(CuVec3 v1, const CuVec3& v2) {
    if (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z) {
      return true;
    }
    return false;
  };

  /*
      General functions 
  */

  __device__ void normalize() {
    float length = std::sqrt(CuVec3::dotProduct(*this, *this));
    *this = (*this) / length;
  };
  
  __device__ void saturate() {
    this->x = this->x > 1.0f ? 1.0f : this->x;
    this->x = this->x < 0.0f ? 0.0f : this->x;

    this->y = this->y > 1.0f ? 1.0f : this->y;
    this->y = this->y < 0.0f ? 0.0f : this->y;

    this->z = this->z > 1.0f ? 1.0f : this->z;
    this->z = this->z < 0.0f ? 0.0f : this->z;
  };

  __device__ void addToXYZ(float scalar) {
    this->x = this->x + scalar;
    this->y = this->y + scalar;
    this->z = this->z + scalar;
  };

  /*
      Static functions 
  */

  __device__ static float dotProduct(const CuVec3& v1, const CuVec3& v2) {
    return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
  }

  __device__ static float distance(const CuVec3& v1, const CuVec3& v2) {
    const CuVec3 diff = v2 - v1;
    return std::sqrt(CuVec3::dotProduct(diff, diff));
  };

  __device__ static CuVec3 getNormalized(const CuVec3& v1) {
    float length = std::sqrt(CuVec3::dotProduct(v1, v1));
    return v1 / length;
  };

};

#pragma endregion


#pragma region Vector & Matrix Algebra - Functions

__device__ CuVec3 mtxVecRMul_3x3(const CuVec3& vec, float* mtx);

__device__ CuVec3 mtxVecCMul_3x3(float* mtx, const CuVec3& vec);

__device__ CuVec3 mtxVecRMul_4x4(const CuVec3& vec, float vecw, float* mtx);

__device__ CuVec3 mtxVecCMul_4x4(float* mtx, const CuVec3& vec, float vecw);

#pragma endregion

#pragma region Computer Graphics

//__device__ CuVec3 calcBlinnPhongReflection(CuVec3 point, CuVec3 norm,
//                                           CuVec3 toEye,
//                                           const CuLightSrc* lights,
//                                           std::size_t lightNum);
//
//__device__ CuVec3 calcPhongReflection(CuVec3 point, CuVec3 norm, CuVec3 toEye,
//                                      const CuLightSrc* lights,
//                                      std::size_t lightNum);

#pragma endregion
}  // namespace rch
