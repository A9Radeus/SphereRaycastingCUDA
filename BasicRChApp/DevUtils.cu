#include "DevUtils.cuh"

namespace rch {

// Returns vec * mtx. The vector "vec" is assumed to be a ROW-vector
// and "mtx" a 3x3 matrix, i.e. size(mtx) = 9
__device__ CuVec3 mtxVecRMul_3x3(const CuVec3& vec, float* mtx) {
  CuVec3 res = {0, 0, 0};

  res.x = vec.x * mtx[0] + vec.y * mtx[1] + vec.z * mtx[2];
  res.y = vec.x * mtx[3] + vec.y * mtx[4] + vec.z * mtx[5];
  res.z = vec.x * mtx[6] + vec.y * mtx[7] + vec.z * mtx[8];

  return res;
}

// Returns mtx * vec. The vector "vec" is assumed to be a COLUMN-vector
// and "mtx" a 3x3 matrix, i.e. size(mtx) = 9
__device__  CuVec3 mtxVecCMul_3x3(float* mtx, const CuVec3& vec) {
  CuVec3 res = {0, 0, 0};

  res.x = vec.x * mtx[0] + vec.y * mtx[1] + vec.z * mtx[2];
  res.y = vec.x * mtx[3] + vec.y * mtx[4] + vec.z * mtx[5];
  res.z = vec.x * mtx[6] + vec.y * mtx[7] + vec.z * mtx[8];

  return res;
}

// Returns vec * mtx. The vector "vec" is assumed to be a ROW-vector
// and "mtx" a 4x4 matrix, i.e. size(mtx) = 16
__device__ CuVec3 mtxVecRMul_4x4(const CuVec3& vec, float vecw, float* mtx) {
  CuVec3 res = {0, 0, 0};

  res.x = vec.x * mtx[0] + vec.y * mtx[4] + vec.z * mtx[8] + vecw * mtx[12];
  res.y = vec.x * mtx[1] + vec.y * mtx[5] + vec.z * mtx[9] + vecw * mtx[13];
  res.z = vec.x * mtx[2] + vec.y * mtx[6] + vec.z * mtx[10] + vecw * mtx[14];
  // float w = vec.x * mtx[3] + vec.y * mtx[7] + vec.z * mtx[11] + vecw *
  // mtx[15];

  // return {res.x / w, res.y / w, res.z / w};
  return res;
}

// Returns mtx * vec. The vector "vec" is assumed to be a COLUMN-vector
// and "mtx" a 4x4 matrix, i.e. size(mtx) = 16
__device__  CuVec3 mtxVecCMul_4x4(float* mtx, const CuVec3& vec, float vecw) {
  CuVec3 res = {0, 0, 0};

  res.x = vec.x * mtx[0] + vec.y * mtx[1] + vec.z * mtx[2] + vecw * mtx[3];
  res.y = vec.x * mtx[4] + vec.y * mtx[5] + vec.z * mtx[6] + vecw * mtx[7];
  res.z = vec.x * mtx[8] + vec.y * mtx[9] + vec.z * mtx[10] + vecw * mtx[11];

  return res;
}

}
