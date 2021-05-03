#include "sphere_raycasting.cuh"
//#include "DevUtils.cuh"
#include <cmath>

namespace rch {

#pragma region Utility functions
__device__ void setCol(float *pixel, float r, float g, float b, float a = 1) {
  pixel[0] = r;  // red
  pixel[1] = g;  // green
  pixel[2] = b;  // blue
  pixel[3] = a;  // alpha
};

__device__ void setCol(float *pixel, const CuVec3& rgb, float a = 1) {
  pixel[0] = rgb.x;  // red
  pixel[1] = rgb.y;  // green
  pixel[2] = rgb.z;  // blue
  pixel[3] = a;  // alpha
};
#pragma endregion

__global__ void kernel_sphere_raycast(unsigned char *surface, int width,
                                      int height, size_t pitch,
                                      const CuSphere *spheres, std::size_t spheresNum,
                                      const CuLightSrc *lights,
                                      std::size_t ligthsNum, float* invMtxVP) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Skip the redundant threads (quantization)
  if (x >= width || y >= height) return;

  // Pointer to the pixel at (x,y)
  float* pixel = (float *)(surface + y * pitch) + 4 * x;

  // Clear RenderTarget to black colour
  setCol(pixel, 0.1f, 0.1f, 0.1f, 1);

  // Projection requires NDC
  float x2 = (2.0f * x) / width - 1.0f;
  float y2 = 1.0f - (2.0f * y) / height; /* Y is flipped */

  //// Get Ray's origin and direction in world-space
  const CuVec3 rayOrigC = {x2,y2,0};
  const CuVec3 rayOrigW = mtxVecRMul_4x4(rayOrigC, 1, invMtxVP);
  const CuVec3 rayDestC = {x2,y2,1};
  const CuVec3 rayDestW = mtxVecRMul_4x4(rayDestC, 1, invMtxVP);
  const CuVec3 rayDirW = CuVec3::getNormalized(rayDestW - rayOrigW);

  // The Raycasting
  float tBuff = INFINITY;  // z-buffer that uses tSol
  for (std::size_t i = 0; i < spheresNum; i++) {
    float tSol = 0;
    if (raySphereIntersection(
            rayOrigW, rayDirW, spheres[i].pos, spheres[i].radius, &tSol)) {
      const CuVec3 spherePt = rayOrigW + (tSol * rayDirW);

      // t-buff checking
      if (tSol >= tBuff) {
        continue;
      }

      tBuff = tSol;

      const CuVec3 norm = CuVec3::getNormalized(spherePt - spheres[i].pos);
      const CuVec3 toEye = -1.f * rayDirW; // unit-vec * -1 = unit-vec

      // Since for every pixel  there is a toEye vector,
      // specular may occur multiple times
      const auto col = calcPhongReflection(spherePt, norm, toEye, lights, ligthsNum);
      //const auto col = calcBlinnPhongReflection(spherePt, norm, toEye, lights, ligthsNum);
      setCol(pixel, col);
    }
  }
}

void sphere_raycast_to_tex2d(const CuSphere *spheresArr, std::size_t spheresNum,
                             const CuLightSrc *lightsArr, std::size_t lightsNum,
                             const cuda::InteropTexture2D &itex,
                             const XMMATRIX &xmInvVpMtx) {
  cudaArray *cuArray;
  auto err = cudaGraphicsSubResourceGetMappedArray(&cuArray,
                                                   itex.m_cudaResource, 0, 0);
  cuda::gpuErrorCheck(err, "[cudaGraphicsSubResourceGetMappedArray]");

  float *invVpMtx = cuda::xmmatToMtx4x4(xmInvVpMtx);
  float *devInvVpMtx = cuda::copyToDevMem(invVpMtx, 16);

  /*
      Run CUDA kernel
  */

  dim3 Db = dim3(32, 32);
  dim3 Dg = dim3(std::ceil(itex.m_width / Db.x),
                 std::ceil((itex.m_height) / Db.y) + 1);
  kernel_sphere_raycast<<<Dg, Db>>>((unsigned char *)itex.m_cudaLinearMemory,
                                    itex.m_width, itex.m_height, itex.m_pitch,
                                    spheresArr, spheresNum, lightsArr,
                                    lightsNum, devInvVpMtx);

  /*
      Cleanup
  */

  delete[] invVpMtx;
  err = cudaFree(devInvVpMtx);
  cuda::gpuErrorCheck(err, "[sphere_raycast_to_tex2d] cudaFree(devInvVpMtx)");

  /*
      Copy the results
  */

  // Copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
  // "itex.m_width * 4" since we re using rgba
  err = cudaMemcpy2DToArray(cuArray,                                // dst array
                            0, 0,                                   // offset
                            itex.m_cudaLinearMemory, itex.m_pitch,  // src
                            (size_t)itex.m_width * 4 * sizeof(float),
                            itex.m_height,              // extent
                            cudaMemcpyDeviceToDevice);  // kind
  cuda::gpuErrorCheck(err, "[cudaMemcpy2DToArray]");
}


// @See solveRaySphereIntersection() in Maths3D.cpp
__device__ bool raySphereIntersection(const CuVec3 &rayOrigin,
                                      const CuVec3 &rayDirection,
                                      const CuVec3 &spherePos, float radius,
                                      float *sol1, float *sol2) {
  /*
    Get coefficients for the quadratic equation
  */

  // Ray origin transformed to the frame of reference matching
  // the spheres position.
  const CuVec3 tRayOrig = {rayOrigin.x - spherePos.x, rayOrigin.y - spherePos.y,
                         rayOrigin.z - spherePos.z};

  // t^2 * dot(dir, dir)
  float a = (rayDirection.x * rayDirection.x) +
            (rayDirection.y * rayDirection.y) +
            (rayDirection.z * rayDirection.z);

  // 2*t * dot(dir, org - C)
  float b =
      2.0f * ((rayDirection.x * tRayOrig.x) + (rayDirection.y * tRayOrig.y) +
              (rayDirection.z * tRayOrig.z));

  // dot(orig - C, orig - C) - r^2
  float c = ((tRayOrig.x * tRayOrig.x) + (tRayOrig.y * tRayOrig.y) +
             (tRayOrig.z * tRayOrig.z)) -
            (radius * radius);

  /*
    Find the intersection points, i.e. solve quadratic eq.
  */

  float delta = (b * b) - (4 * a * c);
  if (delta < 0.0f) {
    return false;
  }

  const double sqrtDelta = std::sqrt(delta);
  if (delta == 0.0f) {
    *sol1 = static_cast<float>(-(b / (2.0 * a)));
    *sol2 = *sol1;
  } else {
    *sol1 = static_cast<float>((-b + sqrtDelta) / (2.0 * a));
    *sol2 = static_cast<float>((-b - sqrtDelta) / (2.0 * a));
  }

  return true;
}

// Wraper that returns the closest solution
__device__ bool raySphereIntersection(const CuVec3 &rayOrigin,
                                      const CuVec3 &rayDirection,
                                      const CuVec3 &spherePos, float radius,
                                      float *sol) {
  
  float s1 = -1;
  float s2 = -1;

  if (raySphereIntersection(rayOrigin, rayDirection, spherePos, radius, &s1, &s2)) {
    if (s1 < s2) { *sol = s1; }
    else         { *sol = s2; }
    return true;
  } 

  return false;
}

// Foreach light : result += Ambient + Diffuse + Specular
__device__ CuVec3 calcBlinnPhongReflection(CuVec3 point, CuVec3 norm,
                                           CuVec3 toEye,
                                           const CuLightSrc *lights,
                                           std::size_t lightNum) {
  // constexpr float ka = 0.1f;   // ambient constant
  constexpr float kd = 0.5f;  // diffuse reflection constant
  constexpr float ks = 0.5f;  // specular reflection constant
  constexpr float m = 100.f;  // shininess of a material

  CuVec3 resCol = {0, 0, 0};
  for (int i = 0; i < lightNum; i++) {
    const CuVec3 lightCol = lights[i].col;
    const CuVec3 lightPos = lights[i].pos;
    const CuVec3 toLight = CuVec3::getNormalized(lightPos - point);
    const CuVec3 halfVec = CuVec3::getNormalized(toEye + toLight);

    float dotLN = CuVec3::dotProduct(toLight, norm);
    dotLN = dotLN > 1.f ? 1.f : dotLN;
    dotLN = dotLN < 0.f ? 0.f : dotLN;

    //// Diffuse
    // resCol += lightCol * kd * dotLN;

    // Specular
    float nh = CuVec3::dotProduct(norm, halfVec);
    nh = nh > 1.f ? 1.f : nh;
    nh = nh < 0.f ? 0.f : nh;
    nh = pow(nh, m);
    nh *= ks;
    resCol += lightCol * nh;
  }

  resCol.saturate();
  return resCol;
}

// Foreach light : result += Ambient + Diffuse + Specular
__device__ CuVec3 calcPhongReflection(CuVec3 point, CuVec3 norm, CuVec3 toEye,
                                      const CuLightSrc *lights,
                                      std::size_t lightNum) {
  constexpr float ka = 0.1f;     // ambient constant
  constexpr float kd = 0.5f;     // diffuse reflection constant
  constexpr float ks = 0.5f;     // specular reflection constant
  constexpr float alf = 1000.f;  // shininess of a material

  CuVec3 resCol = {ka, ka, ka};  // ambient
  // CuVec3 resCol = {0, 0, 0};
  for (int i = 0; i < lightNum; i++) {
    const CuVec3 lightCol = lights[i].col;
    const CuVec3 lightPos = lights[i].pos;
    const CuVec3 toLight = CuVec3::getNormalized(lightPos - point);
    const float dotLN = CuVec3::dotProduct(toLight, norm);
    const CuVec3 reflection = (2.f * dotLN * norm) - toLight;

    //// Diffuse
    resCol += lightCol * kd * dotLN;

    // Specular
    resCol +=
        lightCol * ks * std::powf(CuVec3::dotProduct(reflection, toEye), alf);
  }

  resCol.saturate();
  return resCol;
}

}  // namespace rch