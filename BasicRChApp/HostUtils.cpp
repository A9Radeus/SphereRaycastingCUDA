#include "HostUtils.h"

using namespace DirectX;

namespace rch {
namespace cuda {

void kernelErrorCheck() {
  auto cudaStatus = cudaPeekAtLastError();
  if (cudaStatus != cudaSuccess) {
    std::string err = "[CUDA kernel error] ";
    err += cudaGetErrorString(cudaStatus);
    throw err;
  }
}

void gpuErrorCheck(cudaError_t val, const char* msg) {
  if (val != cudaSuccess) {
    std::string err = "[CUDA error] ";
    if (msg != nullptr) {
      err += msg;
      err += " ";
    }
    err += cudaGetErrorString(val);
    throw err;
  }
}

void gpuErrorCheck(cudaError_t val, const std::string& msg) {
  if (val != cudaSuccess) {
    std::string err = "[CUDA error] " + msg + " ";
    err += cudaGetErrorString(val);
    throw err;
  }
}

float* xmmatToMtx3x3(const DirectX::XMMATRIX& xmmat) {
  XMFLOAT4X4 aux;
  XMStoreFloat4x4(&aux, xmmat);

  float* resMtx = new float[9];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      resMtx[i] = aux(i, j);
    }
  }

  return resMtx;
}

float* xmmatToMtx4x4(const XMMATRIX& xmmat) {
  XMFLOAT4X4 aux;
  XMStoreFloat4x4(&aux, xmmat);

  float* resMtx = new float[16];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      resMtx[i * 4 + j] = aux(i, j);
    }
  }

  return resMtx;
}

// TODO: this function is WIP
// PTODO: as of now throws if an error occurred
// Returns the number of CUDA-supporting Devices found on the machine.
// If zero is returned then Cuda is not available.
int findDevices() {
  int deviceCount = 0;
  // char firstGraphicsName[NAME_LEN], devname[NAME_LEN];

  // This function call returns 0 if there are no CUDA capable devices.
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess) {
    throw cudaGetErrorString(error_id);
  }

  return deviceCount;

  //// Get CUDA device properties
  // cudaDeviceProp deviceProp;

  // for (int dev = 0; dev < deviceCount; ++dev) {
  //  cudaGetDeviceProperties(&deviceProp, dev);
  //  STRCPY(devname, NAME_LEN, deviceProp.name);
  //  printf("> GPU %d: %s\n", dev, devname);
  //}
}

}  // namespace cuda
}  // namespace rch
