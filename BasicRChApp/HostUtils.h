#pragma once
#include "cuda.h"
#include "cuda_d3d11_interop.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dxDevice.h"
#include "exceptions.h"
#include <DirectXMath.h>

namespace rch {
namespace cuda {

#pragma region Initialisation
  int findDevices(); 
#pragma endregion

#pragma region Error Handling
void kernelErrorCheck();
void gpuErrorCheck(cudaError_t val, const char *msg = nullptr);
void gpuErrorCheck(cudaError_t val, const std::string &msg);
#pragma endregion

#pragma region Common Structures

/* Data structure for 2D texture shared between DX11 and CUDA
 * The cudaGraphicsResource has to be mapped firstly by one of the
 * "cudaGraphicsMapResources" functions.
 * Important:
 *    "The resources in resources may be accessed by CUDA
 *    until they are unmapped. The graphics API from which resources were
 *    registered should not access any resources while they are mapped by CUDA.
 *    If an application does so, the results are undefined."
 */
struct InteropTexture2D {
  ID3D11Texture2D *m_pTexture;
  ID3D11ShaderResourceView *m_pSRView;
  cudaGraphicsResource *m_cudaResource;
  void *m_cudaLinearMemory;
  size_t m_pitch;
  int m_width;
  int m_height;

  InteropTexture2D(const mini::DxDevice &dev, int width, int height)
      : m_width(width),
        m_height(height),
        m_pTexture(nullptr),
        m_pSRView(nullptr),
        m_cudaResource(nullptr),
        m_cudaLinearMemory(nullptr) {
    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
    desc.Width = m_width;
    desc.Height = m_height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    auto hr = dev->CreateTexture2D(&desc, NULL, &m_pTexture);
    if (FAILED(hr)) {
      THROW_DX(hr);
    }

    hr = dev->CreateShaderResourceView(m_pTexture, NULL, &m_pSRView);
    if (FAILED(hr)) {
      THROW_DX(hr);
    }

    dev.context()->PSSetShaderResources(0, 1, &m_pSRView);

    registerCudaResc();
  }

  ~InteropTexture2D() {
    // CUDA resources
    cudaError_t err = cudaGraphicsUnregisterResource(m_cudaResource);
    gpuErrorCheck(err, "[cudaGraphicsUnregisterResource(m_cudaResource)]");

    err = cudaFree(m_cudaLinearMemory);
    gpuErrorCheck(err, "[cudaFree(m_cudaLinearMemory)]");

    // DirectX Resources
    m_pSRView->Release();
    m_pTexture->Release();
  }

  void setAsPShaderResc(const mini::DxDevice& dev) {
    dev.context()->PSSetShaderResources(0, 1, &m_pSRView);
  }

  void registerCudaResc() {
    // register the Direct3D resources 
    auto err = cudaGraphicsD3D11RegisterResource(&m_cudaResource, m_pTexture,
                                                 cudaGraphicsRegisterFlagsNone);
    gpuErrorCheck(err, "[cudaGraphicsD3D11RegisterResource]");

    // cuda cannot write into the texture directly : the texture is seen as a
    // cudaArray and can only be mapped as a texture Create a buffer so that
    // cuda can write into it pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
    err = cudaMallocPitch(&m_cudaLinearMemory, &m_pitch,
                          m_width * sizeof(float) * 4, m_height);
    gpuErrorCheck(err, "[cudaGraphicsD3D11RegisterResource]");
    cudaMemset(m_cudaLinearMemory, 1, m_pitch * m_height);
  }

  void getResc() {
    // HRESULT hr;
    // D3D11_MAPPED_SUBRESOURCE mappedResource;
    // ConstantBuffer *pcb;
    // hr = g_pd3dDeviceContext->Map(g_pConstantBuffer, 0,
    // D3D11_MAP_WRITE_DISCARD,
    //                              0, &mappedResource);
    // AssertOrQuit(SUCCEEDED(hr));
    // pcb = (ConstantBuffer *)mappedResource.pData;
    //{
    //  memcpy(pcb->vQuadRect, quadRect, sizeof(float) * 4);
    //  pcb->UseCase = 0;
    //}
    // g_pd3dDeviceContext->Unmap(g_pConstantBuffer, 0);
    // g_pd3dDeviceContext->Draw(4, 0);
  }
};

#pragma endregion

#pragma region DirectX Utils

float* xmmatToMtx3x3(const DirectX::XMMATRIX& xmmat);
float* xmmatToMtx4x4(const DirectX::XMMATRIX& xmmat);

#pragma endregion

#pragma region copyToDevMem
template <typename TData>
static TData* copyToDevMem(const std::vector<TData>& vec) {
  TData* devArr = nullptr;
  cudaError_t err;

  // Alloc device array
  err = cudaMalloc((void**)&devArr, vec.size() * sizeof(TData));
  gpuErrorCheck(err, "cudaMalloc");

  // Copy input array from host to dev
  err = cudaMemcpy(devArr, vec.data(), vec.size() * sizeof(TData),
                   cudaMemcpyHostToDevice);
  gpuErrorCheck(err, "cudaMemcpy");

  return devArr;
}

template <typename TData>
static TData* copyToDevMem(TData* arr, std::size_t arrSize) {
  TData* devArr = nullptr;
  cudaError_t err;

  // Alloc device array
  err = cudaMalloc((void**)&devArr, arrSize * sizeof(TData));
  gpuErrorCheck(err, "cudaMalloc");

  // Copy input array from host to dev
  err =
      cudaMemcpy(devArr, arr, arrSize * sizeof(TData), cudaMemcpyHostToDevice);
  gpuErrorCheck(err, "cudaMemcpy");

  return devArr;
}
#pragma endregion

}  // namespace cuda
}  // namespace rch