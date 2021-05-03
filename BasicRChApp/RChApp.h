#pragma once
#include <cassert>
#include <array>
#include <random>

#include "dxApplication.h"
#include "mesh.h"

#include "SimCube.h"
#include "SceneObj.h"
#include "GuiWidgets.h"

#include "sphere_raycasting.cuh"
#include "HostUtils.h"

namespace rch {

class RChApp : public mini::DxApplication {
public:
  using Base = mini::DxApplication;

  explicit RChApp(HINSTANCE appInstance);
  ~RChApp();

  bool HandleCameraInput(double dt);

 protected:
  void Update(const mini::Clock& dt) override;
  void Render() override;

private: 
  void genRandomSpheres(int number, float maxPos = 10.f, float maxRadius = 1.f);
  void genRandomLights(int number, float maxPos = 10.f);

  std::random_device m_randDev;
  std::mt19937 m_randGen;

  mini::dx_ptr<ID3D11Buffer> m_cbWorldMtx, m_cbProjMtx;                
  mini::dx_ptr<ID3D11Buffer> m_cbViewMtx;  
  mini::dx_ptr<ID3D11Buffer> m_cbSurfaceColor;  
  mini::dx_ptr<ID3D11Buffer> m_cbLightPos;      
  mini::dx_ptr<ID3D11Buffer> m_cbCamPos;

  std::vector<std::shared_ptr<rch::SceneObj>> m_sobjs;

  DirectX::XMFLOAT4X4 m_projMtx;

  mini::dx_ptr<ID3D11InputLayout> m_layoutVertPosNorm;
  mini::dx_ptr<ID3D11InputLayout> m_layoutVertPos;

  mini::dx_ptr<ID3D11VertexShader> m_defaultVS, m_basicTex2D_VS;
  mini::dx_ptr<ID3D11PixelShader> m_defaultPS, m_basicTex2D_PS;

  //mini::mini::dx_ptr<ID3D11DepthStencilView> m_depthShadowBuff;

  // Rasterizers
  mini::dx_ptr<ID3D11RasterizerState> m_rsDefault;
  mini::dx_ptr<ID3D11RasterizerState> m_rsCCW;

  mini::dx_ptr<ID3D11SamplerState> m_texSampler;

  /*
      GUI 
  */
  std::shared_ptr<TextPopup> m_popupInfo;

  /*
      Utility functions
  */
  void drawScene();
  void updateGUI();
  void updateCudaTexture();

  void UpdateCameraCB(DirectX::XMMATRIX viewMtx);
  void UpdateCameraCB() { UpdateCameraCB(m_camera.getViewMatrix()); }

  void DrawMesh(const mini::Mesh& m, DirectX::XMFLOAT4X4 worldMtx);

  void SetWorldMtx(DirectX::XMFLOAT4X4 mtx);
  void SetWorldMtx(const DirectX::XMMATRIX& mtx);
  void SetSurfaceColor(DirectX::XMFLOAT4 color);
  void SetShaders(const mini::dx_ptr<ID3D11VertexShader>& vs,
                  const mini::dx_ptr<ID3D11PixelShader>& ps);
  void SetTextures(std::initializer_list<ID3D11ShaderResourceView*> resList,
                   const mini::dx_ptr<ID3D11SamplerState>& sampler);


  /*
     Constants
  */
  static const DirectX::XMFLOAT4 LIGHT_POS;
  static constexpr float CAMERA_ZOOM = 10.0f;
  static constexpr float CAMERA_SPEED = 3.0f;

  /*
      CUDA
  */
  CuSphere* m_cudaSpheresArr = nullptr;
  CuLightSrc* m_cudaLightsArr = nullptr;
  bool textureUpdated = false;
  cuda::InteropTexture2D m_interopTex2D;
  std::vector<CuSphere> m_spheres;
  std::size_t m_spheresNum = 0;
  std::vector<CuLightSrc> m_lights;
  std::size_t m_lightsNum = 0;
};


}  // namespace rch