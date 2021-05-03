#include <algorithm>
#include <cmath>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"
//#include "implot.h"

// CUDA
#include "HostUtils.h"

#include "Plane.h"
#include "RchApp.h"

using namespace mini;
using namespace DirectX;

namespace rch {

const XMFLOAT4 RChApp::LIGHT_POS = {5.0f, 12.0f, -5.0f, 1.0f};

RChApp::~RChApp() {
  // ImGui 
  ImGui_ImplDX11_Shutdown();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();
  // ImPlot 
  // ImPlot::DestroyContext();

  // Cuda
  if (m_cudaSpheresArr != nullptr) {
    cudaFree(m_cudaSpheresArr);
  }
  if (m_cudaLightsArr != nullptr) {
    cudaFree(m_cudaLightsArr);
  }
}

RChApp::RChApp(HINSTANCE appInstance)
    : DxApplication(appInstance, 1280, 720, L"RChApp"),
      m_interopTex2D(m_device, 1280, 720),
      m_randGen(m_randDev()),
      // Constant Buffers
      m_cbWorldMtx(m_device.CreateConstantBuffer<XMFLOAT4X4>()),
      m_cbProjMtx(m_device.CreateConstantBuffer<XMFLOAT4X4>()),
      m_cbViewMtx(m_device.CreateConstantBuffer<XMFLOAT4X4, 2>()),
      m_cbSurfaceColor(m_device.CreateConstantBuffer<XMFLOAT4>()),
      m_cbLightPos(m_device.CreateConstantBuffer<XMFLOAT4>()),
      m_cbCamPos(m_device.CreateConstantBuffer<XMFLOAT4>()) {
  ///// CUDA Init
  if (cuda::findDevices() < 1) {
    throw "Could not find any CUDA-capable Devices";
  }

  ///// ImGui Init
  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
   ImGuiIO& io = ImGui::GetIO();
  (void)io;

  // Setup Dear ImGui style
  ImGui::StyleColorsClassic();

  // Setup Platform/Renderer bindings
  ImGui_ImplWin32_Init(m_window.getHandle());
  ImGui_ImplDX11_Init(m_device.get(), m_device.context().get());
  /////// ImPlot Init
  // ImPlot::CreateContext();

  // Camera settings (moved here in order to leave base class unchanged)
  m_camera.Zoom(CAMERA_ZOOM - 10.f);
  m_camera.MoveTarget(XMFLOAT3{0.0f, 0.0f, 0.0f});

  // Projection matrix
  auto s = m_window.getClientSize();
  auto ar = static_cast<float>(s.cx) / s.cy;
  XMStoreFloat4x4(&m_projMtx,
                  XMMatrixPerspectiveFovLH(XM_PIDIV4, ar, 0.01f, 100.0f));
  UpdateBuffer(m_cbProjMtx, m_projMtx);
  UpdateCameraCB();

  // GUI elements
  m_popupInfo = std::make_shared<TextPopup>("Info Popup", "Information not set yet.");

  // Scene Objects
  //SetWorldMtx(XMMatrixIdentity());
  //m_sobjs.push_back(std::make_shared<Plane>(m_device, 3.8, 2));
  m_sobjs.push_back(std::make_shared<Plane>(m_device, 3.56f, 2.f));
  //m_spheres = CuSphere::genRandSpheres(m_randGen, 100, 10, 1);
  //m_lights = CuLightSrc::genRandLightSrcs(m_randGen, 10, 10);
  ////m_spheres = {{{0, 0, 0}, 0.05f}, {{0, 0, -0.2}, 0.07f}};
  ////m_lights = {{{-1, 3, -10}, {1, 1, 1}},
  ////            {{6, 3, -10}, {1, 0, 0}},
  ////            {{6, 6, 10}, {0, 0, 1}}};
  ////m_cudaSpheresArr = cuda::copyToDevMem(m_spheres);
  ////m_spheresNum = 2;
  ////m_cudaLightsArr = cuda::copyToDevMem(m_lights);
  ////m_lightsNum = 3;

  genRandomLights(10);
  genRandomSpheres(100, 1.f, 0.05f);

  // Constant buffer's content
  UpdateBuffer(m_cbLightPos, LIGHT_POS);

  // Render states
  RasterizerDescription rsDesc;
   rsDesc.CullMode = D3D11_CULL_NONE; // TODO
  m_rsDefault = m_device.CreateRasterizerState(rsDesc);
  rsDesc.FrontCounterClockwise = true;
  m_rsCCW = m_device.CreateRasterizerState(rsDesc);
  m_device.context()->RSSetState(m_rsDefault.get());

  // Texture Sampler
  SamplerDescription sampDesc;
  sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
  sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_BORDER;
  sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_BORDER;
  sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
  sampDesc.BorderColor[0] = 0.0f;
  sampDesc.BorderColor[1] = 0.0f;
  sampDesc.BorderColor[2] = 0.0f;
  //sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
  //sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
  //sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
  //sampDesc.MaxAnisotropy = 16;
  sampDesc.MaxAnisotropy = 1;
  m_texSampler = m_device.CreateSamplerState(sampDesc); 

  // Shaders
  auto vsCode = m_device.LoadByteCode(L"defaultVS.cso");
  auto psCode = m_device.LoadByteCode(L"defaultPS.cso");
  m_defaultVS = m_device.CreateVertexShader(vsCode);
  m_defaultPS = m_device.CreatePixelShader(psCode);
  m_layoutVertPosNorm =
      m_device.CreateInputLayout(VertexPositionNormal::Layout, vsCode);
  
  vsCode = m_device.LoadByteCode(L"basicTex2D_VS.cso");
  psCode = m_device.LoadByteCode(L"basicTex2D_PS.cso");
  m_basicTex2D_VS = m_device.CreateVertexShader(vsCode);
  m_basicTex2D_PS = m_device.CreatePixelShader(psCode);
  m_layoutVertPos = m_device.CreateInputLayout(VertexPosition::Layout, vsCode);

  ID3D11Buffer* vsb[] = {m_cbWorldMtx.get(), m_cbViewMtx.get(),
                         m_cbProjMtx.get()};
  m_device.context()->VSSetConstantBuffers(0, 3, vsb);
  ID3D11Buffer* psb[] = {m_cbSurfaceColor.get(), m_cbLightPos.get(),
                         m_cbCamPos.get()};
  m_device.context()->PSSetConstantBuffers(0, 3, psb);

  //m_device.context()->IASetInputLayout(m_layoutVertPosNorm.get());

  m_device.context()->IASetPrimitiveTopology(
      D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

  //SetShaders(m_defaultVS, m_defaultPS);
  //UpdateBuffer(m_cbProjMtx, m_projMtx);
  SetShaders(m_basicTex2D_VS, m_basicTex2D_PS);
  m_device.context()->IASetInputLayout(m_layoutVertPos.get());
  auto sampPtr = m_texSampler.get();
  m_device.context()->PSSetSamplers(0, 1, &sampPtr);
  //m_device.context()->PSSetSamplers(0, 1, m_texSampler.get()));
  m_interopTex2D.setAsPShaderResc(m_device);
}

void RChApp::Update(const Clock& c) {
  double dt = c.getFrameTime();
  updateGUI();
  if (auto igio = ImGui::GetIO(); igio.WantTextInput == false 
                               && igio.WantCaptureMouse == false) {
    if (HandleCameraInput(dt)) {
      textureUpdated = false;
    } 
  }

  for (const auto& obj : m_sobjs) {
    obj->update(dt);
  }

  updateCudaTexture();
}

void RChApp::updateCudaTexture() {
  if (textureUpdated == true) {
    return;
  }

  cudaStream_t stream = 0;
  const int nbResources = 1;
  cudaGraphicsResource* ppResources[1] = {m_interopTex2D.m_cudaResource};

  // Map the texture to a CUDA resources
  auto err = cudaGraphicsMapResources(nbResources, ppResources, stream);
  cuda::gpuErrorCheck(err, "[cudaGraphicsMapResources]");

  // Raycast the spheres
  XMMATRIX VP =
      XMMatrixMultiply(m_camera.getViewMatrix(), XMLoadFloat4x4(&m_projMtx));

  sphere_raycast_to_tex2d(m_cudaSpheresArr, m_spheresNum, m_cudaLightsArr,
                          m_lightsNum, m_interopTex2D,
                          XMMatrixInverse(nullptr, VP));

  // Unmap CUDA resource
  err = cudaGraphicsUnmapResources(nbResources, ppResources, stream);
  cuda::gpuErrorCheck(err, "[cudaGraphicsUnmapResources]");

  textureUpdated = true;
}

void RChApp::drawScene() {
  // m_device.context()->RSSetState(m_rsCCW.get());
  // m_device.context()->RSSetState(m_rsCullFront.get());

  SetWorldMtx(XMMatrixTranslation(0,0,2.43));
  for (const auto& obj : m_sobjs) {
    obj->render();
  }
}

void RChApp::Render() {
  Base::Render();

  //ResetRenderTarget();
  UpdateCameraCB();
  drawScene();

  ImGui::Render();
  ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
}

void RChApp::genRandomSpheres(int number, float maxPos, float maxRadius) {
  if (number < 0) {
    return;
  }

  cudaFree(m_cudaSpheresArr);

  m_spheres = CuSphere::genRandSpheres(
      m_randGen, static_cast<std::size_t>(number), maxPos, maxRadius);
  m_spheresNum = static_cast<std::size_t>(number);

  // Copy/Map Resources to CUDA
  m_cudaSpheresArr = cuda::copyToDevMem(m_spheres);
}

void RChApp::genRandomLights(int number, float maxPos) {
  if (number < 0) {
    return;
  }
  
  // c-tor assures m_cudaLightsArr != nullptr
  cudaFree(m_cudaLightsArr);

  m_lights = CuLightSrc::genRandLightSrcs(
      m_randGen, static_cast<std::size_t>(number), maxPos);
  m_lightsNum = static_cast<std::size_t>(number);

  //Copy/Map Resources to CUDA
  m_cudaLightsArr = cuda::copyToDevMem(m_lights);
}

void RChApp::updateGUI() {
  // Start a new frame
  ImGui_ImplDX11_NewFrame();
  ImGui_ImplWin32_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Main", nullptr/*,
               ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize*/);
  if (ImGui::TreeNode("Generate a random scene")) {
    
    static int sphNum = 20;
    static float sphMaxPos = 1.f;
    static float sphMaxR = 0.2f;

    static int ligNum = 1;
    static float ligMaxPos = 10.f;
    
    ImGui::SliderInt("Num of spheres", &sphNum, 1, 5000);
    ImGui::SliderInt("Num of lights", &ligNum, 1, 100);
    ImGui::SliderFloat("Sphere: max pos", &sphMaxPos, 0.1f, 100);
    ImGui::SliderFloat("Sphere: max radius", &sphMaxR, 0.01f, 100);
    ImGui::SliderFloat("Light: max pos", &ligMaxPos, 0.01f, 100);

    if (ImGui::Button("Apply")) {
      genRandomSpheres(sphNum, sphMaxPos, sphMaxR);
      genRandomLights(ligNum, ligMaxPos);
      textureUpdated = false;
    }


    //ImGui::Separator();

    //if (ImGui::Button("Open Popup", {105, 0})) {
    //  m_popupInfo->open();
    //}

    ImGui::TreePop();
  }
  
  if (ImGui::TreeNode("Statistics")) {
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::TreePop();
  }

  if (ImGui::TreeNode("Help / How to use")) {
    ImGui::TextColored({0, 1, 0, 1}, "Scene generation:");
    ImGui::BulletText("CTRL + LMB on a slider: lets you set an exact value. \n However, it is recommended to use suggested  values.");

    ImGui::TextColored({0, 1, 0, 1}, "Camera:");
    ImGui::BulletText("RMB + cursor  (left / right): rotate the camera \n around the OY.");
    ImGui::BulletText("LMB + cursor  (up / down): rotate the camera \n around the OX (capped to [-180, 180] \n for simplicity).");
    ImGui::TreePop();
  }

  ImGui::End();

  //m_popupInfo->update();
}

void RChApp::SetSurfaceColor(DirectX::XMFLOAT4 color) {
  UpdateBuffer(m_cbSurfaceColor, color);
}

void RChApp::UpdateCameraCB(XMMATRIX viewMtx) {
  XMVECTOR det;
  XMMATRIX invViewMtx = XMMatrixInverse(&det, viewMtx);
  XMFLOAT4X4 view[2];
  XMStoreFloat4x4(view, viewMtx);
  XMStoreFloat4x4(view + 1, invViewMtx);
  UpdateBuffer(m_cbViewMtx, view);
  UpdateBuffer(m_cbCamPos, m_camera.getCameraPosition());
}

void RChApp::SetWorldMtx(DirectX::XMFLOAT4X4 mtx) {
  UpdateBuffer(m_cbWorldMtx, mtx);
}

void RChApp::SetWorldMtx(const XMMATRIX& mtx) {
  DirectX::XMFLOAT4X4 aux;
  XMStoreFloat4x4(&aux, mtx);
  UpdateBuffer(m_cbWorldMtx, aux);
}

void RChApp::SetShaders(const dx_ptr<ID3D11VertexShader>& vs,
                        const dx_ptr<ID3D11PixelShader>& ps) {
  m_device.context()->VSSetShader(vs.get(), nullptr, 0);
  m_device.context()->PSSetShader(ps.get(), nullptr, 0);
}

void RChApp::SetTextures(
    std::initializer_list<ID3D11ShaderResourceView*> resList,
    const dx_ptr<ID3D11SamplerState>& sampler) {
  m_device.context()->PSSetShaderResources(0, resList.size(), resList.begin());
  auto s_ptr = sampler.get();
  m_device.context()->PSSetSamplers(0, 1, &s_ptr);
}

void RChApp::DrawMesh(const Mesh& m, DirectX::XMFLOAT4X4 worldMtx) {
  SetWorldMtx(worldMtx);
  m.Render(m_device.context());
}

bool RChApp::HandleCameraInput(double dt) {
  // bool res = Base::HandleCameraInput(dt);
  bool res = false;

  // Handle mouse inputs
  MouseState mstate;
  if (m_mouse.GetState(mstate)) {
    auto mousePos = mstate.getMousePositionChange();
    if (mstate.isButtonDown(0)) {
      m_camera.Rotate(mousePos.y * ROTATION_SPEED, 0);
      res = true;
    } else if (mstate.isButtonDown(1)) {
      m_camera.Rotate(0, mousePos.x * ROTATION_SPEED);
      res = true;
    }

    if (m_camera.getXAngle() > XM_PIDIV2) {
      m_camera.setXAngle(XM_PIDIV2);
    } else if (m_camera.getXAngle() < -XM_PIDIV2) {
      m_camera.setXAngle(-XM_PIDIV2);
    }
  }
  return res;
}

}  // namespace rch