#pragma once
#include <d3d11.h>

#include "SceneObj.h"
#include "dxDevice.h"
#include "dxptr.h"

namespace rch {

class Plane : public SceneObj {
 public:
  Plane(const mini::DxDevice& device, float width, float height);
  virtual ~Plane() = default;

  void update(double dtime = 0.0) override;
  void render() override;

  void setDimensions(float width, float height) {
    setWidth(width);
    setHeight(height);
  }

  void setWidth(float width) {
    m_width = width;
    m_meshReady = false;
  };

  void setHeight(float height) {
    m_height = height;
    m_meshReady = false;
  };


 private:
  // ----------- Functions -----------
  void generateMesh();

  // ----------- Variables -----------
  mini::dx_ptr_vector<ID3D11Buffer> m_vertBuff;
  mini::dx_ptr<ID3D11Buffer> m_idxBuff;
  unsigned int m_idxNum = 0;
  bool m_meshReady = false;

  // Plane attributes
  float m_width ;
  float m_height;
};

}  // namespace rch