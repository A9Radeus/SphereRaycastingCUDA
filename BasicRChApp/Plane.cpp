#include "Plane.h"
#include "Mesh.h"

extern
void testRR3() {
  int ad2 = 3;
  auto ff = 0;
  return;
}

namespace rch {

Plane::Plane(const mini::DxDevice& device, float width, float height)
    : SceneObj(device), m_width(width), m_height(height) {
  auto inds = mini::Mesh::BoxIdxs();
  m_idxNum = inds.size();
  m_idxBuff = m_dev.CreateIndexBuffer(inds);
}

void Plane::update(double dtime) {
  if (m_meshReady == false) {
    generateMesh();
  }
}

void Plane::render() {
  if (m_meshReady == false) {
    return;
  }
  unsigned int strides = sizeof(mini::VertexPosition);
  unsigned int offset = 0;

  m_dev.context()->IASetPrimitiveTopology(
      D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  m_dev.context()->IASetIndexBuffer(m_idxBuff.get(), DXGI_FORMAT_R16_UINT, 0);
  m_dev.context()->IASetVertexBuffers(0, m_vertBuff.size(), m_vertBuff.data(),
                                      &strides, &offset);
  m_dev.context()->DrawIndexed(m_idxNum, 0, 0);
}

void Plane::generateMesh() {
  // auto verts = mini::Mesh::ShadedBoxVerts(m_sideLen);
  auto verts = mini::Mesh::RectangleVertsPos(m_width, m_height);

  m_vertBuff.clear();
  m_vertBuff.push_back(m_dev.CreateVertexBuffer(verts));
  m_meshReady = true;
}

}  // namespace rch