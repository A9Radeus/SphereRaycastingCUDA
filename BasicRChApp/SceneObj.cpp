#include "SceneObj.h"

using namespace DirectX;

namespace rch {

DirectX::XMMATRIX SceneObj::getModelMtx() const {
  if ((m_angleX == 0) && (m_angleY == 0) && (m_angleZ == 0)) {
    return rch::gen_scale_mtx(m_scaling.x, m_scaling.y, m_scaling.z) *
           rch::gen_translation_mtx(m_pos.x, m_pos.y, m_pos.z);
  } else {
    return rch::gen_rotx_mtx(m_angleX) * rch::gen_roty_mtx(m_angleY) *
           rch::gen_rotz_mtx(m_angleZ) *
           rch::gen_scale_mtx(m_scaling.x, m_scaling.y, m_scaling.z) *
           rch::gen_translation_mtx(m_pos.x, m_pos.y, m_pos.z);
  }
}

// TODO ! @see commented lines below
DirectX::XMMATRIX SceneObj::getInvModelMtx() const {
  // return rch::gen_translation_mtx(-m_pos.x, -m_pos.y, -m_pos.z) *
  //        rch::gen_rotz_mtx(-m_angleZ) *
  //        rch::gen_rotz_mtx(-m_angleY) *
  //        rch::gen_roty_mtx(-m_angleX);
  auto invMtx = XMMatrixInverse(nullptr, getModelMtx());
  return invMtx;
}

void SceneObj::translate(rch::Vec3 vec) {
  m_pos += vec;
}

void SceneObj::rotateX(double angle) {
  m_angleX = XMScalarModAngle(m_angleX + angle);
}

void SceneObj::rotateY(double angle) {
  m_angleY = XMScalarModAngle(m_angleY + angle);
}

void SceneObj::rotateZ(double angle) {
  m_angleZ = XMScalarModAngle(m_angleZ + angle);
}

void SceneObj::setModelMtxParams(double angleX, double angleY, double angleZ,
                                 rch::Vec3 pos) {
  m_angleX = angleX;
  m_angleY = angleY;
  m_angleZ = angleZ;
  m_pos = pos;
}

void SceneObj::setModelMtxParams(const SceneObj& secObj) {
  m_angleX = secObj.getAngleX();
  m_angleY = secObj.getAngleY();
  m_angleZ = secObj.getAngleZ();
  m_pos = secObj.getCenterPos();
}

}  // namespace rch