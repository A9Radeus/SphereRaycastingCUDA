#pragma once
#include "DxDevice.h"
#include "Maths3D.h"

namespace rch {

class SceneObj {
 public:
  SceneObj(const mini::DxDevice& dxDevice) : m_dev(dxDevice), m_pos(0.0f) {
    m_angleX = 0.0f;
    m_angleY = 0.0f;
    m_angleZ = 0.0f;
  };
  SceneObj(const SceneObj&) = delete;
  SceneObj& operator=(const SceneObj&) = delete;

  SceneObj(SceneObj&&) = delete;
  SceneObj& operator=(SceneObj&&) = delete;

  virtual ~SceneObj() = default;

  virtual void update(double dtime = 0.0) = 0;
  virtual void render() = 0;

  virtual void setRotation(Vec3 rot) {
    m_angleX = rot.x;
    m_angleY = rot.y;
    m_angleZ = rot.z;
  }

  //virtual void rotateXAround(double angle, Vec3 rotPoint);
  //virtual void rotateYAround(double angle, Vec3 rotPoint);
  //virtual void rotateZAround(double angle, Vec3 rotPoint);

  virtual void rotateX(double angle);
  virtual void rotateY(double angle);
  virtual void rotateZ(double angle);

  // TODO ! Scale isn't being handled in 1) modelMtx 2) App 3) setModelParams
  virtual void scale(Vec3 val) {
    m_scaling += val;
    m_prsChanged = true;
  };
  virtual void setScaling(Vec3 val) {
    m_scaling = val;
    m_prsChanged = true;
  };
  virtual void scaleX(float d) {
    m_scaling.x += d;
    m_prsChanged = true;
  };
  virtual void scaleY(float d) {
    m_scaling.y += d;
    m_prsChanged = true;
  };
  virtual void scaleZ(float d) {
    m_scaling.z += d;
    m_prsChanged = true;
  };

  virtual void translate(rch::Vec3 vec);
  virtual void setPos(const rch::Vec3 newPos) { m_pos = newPos; };

  /* Copies the parameters from another SceneObj */
  virtual void setModelMtxParams(const SceneObj& secObj);
  virtual void setModelMtxParams(double angleX, double angleY, double angleZ,
                                 rch::Vec3 pos);

  // Getters:
  virtual XMMATRIX getModelMtx() const;
  virtual XMMATRIX getInvModelMtx() const;
  virtual rch::Vec3 getCenterPos() const { return m_pos; };
  virtual double getAngleX() const { return m_angleX; };
  virtual double getAngleY() const { return m_angleY; };
  virtual double getAngleZ() const { return m_angleZ; };
  virtual Vec3 getScaling() const { return m_scaling; };

 protected:
  const mini::DxDevice& m_dev;

  rch::Vec3 m_pos;
  rch::Vec3 m_scaling{1.0f, 1.0f, 1.0f};
  bool m_prsChanged = false;
  double m_angleX, m_angleY, m_angleZ;
};

}