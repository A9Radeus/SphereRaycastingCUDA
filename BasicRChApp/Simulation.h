#pragma once
#include <functional>

namespace rch {

struct SpringSimParams {
  float x0, v0, dt, m, k, c;

  std::function<float(float)> w_fun;
  std::function<float(float)> h_fun;

  SpringSimParams(
      float x0_ = -6.94, float v0_ = 0, float dt_ = 0.01f, float m_ = 1.0f,
      float k_ = 0.1f, float c_ = 5.0f,
      std::function<float(float)>&& w_fun_ = [](float t) { return 0; },
      std::function<float(float)>&& h_fun_ = [](float t) { return 0; })
      : x0(x0_),
        v0(v0_),
        dt(dt_),
        m(m_),
        k(k_),
        c(c_),
        w_fun(w_fun_),
        h_fun(h_fun_){};
};


class SpringSimulation {
 public:
  //using StateType = float;
  struct StateType {
    float f, g, h, w, x, x_t, x_tt;
  };

  //void setParams(SpringSimParams&& params); // PTODO
  void setParams(const SpringSimParams& params) { 
    x0 = params.x0;
    v0 = params.v0;
    dt = params.dt;
    m = params.m;
    k = params.k;
    c = params.c;

    w_fun = params.w_fun;
    h_fun = params.h_fun;

    xCu = x0;
    xPr = x0 - (v0) * dt;
    timeAcc = 0;
  };
  
  //// (!) Doesn't return the actual w_fun and h_fun
  //SpringSimParams getParams() const {
  //  return {x0, v0, dt, m, k, c};
  //};
  //void changeParams(const SpringSimParams& params) {
  //}

  StateType step() {
    StateType state;
    state.h = h_fun(timeAcc); 
    state.w = w_fun(timeAcc);  

    const float fCu = (c * (state.w - xCu));
    const float dt2 = dt * dt;
    
    /********** Version with 21 flops ************************************
    const float xNew = (((fCu) / m) + ((k * xPr) / (2.0f * dt * m)) +
                        (state.h / m) + (((2.0f * xCu) - xPr) / dt2)) *
                       ((2.0f * m * dt2) / ((2.0f * m) + (k * dt)));
    ***********************************************************************/
    
    // 16 flops:
    const float xNew = ((2.0f * dt2 * (state.h + fCu)) + (dt * k * xPr) -
                        (2.0f * m * (xPr - 2.0f * xCu))) /
                       (dt * k + 2 * m);
    
    state.x = xCu;
    state.x_t = (xNew - xPr) / (2.0f * dt);
    state.x_tt = (xNew - (2.0f * xCu) + xPr) / (dt2);
    state.f = fCu;
    state.g = -k * state.x_t;

    xPr = xCu;
    xCu = xNew;
    timeAcc += dt;

    return state;
  }

 private:
  // -- Parameters -- 
  float x0 = 1, v0 = 1; // x(0), v(0)
  float dt = 1; // time delta
  float m = 1;  // mass
  float k = 1, c = 1; // damping & elasticity factors 
  std::function<float(float)> w_fun = [](float t) { return 0.0f; };
  std::function<float(float)> h_fun = [](float t) { return 0.0f; };
  
  // -- State -- 
  float xPr = 0; // previous value i.e.: x(t - delta) 
  float xCu = 0; // current value i.e.: x(t)
  float timeAcc = 0; // current value i.e.: x(t)
};

// Inner algorithm as a template provides additional type-safety
// since AsynchSimulation<A1> may be significantly different from
// AsynchSimulation<A2>.
// In addition, no vtable is required.
template <typename Algo>
class AsynchSimulation {
 public:
  using AStateType = typename Algo::StateType;

  AsynchSimulation(unsigned int maxSteps = 500, float sbs = 0.05f)
      : m_maxSteps(maxSteps), m_sbs(sbs), m_stepState() {};

  template <typename... AlgoParams>
  void start(AlgoParams&&... params) {
    m_core.setParams(std::forward<AlgoParams>(params)...);
    m_running = true;
  }

  void resume()  { m_running = true; }
  void pause()   { m_running = false; }

  void stop() {
    m_running = false;
    m_step = 0;
    m_accTime = 0;
  }

  void restart() {
    m_running = true;
    m_step = 0;
    m_accTime = 0;
  }

  void setMaxSteps(float newMax) {
    assert((m_running == false));
    m_maxSteps = newMax;
  }

  // Sets the sbs to (1/spsTarget)
  void setSpsTarget(float spsTarget) {
    assert((m_running == false));
    m_sbs = (1.0f / spsTarget);
  }

  bool update(const mini::Clock& clock) {
    if (m_running == false) {
      return false;
    }

    // Enforce given "steps per second"
    auto frameDt = static_cast<float>(clock.getFrameTime());
    m_accTime += frameDt;
    if (m_accTime < m_sbs) {
      return false;
    }
    m_accTime = 0;

    m_stepState = m_core.step();

    // Update m_step and m_running
    m_step += 1;
    if (m_step > m_maxSteps) {
      m_running = false;
    }

    return true;
  }

  const AStateType& getState() const {
    return m_stepState;
  } 

 protected:
  unsigned int m_step = 0;
  float m_accTime = 0;
  bool m_running = false;
  unsigned int m_maxSteps;
  float m_sbs;  // = (1 / simulation's "steps per second" target)

  Algo m_core;
  AStateType m_stepState;  // Simulation's current state
};

//class SpringAsynchSimulation : public AsynchSimulation<SpringSimulation> {
//public:
//  using Params = SpringSimParams;
//
//  Params getParamsRef() const { return m_core.getParams(); };
//
//  
//  //void changeParams(const Params& prs); // PTODO 
//
//  void changeParams(Params&& prs){
//    m_core.changeParams(prs)
//  };
//
//private:
//};

}  // namespace rch