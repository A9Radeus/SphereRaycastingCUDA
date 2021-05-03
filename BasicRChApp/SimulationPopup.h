#pragma once
#include <functional>
#include <cmath>

#include "ImGuiWidgetBase.h"
#include "Simulation.h"


namespace rch {

namespace simul_detail {
struct SimulationPrs {
  SpringSimParams springPrs;
  int maxStep = 5000;
  float stepsPerSec = 60.0f;
};
}

class SimulationPopup : public GuiPopup<simul_detail::SimulationPrs> {
 public:
  using Params = simul_detail::SimulationPrs;

  SimulationPopup(const char* label) : GuiPopup(label){};
  virtual ~SimulationPopup() = default;

  virtual void update() {
    if (m_isOpen) {
      ImGui::OpenPopup(m_winLabel.c_str());
    }
    if (ImGui::BeginPopupModal(m_winLabel.c_str())) {
      inputFloatClamped("x(0)", &(m_chosen.springPrs.x0), -10000, 10000, 1);
      inputFloatClamped("v(0)", &(m_chosen.springPrs.v0), -10000, 10000, 0);
      inputFloatClamped("dt", &(m_chosen.springPrs.dt), s_eps, 10000, 0.2f);
      inputFloatClamped("mass", &(m_chosen.springPrs.m), s_eps, 10000, 1);
      inputFloatClamped("damping (k)", &(m_chosen.springPrs.k), -10000, 10000, 1);
      inputFloatClamped("elasticity (c)", &(m_chosen.springPrs.c), 0, 10000, 1);
      inputFloatClamped("steps per second", &(m_chosen.stepsPerSec), 1, 300, 1);
      inputIntClamped("max steps", &(m_chosen.maxStep), 1, 100000, 1);
      
      ImGui::Separator();
      ImGui::Text("w(t): ");
      static float w_A = 0;
      static float w_omega = 0;
      static float w_phi = 0;
      inputFloatClamped("w.A", &w_A, -200, 200, 1);
      inputFloatClamped("w.omega", &w_omega, -200, 200, 1);
      inputFloatClamped("w.phi", &w_phi, -200, 200, 1);
      static rch::ComboBoxWidget comboWidgetW({"Const", "Sgn", "Sin"}, "w(t)");
      comboWidgetW.update();

      ImGui::Separator();
      ImGui::Text("h(t): ");
      static float h_A = 0;
      static float h_omega = 0;
      static float h_phi = 0;
      inputFloatClamped("h.A", &h_A, -200, 200, 1);
      inputFloatClamped("h.omega", &h_omega, -200, 200, 1);
      inputFloatClamped("h.phi", &h_phi, -200, 200, 1);
      static rch::ComboBoxWidget comboWidgetH({"Const", "Sgn", "Sin"}, "h(t)");
      comboWidgetH.update();
      ImGui::Separator();

      if (ImGui::Button("Create")) {
        m_choiceReady = true;
        switch (comboWidgetW.getSelectedIdx()) {
          case 0: {
            m_chosen.springPrs.w_fun = [&](float t) { return w_A; };
            break;
          } 
          case 1: {
            m_chosen.springPrs.w_fun = [&](float t) {
              return copysignf(1.0f, w_A * std::sinf(w_omega * t + w_phi));
            };
            break;
          }
          case 2: {
            m_chosen.springPrs.w_fun = [&](float t) {
              return w_A * std::sinf(w_omega * t + w_phi);
            };
            break;
          }
        }
        switch (comboWidgetH.getSelectedIdx()) {
          case 0: {
            m_chosen.springPrs.h_fun = [&](float t) { return h_A; };
            break;
          }
          case 1: {
            m_chosen.springPrs.h_fun = [&](float t) {
              return copysignf(1.0f, h_A * std::sinf(h_omega * t + h_phi));
            };
            break;
          }
          case 2: {
            m_chosen.springPrs.h_fun = [&](float t) {
              return h_A * std::sinf(h_omega * t + h_phi);
            };
            break;
          }
        }
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if (ImGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
      }

      ImGui::EndPopup();
      m_isOpen = false;
    }
  }

 private:
  static constexpr float s_eps = 0.0000001f;
};

}  // namespace rch