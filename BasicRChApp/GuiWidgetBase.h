#pragma once
#include <string>

#include "imgui.h"
//#include "imgui_impl_dx11.h"
//#include "imgui_impl_win32.h"

namespace rch {

class GuiWidget {
 public:
  GuiWidget() = default;
  virtual ~GuiWidget() = default;

  GuiWidget(const GuiWidget&) = delete;
  GuiWidget& operator=(const GuiWidget&) = delete;

  GuiWidget(GuiWidget&&) = delete;
  GuiWidget& operator=(GuiWidget&&) = delete;

  virtual void update() = 0;
  /* ImGui handles rendering on its own */
};

// PTODO pointer to DataType instead?
template <typename DataType>
class GuiPopup {
 public:
  GuiPopup(const char* label) : m_winLabel(label), m_chosen(){};
  virtual ~GuiPopup() = default;

  GuiPopup(const GuiPopup&) = delete;
  GuiPopup& operator=(const GuiPopup&) = delete;

  GuiPopup(GuiPopup&&) = delete;
  GuiPopup& operator=(GuiPopup&&) = delete;

  virtual void update() = 0;
  // rendering's done by the ImGui

  void open() { m_isOpen = true; }
  void close() { m_isOpen = false; }

  virtual bool getChosen(DataType* result) {
    if (m_choiceReady == false) {
      return false;
    }
    *result = m_chosen;
    m_choiceReady = false;
    return true;
  }

 protected:
  std::string m_winLabel;
  DataType m_chosen;
  ImVec2 m_frameDims;

  bool m_choiceReady = false;
  bool m_isOpen = false;
};

template <>
class GuiPopup<void> {
 public:
  GuiPopup(const char* label) : m_winLabel(label){};
  virtual ~GuiPopup() = default;

  GuiPopup(const GuiPopup&) = delete;
  GuiPopup& operator=(const GuiPopup&) = delete;

  GuiPopup(GuiPopup&&) = delete;
  GuiPopup& operator=(GuiPopup&&) = delete;

  virtual void update() = 0;
  // rendering's done by the ImGui

  void open() { m_isOpen = true; }
  void close() { m_isOpen = false; }

 protected:
  std::string m_winLabel;
  // DataType m_chosen;
  ImVec2 m_frameDims;

  bool m_choiceReady = false;
  bool m_isOpen = false;
};

}  // namespace rch
