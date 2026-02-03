#include "triton/Tools/PluginUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Signals.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_env_vars(pybind11::module &m);
void init_triton_ir(pybind11::module &&m);
void init_triton_llvm(pybind11::module &&m);
void init_triton_interpreter(pybind11::module &&m);
void init_triton_passes(pybind11::module &&m);
void init_triton_stacktrace_hook(pybind11::module &m);
void init_gluon_ir(pybind11::module &&m);
void init_linear_layout(pybind11::module &&m);
void init_native_specialize(pybind11::module &m);

PYBIND11_MODULE(libtriton, m) {
  m.doc() = "Python bindings to the C++ Triton API";
  init_triton_stacktrace_hook(m);
  init_triton_env_vars(m);
  init_native_specialize(m);
  init_triton_ir(m.def_submodule("ir"));
  init_triton_passes(m.def_submodule("passes"));
  init_triton_interpreter(m.def_submodule("interpreter"));
  init_triton_llvm(m.def_submodule("llvm"));
  init_linear_layout(m.def_submodule("linear_layout"));
  init_gluon_ir(m.def_submodule("gluon_ir"));

  for (auto TP : loadPlugins()) {
    std::vector<const char *> backendNames;
    if (auto result = TP.getBackendHandles(backendNames); !result)
      llvm::report_fatal_error(result.takeError());
    for (const char *backendName : backendNames) {
      auto result = TP.initializeBackend(backendName);
      if (!result)
        llvm::report_fatal_error(result.takeError());
    }
  }
}
