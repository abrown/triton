// Defines the external and internal interface for Triton plugins.
//
// This is largely meant to follow the plugin pattern outlined in upstream MLIR
// ([DialectPlugin], [PassPlugin]); use those as references for further
// additions.
//
// [DialectPlugin]:
// https://github.com/llvm/llvm-project/blob/80d6e0b8/mlir/include/mlir/Tools/Plugins/DialectPlugin.h
// [PassPlugin]:
// https://github.com/llvm/llvm-project/blob/80d6e0b8/mlir/include/mlir/Tools/Plugins/PassPlugin.h

#ifndef TRITON_PLUGIN_UTILS_H
#define TRITON_PLUGIN_UTILS_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

/// Identifies the API version understood by this plugin.
///
/// This version should be incremented for ABI-breaking changes in the structs
/// below; we check this version when loading a new \c TritonPlugin. See
/// similar: [MLIR_PLUGIN_API_VERSION].
///
/// [MLIR_PLUGIN_API_VERSION]:
/// https://github.com/llvm/llvm-project/blob/80d6e0b8/mlir/include/mlir/Tools/Plugins/PassPlugin.h#L32
#define TRITON_PLUGIN_API_VERSION 1

/// Use this helper macro on the public entry point for a Triton plugin.
#define TRITON_PLUGIN_API extern "C" __attribute__((visibility("default")))

namespace mlir::triton::plugin {
extern "C" {

// Types for plugin callback functions.
typedef void (*AddPassCallback)(PassManager *);
typedef void (*RegisterPassCallback)();
typedef void (*RegisterDialectCallback)(DialectRegistry *);

/// Information provided by a plugin for loading its passes.
typedef struct PassInfo {
  const char *name;
  const char *version;
  AddPassCallback addPass;
  RegisterPassCallback registerPass;
} PassInfo;

/// Information provided by a plugin for loading its dialects.
typedef struct DialectInfo {
  const char *name;
  const char *version;
  RegisterDialectCallback registerDialect;
} DialectInfo;

/// Container for all plugin information; this is returned by the plugin
/// library's public entry point, @ref tritonGetPluginInfo.
typedef struct PluginInfo {
  /// The API version used by this plugin, see \c TRITON_PLUGIN_API_VERSION.
  uint32_t apiVersion;

  /// A meaningful name of the plugin.
  const char *pluginName;
  /// The version of the plugin.
  const char *pluginVersion;

  /// The list of passes.
  PassInfo *passes;
  size_t numPasses;

  /// The list of dialects.
  DialectInfo *dialects;
  size_t numDialects;
  // TODO: do we really want to support multiple dialects per plugin?
} PluginInfo;
}

/// A helper structure for storing information about a pass registered by a
/// plugin.
struct Pass {
  Pass(const char *name, AddPassCallback addPass)
      : name(name), addPass(addPass) {}

  const char *name;
  const AddPassCallback addPass;
};

/// A loaded Triton plugin.
///
/// An instance of this class wraps a loaded dialect plugin and gives access to
/// its interface defined by the \c PluginInfo it exposes.
class TritonPlugin {
public:
  /// Attempts to load a Triton plugin from a given file.
  ///
  /// \returns Returns an error if either the library cannot be found or loaded,
  /// there is no public entry point, or the plugin implements the wrong API
  /// version.
  static llvm::Expected<TritonPlugin> load(const std::string &filename);

  /// Get the filename of the loaded plugin.
  llvm::StringRef getFilename() const { return filename; }

  /// Get the plugin name.
  llvm::StringRef getPluginName() const { return info->pluginName; }

  /// Get the plugin version.
  llvm::StringRef getPluginVersion() const { return info->pluginVersion; }

  /// Get the plugin API version.
  uint32_t getAPIVersion() const { return info->apiVersion; }

  /// List the available passes; this allows us invoke the \c
  /// AddPassCallback while knowing the pass name.
  const llvm::Expected<std::vector<std::shared_ptr<PassInfo>>>
  listPasses() const;

  /// Invoke the \c RegisterPassCallback for each pass registered in this
  /// plugin.
  llvm::Error registerPasses() const;

  /// Invoke the \c RegisterDialectCallback for each dialect registered in this
  /// plugin.
  llvm::Error registerDialects(DialectRegistry &dialectRegistry) const;

private:
  TritonPlugin(const std::string &filename,
               const llvm::sys::DynamicLibrary &library)
      : filename(filename), library(library), info() {}

  std::string filename;
  llvm::sys::DynamicLibrary library;
  std::shared_ptr<PluginInfo> info;
};

/// Load all plugins specified in the `TRITON_PLUGIN_PATHS` environment
/// variable. This variable should contain a colon-separated list of paths to
/// plugin shared libraries.
///
/// \returns Returns the list of successfully loaded plugins. If any plugin
/// fails to load, it crashes with an LLVM usage error.
const std::vector<TritonPlugin> &loadPlugins();

} // namespace mlir::triton::plugin

/// The public entry point for retrieving a Triton plugin.
///
/// When a plugin is loaded by the driver, Triton call this entry point to
/// obtain information about the plugin and how to load it. This function needs
/// to be implemented by the plugin.
///
/// Triton expects this function to return a pointer to a valid \c PluginInfo
/// struct. Though it is likely a static struct, Triton expects plugins to
/// provide a corresponding release function, \c tritonReleasePluginInfo, to
/// free any resources.
extern "C" mlir::triton::plugin::PluginInfo *LLVM_ATTRIBUTE_WEAK
tritonGetPluginInfo();

/// Release the plugin information struct returned by \c tritonGetPluginInfo.
///
/// This function is expected to free any resources associated with the provided
/// \c PluginInfo pointer. Triton will call this function when the plugin is
/// unloaded.
///
/// The most common pattern is to define a static \c PluginInfo struct in the
/// plugin and implement this with a no-op function:
///
/// ```
/// mlir::triton::plugin::PluginInfo *tritonGetPluginInfo() {
///   static mlir::triton::plugin::PluginInfo info = { ... };
///   return &info;
/// }
///
/// void tritonReleasePluginInfo(mlir::triton::plugin::PluginInfo *info) {
///   // No resources to free in this case.
/// }
/// ```
extern "C" void LLVM_ATTRIBUTE_WEAK
tritonReleasePluginInfo(mlir::triton::plugin::PluginInfo *info);

#endif // TRITON_PLUGIN_UTILS_H
