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

namespace mlir::triton::plugin {

extern "C" {

/// A result code for plugin operations; this is used by plugin callbacks to
/// signal success or failure to Triton.
enum Result {
  TP_SUCCESS = 0,
  TP_GENERIC_FAILURE = 1,
};

// Types for plugin callback functions.
typedef Result (*AddPassCallback)(PassManager *);
typedef Result (*RegisterPassCallback)();
typedef Result (*RegisterDialectCallback)(DialectRegistry *);

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
  /// The API version understood by this plugin, usually \c
  /// MLIR_PLUGIN_API_VERSION
  uint32_t apiVersion;
  // TODO: not sure if we want to worry about this yet...

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

  /// Invoke the \c AddPassCallback for each pass registered in this
  /// plugin.
  llvm::Error addPasses(PassManager &passManager) const;

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
  PluginInfo *info;
};

/// Load all plugins specified in the `TRITON_PLUGIN_PATHS` environment
/// variable. This variable should contain a colon-separated list of paths to
/// plugin shared libraries.
///
/// \returns Returns the list of successfully loaded plugins. If any plugin
/// fails to load, it crashes with an LLVM usage error.
const std::vector<TritonPlugin> loadPlugins();

} // namespace mlir::triton::plugin

/// The public entry point for retrieving a Triton plugin.
///
/// When a plugin is loaded by the driver, it will call this entry point to
/// obtain information about this plugin and how to load it. This function needs
/// to be implemented by the plugin.
extern "C" mlir::triton::plugin::PluginInfo *LLVM_ATTRIBUTE_WEAK
tritonGetPluginInfo();
// TODO: needs deallocation API

#endif // TRITON_PLUGIN_UTILS_H
