#include "triton/Tools/PluginUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#define DEBUG_TYPE "triton-plugins"

using namespace mlir::triton::plugin;

llvm::Expected<TritonPlugin> TritonPlugin::load(const std::string &filename) {
  std::string error;
  auto library =
      llvm::sys::DynamicLibrary::getPermanentLibrary(filename.c_str(), &error);
  if (!library.isValid())
    return llvm::make_error<llvm::StringError>(
        Twine("Could not load library '") + filename + "': " + error,
        llvm::inconvertibleErrorCode());

  TritonPlugin plugin{filename, library};

  // tritonGetPluginInfo should be resolved to the definition from the
  // plugin we are currently loading.
  intptr_t getInfoFn =
      (intptr_t)library.getAddressOfSymbol("tritonGetPluginInfo");

  if (!getInfoFn)
    return llvm::make_error<llvm::StringError>(
        Twine("Plugin entry point not found in '") + filename,
        llvm::inconvertibleErrorCode());

  plugin.info = reinterpret_cast<decltype(tritonGetPluginInfo) *>(getInfoFn)();

  if (plugin.info->apiVersion != MLIR_PLUGIN_API_VERSION)
    return llvm::make_error<llvm::StringError>(
        Twine("Wrong API version on plugin '") + filename + "'. Got version " +
            Twine(plugin.info->apiVersion) + ", supported version is " +
            Twine(MLIR_PLUGIN_API_VERSION) + ".",
        llvm::inconvertibleErrorCode());

  return plugin;
}

llvm::Error TritonPlugin::addPasses(PassManager &PassManager) const {
  if (!info->passes && info->numPasses > 0)
    return llvm::make_error<llvm::StringError>(
        Twine("Invalid pass pointer in plugin '") + filename + "'.'",
        llvm::inconvertibleErrorCode());
  LLVM_DEBUG(llvm::dbgs() << "Adding " << info->numPasses
                          << " passes for plugin " << info->pluginName << ":"
                          << info->pluginVersion << "\n");

  for (auto i = 0; i < info->numPasses; ++i) {
    const auto &pass = info->passes[i];
    if (pass.addPass) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Adding pass " << pass.name << ":" << pass.version << "\n");
      pass.addPass(&PassManager);
    }
  }
  return llvm::Error::success();
}

llvm::Error TritonPlugin::registerPasses() const {
  if (!info->passes && info->numPasses > 0)
    return llvm::make_error<llvm::StringError>(
        Twine("Invalid pass pointer in plugin '") + filename + "'.'",
        llvm::inconvertibleErrorCode());
  LLVM_DEBUG(llvm::dbgs() << "Registering " << info->numPasses
                          << " passes for plugin " << info->pluginName << ":"
                          << info->pluginVersion << "\n");

  for (auto i = 0; i < info->numPasses; ++i) {
    const auto &pass = info->passes[i];
    if (pass.registerPass) {
      LLVM_DEBUG(llvm::dbgs() << "Registering pass " << pass.name << ":"
                              << pass.version << "\n");
      pass.registerPass();
    }
  }
  return llvm::Error::success();
}

llvm::Error
TritonPlugin::registerDialects(DialectRegistry &dialectRegistry) const {
  if (!info->dialects && info->numDialects > 0)
    return llvm::make_error<llvm::StringError>(
        Twine("Invalid dialect pointer in plugin '") + filename + "'.'",
        llvm::inconvertibleErrorCode());
  LLVM_DEBUG(llvm::dbgs() << "Registering " << info->numDialects
                          << " dialects for plugin " << info->pluginName << ":"
                          << info->pluginVersion << "\n");

  for (auto i = 0; i < info->numDialects; ++i) {
    const auto &dialect = info->dialects[i];
    if (dialect.registerDialect) {
      LLVM_DEBUG(llvm::dbgs() << "Registering dialect " << dialect.name << ":"
                              << dialect.version << "\n");
      dialect.registerDialect(&dialectRegistry);
    }
  }
  return llvm::Error::success();
}

const std::vector<TritonPlugin> mlir::triton::plugin::loadPlugins() {
  std::vector<TritonPlugin> plugins;
  if (const char *env = std::getenv("TRITON_PLUGIN_PATHS")) {
    llvm::SmallVector<llvm::StringRef, 4> paths;
    llvm::StringRef(env).split(paths, ':');
    for (const auto &path : paths) {
      LLVM_DEBUG(llvm::dbgs() << "Loading plugin from path: " << path << "\n");
      auto pluginOrErr = TritonPlugin::load(path.str());
      if (auto err = pluginOrErr.takeError()) {
        llvm::Error wrappedErr = llvm::createStringError(
            llvm::Twine("Failed to load plugin from path: ") + path +
            ". Error: " + llvm::toString(std::move(err)));
        llvm::reportFatalUsageError(std::move(wrappedErr));
      }
      plugins.push_back(std::move(*pluginOrErr));
    }
  }
  return plugins;
}

#undef DEBUG_TYPE
