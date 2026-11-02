#include "solver.h"
#include "storage.h"

std::unique_ptr<operations_research::Solver> solver;

std::map<std::string, Storage *> g_storage;

std::map<std::string, Storage *> build_storages() {
  return {
      {"HBM", new Storage("HBM", {HBM_SIZE}, 0)},
      // {{GLOBALS_DATA_MODELS}}
  };
}
