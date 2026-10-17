#ifndef EMITTER_H
#define EMITTER_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "instructions.h"
#include "tensor.h"

namespace emitter {

class MetaDataInfo {
public:
  int64 addr;
  std::vector<int64> shape;
  std::string dtype;
  std::string value = ""; // Optional, for constants

  MetaDataInfo(int addr, const std::vector<int64> &shape, const std::string &dtype,
               const std::string &value = "")
      : addr(addr), shape(shape), dtype(dtype), value(value) {
  }

  MetaDataInfo(nlohmann::json &j);
  std::string str() const;
};

class MetaData {
public:
  std::string module_name;
  // The kernel's path, as the compile was given it
  std::string input_file;

  int64 max_hbm_size;
  std::vector<MetaDataInfo> input_info;
  std::vector<MetaDataInfo> output_info;
  std::vector<MetaDataInfo> constant_info;
  // Why the metadata cannot be read, empty where it can
  std::string why;

  MetaData(std::ifstream &metadata_path,
           const std::unordered_map<std::string, Tensor *> &tensor_map,
           const std::unordered_map<Tensor *, std::string> &known_constants);
};

// Write a candidate's assembly, or set why it cannot
bool assembly_dump(char *outpath, const std::vector<Instruction *> &instructions,
                   const MetaData &metadata, std::string &why);

} // namespace emitter

#endif // EMITTER_H
