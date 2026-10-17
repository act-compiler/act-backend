#include "emitter.h"

#include <cstdlib>

namespace emitter {

MetaDataInfo::MetaDataInfo(nlohmann::json &j) {
  addr = j["addr"].get<int64>();
  shape = j["shape"].get<std::vector<int64>>();
  dtype = j["dtype"].get<std::string>();
}

std::string MetaDataInfo::str() const {
  std::ostringstream oss;
  oss << "{'addr': " << addr << ", 'shape': (";
  for (size_t i = 0; i < shape.size(); i++) {
    oss << shape[i];
    if (i != shape.size() - 1)
      oss << ", ";
  }
  // A tuple of one ends in a comma
  if (shape.size() == 1)
    oss << ",";
  oss << "), 'dtype': " << dtype;
  if (value != "")
    oss << ", 'value': " << value;
  oss << "}";
  return oss.str();
}

MetaData::MetaData(std::ifstream &metadata_path,
                   const std::unordered_map<std::string, Tensor *> &tensor_map,
                   const std::unordered_map<Tensor *, std::string> &known_constants)
    : max_hbm_size(0) {
  nlohmann::json j;
  metadata_path.clear();
  metadata_path.seekg(0);
  try {
    metadata_path >> j;
  } catch (const nlohmann::json::parse_error &e) {
    // Every candidate reads this one file, so the compile ends
    std::cerr << "Error: failed to parse metadata.json: " << e.what() << std::endl;
    std::exit(1);
  }

  this->module_name = j["module_name"].get<std::string>();
  for (const auto &entry : j["input"]) {
    input_info.emplace_back(MetaDataInfo(const_cast<nlohmann::json &>(entry)));
  }
  for (const auto &entry : j["output"]) {
    output_info.emplace_back(MetaDataInfo(const_cast<nlohmann::json &>(entry)));
  }

  for (const auto &pair : tensor_map) {
    auto *tensor = pair.second;
    if (tensor->type == Tensor::CONSTANT) {
      if (known_constants.at(tensor) != "reshape[shape='256'](bitcvt(eye[ttype='16,16,I8']()))") {
        std::cerr << "Warning: only constant tensor with value reshape(eye) is "
                     "supported now, continue anyway."
                  << std::endl;
      }

      constant_info.emplace_back(
          MetaDataInfo(tensor->get_offsets()[0]->Min(), tensor->get_sizes(), "jnp.uint8",
                       "jnp.reshape(jnp.eye(16, dtype=jnp.int8).astype(jnp.uint8), (256,))"));
    }
  }

  for (const auto &pair : tensor_map) {
    auto *tensor = pair.second;
    if (tensor->get_storage()->get_name() == "HBM") {
      int64 offset = tensor->get_offsets()[0]->Min();
      int64 size = tensor->get_sizes()[0];
      if (offset + size > this->max_hbm_size)
        this->max_hbm_size = offset + size;
    }
  }
}

bool assembly_dump(char *outpath, const std::vector<Instruction *> &instructions,
                   const MetaData &metadata) {
  std::string indent1 = "    ";
  std::string indent2 = indent1 + indent1;
  std::string indent3 = indent2 + indent1;
  std::string indent4 = indent3 + indent1;

  // hlo_name and pii_number from outpath's directory and stem
  std::string path_str(outpath);
  size_t lslash = path_str.find_last_of('/');
  size_t fname_start = (lslash == std::string::npos) ? 0 : lslash + 1;
  size_t ldot = path_str.find_last_of('.');
  size_t stem_end = path_str.size();
  if (ldot != std::string::npos && ldot > fname_start) {
    stem_end = ldot;
  }

  std::string pii_number = path_str.substr(fname_start, stem_end - fname_start);
  std::string hlo_name = metadata.module_name;
  if (lslash != std::string::npos && lslash > 0) {
    size_t l2slash = path_str.find_last_of('/', lslash - 1);
    size_t parent_start = (l2slash == std::string::npos) ? 0 : l2slash + 1;
    if (lslash > parent_start) {
      hlo_name = path_str.substr(parent_start, lslash - parent_start);
    }
  }

  std::ofstream outfile(outpath);
  if (!outfile) {
    std::cerr << "Warning: failed to open output file: " << outpath
              << "; defaulting to no solution." << std::endl;
    return false;
  }

  // Process kernel function name
  std::string kernel_name = metadata.module_name;

  // HEADER
  outfile << "# Input file: " << hlo_name << ".hlo" << std::endl;
  outfile << "# Kernel name: " << kernel_name << std::endl;
  outfile << "# PII number: " << pii_number << std::endl;
  outfile << "# Do not edit!" << std::endl << std::endl;

  outfile << "import jax.numpy as jnp" << std::endl << std::endl << std::endl;

  // Kernel function metadata
  outfile << "def " << kernel_name << "(kernel, api):" << std::endl;

  outfile << indent1 << "@kernel(hbm=" << metadata.max_hbm_size << "," << std::endl;

  outfile << indent3 << "input=[" << std::endl;
  for (const auto &info : metadata.input_info) {
    outfile << indent4 << info.str() << "," << std::endl;
  }
  outfile << indent3 << "]," << std::endl;

  if (!metadata.constant_info.empty()) {
    outfile << indent3 << "constant=[" << std::endl;
    for (const auto &info : metadata.constant_info) {
      outfile << indent4 << info.str() << "," << std::endl;
    }
    outfile << indent3 << "]," << std::endl;
  } else {
    outfile << indent3 << "constant=[]," << std::endl;
  }

  outfile << indent3 << "output=[" << std::endl;
  for (const auto &info : metadata.output_info) {
    outfile << indent4 << info.str() << "," << std::endl;
  }
  outfile << indent3 << "]" << std::endl;

  outfile << indent3 << ")" << std::endl;
  outfile << indent1 << "def " << kernel_name << "_" << "():" << std::endl;

  // Assembly code
  for (auto *instr : instructions) {
    outfile << indent2 << "api." << instr->str() << std::endl;
  }

  outfile << std::endl;
  outfile << indent1 << "return " << kernel_name << "_" << std::endl;

  outfile.close();
  return true;
}

} // namespace emitter
