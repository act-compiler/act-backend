#include "emitter.h"

namespace emitter {

MetaDataInfo::MetaDataInfo(nlohmann::json &j) {
  addr = 0;
  if (j.contains("addr") && j["addr"].is_number_integer())
    addr = j["addr"].get<int64>();
  else
    throw std::runtime_error("addr is required in metadata.json");

  shape.clear();
  if (j.contains("shape") && j["shape"].is_array()) {
    for (const auto &s : j["shape"]) {
      if (s.is_number_integer())
        shape.push_back(s.get<int64>());
      else
        throw std::runtime_error("shape must be an array of integers");
    }
  } else
    throw std::runtime_error("shape is required in metadata.json");

  if (j.contains("dtype") && j["dtype"].is_string())
    dtype = j["dtype"].get<std::string>();
  else
    throw std::runtime_error("dtype is required in metadata.json");
}

std::string MetaDataInfo::str() const {
  std::ostringstream oss;
  oss << "{'addr': " << addr << ", 'shape': (";
  for (size_t i = 0; i < shape.size(); i++) {
    oss << shape[i];
    if (i != shape.size() - 1)
      oss << ", ";
  }
  oss << "), 'dtype': " << dtype;
  if (value != "")
    oss << ", 'value': " << value;
  oss << "}";
  return oss.str();
}

MetaData::MetaData(
    std::ifstream &metadata_path,
    const std::unordered_map<std::string, Tensor *> &tensor_map,
    const std::unordered_map<Tensor *, std::string> &known_constants)
    : max_hbm_size(HBM_SIZE) {
  try {
    if (!metadata_path)
      throw std::runtime_error("failed to open metadata.json");

    nlohmann::json j;
    metadata_path.clear();
    metadata_path.seekg(0);
    metadata_path >> j;

    if (j.contains("module_name") && j["module_name"].is_string()) {
      this->module_name = j["module_name"].get<std::string>();
    } else
      throw std::runtime_error("module_name is required in metadata.json");

    if (j.contains("input") && j["input"].is_array()) {
      for (const auto &entry : j["input"]) {
        input_info.emplace_back(
            MetaDataInfo(const_cast<nlohmann::json &>(entry)));
      }
    } else
      throw std::runtime_error("input is required in metadata.json");

    if (j.contains("output") && j["output"].is_array()) {
      if (j["output"].size() != 1)
        throw std::runtime_error("supports only one output in metadata.json");

      for (const auto &entry : j["output"]) {
        output_info.emplace_back(
            MetaDataInfo(const_cast<nlohmann::json &>(entry)));
      }
    } else
      throw std::runtime_error("output is required in metadata.json");
  } catch (const std::exception &e) {
    std::cerr << "Warning: failed to parse metadata.json: " << e.what()
              << std::endl;
    assert(false && "nlohmann json parse error");
  }

  for (const auto &pair : tensor_map) {
    auto *tensor = pair.second;
    if (tensor->type == Tensor::CONSTANT) {
      if (known_constants.find(tensor) == known_constants.end()) {
        std::cerr << "Warning: constant tensor " << tensor->get_name()
                  << " not found in known constants." << std::endl;
        assert(false && "unknown constant tensor");
      }
      if (known_constants.at(tensor) !=
          "reshape[shape='256'](bitcvt(eye[ttype='16,16,I8']()))") {
        std::cerr << "Warning: only constant tensor with value reshape(eye) is "
                     "supported now, continue anyway."
                  << std::endl;
      }

      constant_info.emplace_back(MetaDataInfo(
          tensor->get_offsets()[0]->Min(), tensor->get_sizes(), "jnp.uint8",
          "jnp.reshape(jnp.eye(16, dtype=jnp.int8).astype(jnp.uint8), (256,))"));
    }
  }

  this->max_hbm_size = 0;
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

bool assembly_dump(char *outpath,
                   const std::vector<Instruction *> &instructions,
                   const MetaData &metadata) {
  std::ofstream outfile(outpath);
  if (!outfile) {
    std::cerr << "Warning: failed to open output file: " << outpath
              << "; defaulting to no solution." << std::endl;
    return false;
  }

  std::string indent1 = "    ";
  std::string indent2 = indent1 + indent1;
  std::string indent3 = indent2 + indent1;
  std::string indent4 = indent3 + indent1;

  // Process hlo_name and pii_number from outpath
  // outpath = */<hlo_name>/<pii_number>.py
  std::string path_str(outpath);
  size_t lslash = path_str.find_last_of('/');
  size_t ldot = path_str.find_last_of('.');
  if (lslash == std::string::npos || ldot == std::string::npos ||
      ldot <= lslash + 1) {
    std::cerr
        << "Warning: output path " << outpath
        << " is not in the expected format. '*/<hlo_name>/<pii_number>.py'. "
           "Defaulting to no solution."
        << std::endl;
    return false;
  }
  std::string pii_number = path_str.substr(lslash + 1, ldot - lslash - 1);
  size_t l2slash = path_str.find_last_of('/', lslash - 1);
  if (l2slash == std::string::npos || lslash <= l2slash + 1) {
    std::cerr
        << "Warning: output path " << outpath
        << " is not in the expected format. '*/<hlo_name>/<pii_number>.py'. "
           "Defaulting to no solution."
        << std::endl;
    return false;
  }
  std::string hlo_name = path_str.substr(l2slash + 1, lslash - l2slash - 1);

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

  outfile << indent1 << "@kernel(hbm=" << metadata.max_hbm_size << ","
          << std::endl;

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
