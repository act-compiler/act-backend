#include "emitter.h"

#include <algorithm>
#include <cstdlib>
#include <unordered_map>

namespace emitter {

// Each element type's jnp name and width in bytes
static const std::unordered_map<std::string, std::pair<std::string, int64>> ELEMENT_TYPES = {
    {"pred", {"jnp.bool_", 1}},    {"s8", {"jnp.int8", 1}},    {"u8", {"jnp.uint8", 1}},
    {"s16", {"jnp.int16", 2}},     {"u16", {"jnp.uint16", 2}}, {"f16", {"jnp.float16", 2}},
    {"bf16", {"jnp.bfloat16", 2}}, {"s32", {"jnp.int32", 4}},  {"u32", {"jnp.uint32", 4}},
    {"f32", {"jnp.float32", 4}},   {"s64", {"jnp.int64", 8}},  {"u64", {"jnp.uint64", 8}},
    {"f64", {"jnp.float64", 8}},
};

// A value in jnp: its text, element type and shape
struct Value {
  std::string text;
  std::string type;
  std::vector<int64> shape;
};

// An element type's jnp name and width, or set why it has none
static bool element_type(const std::string &type, std::pair<std::string, int64> &named,
                         std::string &why) {
  auto it = ELEMENT_TYPES.find(type);
  if (it == ELEMENT_TYPES.end()) {
    why = "a constant of element type " + type;
    return false;
  }
  named = it->second;
  return true;
}

static bool is_float(const std::string &type) {
  return type == "f16" || type == "bf16" || type == "f32" || type == "f64";
}

// The integers of a list, brackets, braces and spaces aside
static std::vector<int64> integers(const std::string &text) {
  std::vector<int64> list;
  std::string digits;
  for (char c : text + ",") {
    if (c == ',') {
      if (!digits.empty())
        list.push_back(std::stoll(digits));
      digits.clear();
    } else if (std::string("[]{} ").find(c) == std::string::npos) {
      digits += c;
    }
  }
  return list;
}

// A shape as a Python tuple: (), (n,) or (a, b)
static std::string tuple(const std::vector<int64> &shape) {
  std::string text = "(";
  for (size_t i = 0; i < shape.size(); i++)
    text += (i ? ", " : "") + std::to_string(shape[i]);
  return text + (shape.size() == 1 ? ",)" : ")");
}

// A kernel's literal as Python reads it at a type, or set why it has none
static bool literal(const std::string &text, const std::string &type, std::string &written,
                    std::string &why) {
  static const std::unordered_map<std::string, std::string> words = {
      {"true", "True"},     {"false", "False"}, {"inf", "jnp.inf"},
      {"-inf", "-jnp.inf"}, {"nan", "jnp.nan"}, {"-nan", "-jnp.nan"}};
  if (words.count(text)) {
    written = words.at(text);
    return true;
  }
  if (text.empty() || text.find_first_not_of("0123456789.eE+-") != std::string::npos) {
    why = "a constant of value " + text;
    return false;
  }
  // A float's literal as a Python float, so -0 keeps its sign
  written = text;
  if (is_float(type) && text.find_first_of(".eE") == std::string::npos)
    written += ".0";
  return true;
}

// A jnp function of two values of one type
static Value binary(const std::string &function, const std::vector<Value> &args) {
  return {function + "(" + args[0].text + ", " + args[1].text + ")", args[0].type, args[0].shape};
}

// One operator in jnp, from its payload and its operands' values, or set why it has none
static bool translate(const std::string &op, const std::string &payload,
                      const std::vector<Value> &args, Value &out, std::string &why) {
  std::pair<std::string, int64> named;
  if (op == "constant") {
    // The payload is value:type[shape], the value filling the shape
    size_t colon = payload.rfind(':');
    size_t bracket = payload.find('[', colon);
    std::string type = payload.substr(colon + 1, bracket - colon - 1);
    std::vector<int64> shape = integers(payload.substr(bracket + 1, payload.size() - bracket - 2));
    std::string value = payload.substr(0, colon);
    // An array is named by its type and shape alone
    if (value[0] == '{') {
      why = "a " + payload.substr(colon + 1) + " constant given as an array";
      return false;
    }
    std::string written;
    if (!literal(value, type, written, why) || !element_type(type, named, why))
      return false;
    out = {"jnp.full(" + tuple(shape) + ", " + written + ", dtype=" + named.first + ")", type,
           shape};
    return true;
  }
  if (op == "eye") {
    // The payload is rows,columns,type
    size_t comma = payload.rfind(',');
    std::string type = payload.substr(comma + 1);
    std::vector<int64> shape = integers(payload.substr(0, comma));
    std::string columns = shape[0] == shape[1] ? "" : ", " + std::to_string(shape[1]);
    if (!element_type(type, named, why))
      return false;
    out = {"jnp.eye(" + std::to_string(shape[0]) + columns + ", dtype=" + named.first + ")", type,
           shape};
    return true;
  }
  if (op == "bitcvt") {
    // A value's bytes, of a type not u8
    const Value &x = args[0];
    if (x.type == "u8") {
      why = "a constant built by bitcast-convert from u8";
      return false;
    }
    if (!element_type(x.type, named, why))
      return false;
    // A one-byte value's bytes are its convert
    if (named.second == 1) {
      out = {x.text + ".astype(jnp.uint8)", "u8", x.shape};
      return true;
    }
    std::vector<int64> shape = x.shape;
    shape.push_back(named.second);
    out = {"jnp.reshape(jnp.ravel(" + x.text + ").view(jnp.uint8), " + tuple(shape) + ")", "u8",
           shape};
    return true;
  }
  if (op == "reshape") {
    std::vector<int64> shape = integers(payload);
    out = {"jnp.reshape(" + args[0].text + ", " + tuple(shape) + ")", args[0].type, shape};
    return true;
  }
  if (op == "convert") {
    if (!element_type(payload, named, why))
      return false;
    out = {args[0].text + ".astype(" + named.first + ")", payload, args[0].shape};
    return true;
  }
  if (op == "transpose") {
    std::vector<int64> perm = integers(payload), shape;
    for (int64 axis : perm)
      shape.push_back(args[0].shape[axis]);
    out = {"jnp.transpose(" + args[0].text + ", " + tuple(perm) + ")", args[0].type, shape};
    return true;
  }
  if (op == "broadcast") {
    // The payload is shape;dimension map
    size_t semicolon = payload.find(';');
    std::vector<int64> shape = integers(payload.substr(0, semicolon));
    std::vector<int64> map = integers(payload.substr(semicolon + 1));
    // The operand's dimensions where the map puts them, ones elsewhere
    std::vector<int64> spread(shape.size(), 1);
    for (size_t k = 0; k < map.size(); k++)
      spread[map[k]] = args[0].shape[k];
    out = {"jnp.broadcast_to(jnp.reshape(" + args[0].text + ", " + tuple(spread) + "), " +
               tuple(shape) + ")",
           args[0].type, shape};
    return true;
  }
  if (op == "add") {
    out = binary("jnp.add", args);
    return true;
  }
  if (op == "or") {
    out = binary("jnp.bitwise_or", args);
    return true;
  }
  if (op == "xor") {
    out = binary("jnp.bitwise_xor", args);
    return true;
  }
  if (op == "shift_left") {
    out = binary("jnp.left_shift", args);
    return true;
  }
  if (op == "exponential") {
    out = {"jnp.exp(" + args[0].text + ")", args[0].type, args[0].shape};
    return true;
  }
  if (op == "divide") {
    if (is_float(args[0].type)) {
      out = binary("jnp.divide", args);
      return true;
    }
    // An integer quotient rounds toward zero, bracketed whole
    const std::string &a = args[0].text, &b = args[1].text;
    out = {"((" + a + " - jnp.fmod(" + a + ", " + b + ")) // " + b + ")", args[0].type,
           args[0].shape};
    return true;
  }
  if (op == "shift_right_logical") {
    // A signed value shifts as its unsigned bits
    std::string type = args[0].type;
    if (type[0] != 's') {
      out = binary("jnp.right_shift", args);
      return true;
    }
    std::pair<std::string, int64> bits;
    if (!element_type("u" + type.substr(1), bits, why) || !element_type(type, named, why))
      return false;
    out = {"jnp.right_shift(" + args[0].text + ".view(" + bits.first + "), " + args[1].text +
               ".view(" + bits.first + ")).view(" + named.first + ")",
           type, args[0].shape};
    return true;
  }
  if (op == "reduce") {
    // A sum over the payload's dimensions, at the type after them
    size_t colon = payload.rfind(':');
    std::string type = payload.substr(colon + 1);
    std::vector<int64> axes = integers(payload.substr(0, colon)), shape;
    for (size_t k = 0; k < args[0].shape.size(); k++)
      if (std::find(axes.begin(), axes.end(), int64(k)) == axes.end())
        shape.push_back(args[0].shape[k]);
    if (!element_type(type, named, why))
      return false;
    out = {"jnp.sum(" + args[0].text + ", axis=" + tuple(axes) + ", dtype=" + named.first + ")",
           type, shape};
    return true;
  }
  if (op == "dot") {
    // The payload is four dimension lists, then the result type
    size_t colon = payload.rfind(':');
    std::string type = payload.substr(colon + 1);
    std::vector<std::vector<int64>> lists;
    std::stringstream ss(payload.substr(0, colon));
    for (std::string list; std::getline(ss, list, ';');)
      lists.push_back(integers(list));
    lists.resize(4);
    // Batch, contracted and free dimensions each a letter of their own
    std::string lhs, rhs, batch, free;
    std::vector<int64> shape;
    char next = 'n';
    for (int side = 0; side < 2; side++) {
      const Value &x = args[side];
      const std::vector<int64> &batched = lists[side * 2], &contracted = lists[side * 2 + 1];
      std::string &letters = side ? rhs : lhs;
      for (size_t k = 0; k < x.shape.size(); k++) {
        auto b = std::find(batched.begin(), batched.end(), int64(k));
        auto c = std::find(contracted.begin(), contracted.end(), int64(k));
        if (b != batched.end()) {
          letters += char('a' + (b - batched.begin()));
        } else if (c != contracted.end()) {
          letters += char('h' + (c - contracted.begin()));
        } else {
          letters += next;
          free += next++;
          shape.push_back(x.shape[k]);
        }
      }
    }
    // The batch dimensions lead the result
    std::vector<int64> result;
    for (size_t b = 0; b < lists[0].size(); b++) {
      batch += char('a' + b);
      result.push_back(args[0].shape[lists[0][b]]);
    }
    result.insert(result.end(), shape.begin(), shape.end());
    if (!element_type(type, named, why))
      return false;
    out = {"jnp.einsum('" + lhs + "," + rhs + "->" + batch + free + "', " + args[0].text + ", " +
               args[1].text + ", preferred_element_type=" + named.first + ")",
           type, result};
    return true;
  }
  // The kernel's word for the operator
  why = "a constant built by " + (op == "concat" ? std::string("concatenate") : op);
  return false;
}

// One operator of a constant's text from pos, op[name='payload'](operands), or set why it has none
static bool parse(const std::string &expr, size_t &pos, Value &out, std::string &why) {
  size_t end = expr.find_first_of("[(", pos);
  std::string op = expr.substr(pos, end - pos);
  std::string payload;
  pos = end;
  if (expr[pos] == '[') {
    size_t open = expr.find("='", pos) + 2;
    size_t close = expr.find("']", open);
    payload = expr.substr(open, close - open);
    pos = close + 2;
  }
  // The operands, comma-separated
  std::vector<Value> args;
  for (pos++; expr[pos] != ')';) {
    args.emplace_back();
    if (!parse(expr, pos, args.back(), why))
      return false;
    if (expr[pos] == ',')
      pos += 2;
  }
  pos++;
  return translate(op, payload, args, out, why);
}

bool constant_value(const std::string &expr, std::string &value, std::string &why) {
  size_t pos = 0;
  Value parsed;
  if (!parse(expr, pos, parsed, why)) {
    why = "the allocator cannot write " + why;
    return false;
  }
  value = parsed.text;
  return true;
}

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
                   const std::unordered_map<Tensor *, std::string> &constant_values)
    : max_hbm_size(0) {
  nlohmann::json j;
  metadata_path.clear();
  metadata_path.seekg(0);
  try {
    metadata_path >> j;

    if (j.contains("module_name") && j["module_name"].is_string()) {
      this->module_name = j["module_name"].get<std::string>();
    } else
      throw std::runtime_error("module_name is required in metadata.json");
    this->input_file = j["input_file"].get<std::string>();

    if (j.contains("input") && j["input"].is_array()) {
      for (const auto &entry : j["input"]) {
        input_info.emplace_back(MetaDataInfo(const_cast<nlohmann::json &>(entry)));
      }
    } else
      throw std::runtime_error("input is required in metadata.json");

    if (j.contains("output") && j["output"].is_array()) {
      if (j["output"].size() != 1)
        throw std::runtime_error("supports only one output in metadata.json");

      for (const auto &entry : j["output"]) {
        output_info.emplace_back(MetaDataInfo(const_cast<nlohmann::json &>(entry)));
      }
    } else
      throw std::runtime_error("output is required in metadata.json");
  } catch (const std::exception &e) {
    why = std::string("the allocator cannot parse metadata.json: ") + e.what();
    return;
  }

  // Each constant as its bytes in HBM
  for (const auto &pair : tensor_map) {
    auto *tensor = pair.second;
    if (tensor->type == Tensor::CONSTANT) {
      if (constant_values.find(tensor) == constant_values.end()) {
        why = "the constant tensor " + tensor->get_name() + " has no known value";
        return;
      }
      constant_info.emplace_back(MetaDataInfo(tensor->get_offsets()[0]->Min(), tensor->get_sizes(),
                                              "jnp.uint8", constant_values.at(tensor)));
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
                   const MetaData &metadata, std::string &why) {
  std::string indent1 = "    ";
  std::string indent2 = indent1 + indent1;
  std::string indent3 = indent2 + indent1;
  std::string indent4 = indent3 + indent1;

  // pii_number from outpath's stem
  std::string path_str(outpath);
  size_t lslash = path_str.find_last_of('/');
  size_t fname_start = (lslash == std::string::npos) ? 0 : lslash + 1;
  if (fname_start >= path_str.size()) {
    why = std::string("the output path ") + outpath + " has no filename component";
    return false;
  }

  size_t ldot = path_str.find_last_of('.');
  size_t stem_end = path_str.size();
  if (ldot != std::string::npos && ldot > fname_start) {
    stem_end = ldot;
  }

  if (stem_end <= fname_start) {
    why = std::string("the output path ") + outpath + " does not contain a valid filename stem";
    return false;
  }

  std::string pii_number = path_str.substr(fname_start, stem_end - fname_start);

  std::ofstream outfile(outpath);
  if (!outfile) {
    why = std::string("the allocator cannot write ") + outpath;
    return false;
  }

  // Process kernel function name
  std::string kernel_name = metadata.module_name;

  // HEADER
  outfile << "# Input file: " << metadata.input_file << std::endl;
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
