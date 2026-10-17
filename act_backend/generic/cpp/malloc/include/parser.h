#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <unordered_map>
#include <vector>

#include "instructions.h"
#include "solver.h"
#include "storage.h"
#include "tensor.h"

namespace parser {

struct ParsedLine {
  std::string lhs_variable_name = "";
  std::string lhs_storage_name = "";
  int64 lhs_offset = -1; // -1 means not specified
  std::vector<int64> lhs_shape = {};
  std::string op_name = "";
  std::string op_attr = ""; // raw attribute string inside []
  std::vector<std::string> operand_names = {};
};

// A line's parts, or false and why names the part it lacks
bool parse_line(const std::string &line, ParsedLine &pl, std::string &why);

// A line's instruction, or none; a fault of the line sets why
Instruction *create_instruction(const ParsedLine &pl,
                                std::unordered_map<std::string, Tensor *> &tensor_map,
                                std::unordered_map<Tensor *, int64> &known_offsets,
                                std::unordered_map<Tensor *, std::string> &known_constants,
                                std::string &why);

} // namespace parser

#endif // PARSER_H
