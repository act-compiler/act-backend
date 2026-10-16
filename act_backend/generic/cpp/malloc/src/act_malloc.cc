#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "act_malloc.h"

// The solver's time for one candidate
static const int64 SOLVER_TIME_LIMIT_MS = 500;

// Hand the caller why a candidate gives no assembly
static int no_assembly(char *reason, size_t reason_size, const std::string &why) {
  snprintf(reason, reason_size, "%s", why.c_str());
  return 1;
}

extern "C" int act_malloc(char *pii_path, char *metadata_path, char *asm_path, int verbose,
                          char *reason, size_t reason_size) {
  // The allocator's progress, printed under --verbose alone
  std::ostream discard(nullptr);
  std::ostream &progress = verbose ? std::cout : discard;

  std::ifstream pii_file(pii_path);
  if (!pii_file)
    return no_assembly(reason, reason_size, std::string("the allocator cannot read ") + pii_path);

  std::unordered_map<std::string, Tensor *> tensor_map;
  std::unordered_map<Tensor *, int64> known_offsets;
  std::unordered_map<Tensor *, std::string> known_constants;
  std::vector<Instruction *> instructions;

  std::string line;
  while (std::getline(pii_file, line)) {
    // trim
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty() || line[0] == '#')
      continue;

    // A line the allocator cannot read drops the candidate
    parser::ParsedLine pl;
    std::string why;
    Instruction *instr = nullptr;
    if (parser::parse_line(line, pl, why))
      instr = parser::create_instruction(pl, tensor_map, known_offsets, known_constants, why);
    if (!why.empty())
      return no_assembly(reason, reason_size,
                         "the allocator cannot read the line `" + line + "`: " + why);
    if (instr)
      instructions.push_back(instr);
  }

  // Each constant's value, else the candidate is dropped
  std::unordered_map<Tensor *, std::string> constant_values;
  for (const auto &[tensor, expr] : known_constants) {
    std::string why;
    if (!emitter::constant_value(expr, constant_values[tensor], why))
      return no_assembly(reason, reason_size, why);
  }

  Instruction *root = instructions.back();
  root->get_lhs()->type = Tensor::node_type::OUTPUT;

  progress << "Starting Phase 2 Module 4: Topological Ordering Generator..." << std::endl;

  // {{TOPOLOGICAL_ORDER}}
  if (scheduled.size() != instructions.size())
    return no_assembly(reason, reason_size,
                       "the allocator orders only " + std::to_string(scheduled.size()) +
                           " of its " + std::to_string(instructions.size()) + " instructions");

  progress << "Starting Phase 2 Module 5: Constraint Satisfaction Problem "
              "Generator..."
           << std::endl;

  constraint::pii_node_constraints(scheduled);
  constraint::def_use_constraints(scheduled);
  std::string why;
  if (!constraint::initial_constraints(known_offsets, why))
    return no_assembly(reason, reason_size, why);
  std::vector<operations_research::IntVar *> new_vars;
  if (!constraint::overlap_constraints(scheduled, tensor_map, new_vars, why))
    return no_assembly(reason, reason_size, why);

  std::vector<operations_research::IntVar *> all_vars;
  for (auto &buffer : g_storage) {
    auto dims = buffer.second->get_addressing_dims();
    for (size_t i = 0; i < dims; i++) {
      all_vars.push_back(buffer.second->get_capacity(i));
    }
  }
  for (auto &pair : tensor_map) {
    auto *t = pair.second;
    auto offset_vars = t->get_offsets();
    all_vars.insert(all_vars.end(), offset_vars.begin(), offset_vars.end());
  }
  for (auto *instr : scheduled) {
    auto vars = instr->get_int_var();
    all_vars.insert(all_vars.end(), vars.begin(), vars.end());
  }
  all_vars.insert(all_vars.end(), new_vars.begin(), new_vars.end());

  // Narrowest domains first: the tightest variables lead
  std::vector<operations_research::IntVar *> ordered_vars = all_vars;
  std::sort(ordered_vars.begin(), ordered_vars.end(),
            [](operations_research::IntVar *a, operations_research::IntVar *b) {
              // By domain width, then by lower bound
              int64 w_a = a->Max() - a->Min();
              int64 w_b = b->Max() - b->Min();
              if (w_a != w_b)
                return w_a < w_b;
              return a->Min() < b->Min();
            });
  progress << "Starting Google OR-Tools constraint solver..." << std::endl;

  auto decision_builder =
      solver.MakePhase(ordered_vars, operations_research::Solver::CHOOSE_FIRST_UNBOUND,
                       operations_research::Solver::ASSIGN_MIN_VALUE);

  // A 0.5 s time limit against long hangs
  auto time_limit = solver.MakeTimeLimit(absl::Milliseconds(SOLVER_TIME_LIMIT_MS));
  solver.NewSearch(decision_builder, time_limit);
  if (solver.NextSolution()) {
    progress << "Found a solution!" << std::endl;

    progress << "Starting Phase 2 Module 6: Code Emitter..." << std::endl;

    std::ifstream metadata_file(metadata_path);
    if (!metadata_file) {
      solver.EndSearch();
      return no_assembly(reason, reason_size,
                         std::string("the allocator cannot read ") + metadata_path);
    }
    emitter::MetaData metadata(metadata_file, tensor_map, constant_values);
    if (!metadata.why.empty()) {
      solver.EndSearch();
      return no_assembly(reason, reason_size, metadata.why);
    }

    if (emitter::assembly_dump(asm_path, scheduled, metadata, why)) {
      progress << "New ASM candidate generated: " << asm_path << std::endl << std::endl;
      solver.EndSearch();
      return 0;
    }

    progress << "No ASM candidate generated." << std::endl << std::endl;
    solver.EndSearch();
    return no_assembly(reason, reason_size, why);

  } else {
    progress << "No solution found." << std::endl;

    progress << "No ASM candidate generated." << std::endl << std::endl;

    // Whether the solver's time limit stopped the search
    bool timed_out = time_limit->crossed();
    solver.EndSearch();
    if (timed_out)
      return no_assembly(reason, reason_size,
                         "the solver found no placement within its time limit of " +
                             std::to_string(SOLVER_TIME_LIMIT_MS) + "ms");
    return no_assembly(reason, reason_size, "the solver found no placement of its tensors");
  }
}
