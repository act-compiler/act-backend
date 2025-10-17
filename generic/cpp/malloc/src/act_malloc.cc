#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "act_malloc.h"

extern "C" int act_malloc(char *pii_path, char *metadata_path, char *asm_path) {
  std::ifstream pii_file(pii_path);
  if (!pii_file) {
    std::cerr << "Failed to open file: " << pii_path << std::endl;
    return 1;
  }

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

    Instruction *instr = parser::create_instruction(
        parser::parse_line(line), tensor_map, known_offsets, known_constants);
    if (instr)
      instructions.push_back(instr);
  }

  Instruction *root = instructions.back();
  root->get_lhs()->type = Tensor::node_type::OUTPUT;

  std::cout << "Starting Phase 2 Module 4: Topological Ordering Generator..."
            << std::endl;

  auto scheduled =
      scheduler::topological(instructions, root, {{{BUFFER_NAMES}}});
  if (scheduled.size() != instructions.size()) {
    std::cerr << "Warning: scheduled instructions size (" << scheduled.size()
              << ") does not match original instructions size ("
              << instructions.size() << ")." << std::endl;
    return 1;
  }

  std::cout << "Starting Phase 2 Module 5: Constraint Satisfaction Problem "
               "Generator..."
            << std::endl;

  constraint::pii_node_constraints(scheduled);
  constraint::def_use_constraints(scheduled);
  constraint::initial_constraints(known_offsets);
  std::vector<operations_research::IntVar *> new_vars;
  constraint::overlap_constraints(scheduled, tensor_map, new_vars);

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

  // Heuristic: order variables by domain width (smallest first) to prioritize
  // tightly constrained variables first
  std::vector<operations_research::IntVar *> ordered_vars = all_vars;
  std::sort(ordered_vars.begin(), ordered_vars.end(),
            [](operations_research::IntVar *a, operations_research::IntVar *b) {
              // domain width may overflow if computed as unsigned; use 128-bit
              // safe path
              int64 w_a = a->Max() - a->Min();
              int64 w_b = b->Max() - b->Min();
              if (w_a != w_b)
                return w_a < w_b;
              return a->Min() < b->Min();
            });
  std::cout << "Starting Phase 2 Module X: Google OR-Tools CP-SAT solver..."
            << std::endl;

  auto decision_builder = solver.MakePhase(
      ordered_vars, operations_research::Solver::CHOOSE_FIRST_UNBOUND,
      operations_research::Solver::ASSIGN_MIN_VALUE);

  // Add a conservative time limit (0.5 seconds) to prevent long hangs
  auto time_limit = solver.MakeTimeLimit(absl::Milliseconds(500));
  solver.NewSearch(decision_builder, time_limit);
  if (solver.NextSolution()) {
    std::cout << "Found a solution!" << std::endl;

    std::cout << "Starting Phase 2 Module 6: Code Emitter..." << std::endl;

    std::ifstream metadata_file(metadata_path);
    if (!metadata_file) {
      std::cerr << "Failed to open file: " << metadata_path << std::endl;
      return 1;
    }
    emitter::MetaData metadata(metadata_file, tensor_map, known_constants);

    if (emitter::assembly_dump(asm_path, scheduled, metadata)) {
      std::cout << "New ASM candidate generated: " << asm_path << std::endl
                << std::endl;
      solver.EndSearch();
      return 0;
    }

    std::cout << "No ASM candidate generated." << std::endl << std::endl;
    solver.EndSearch();
    return 1;

  } else {
    std::cout << "No solution found." << std::endl;

    std::cout << "Starting Phase 2 Module 6: Code Emitter..." << std::endl;
    std::cout << "No ASM candidate generated." << std::endl << std::endl;

    solver.EndSearch();
    return 1;
  }
}
