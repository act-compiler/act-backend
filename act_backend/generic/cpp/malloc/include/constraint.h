#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "instructions.h"
#include "solver.h"
#include "tensor.h"

using LiveRange = std::pair<size_t, size_t>;
using InterferenceGraph = std::vector<std::pair<Tensor *, Tensor *>>;

namespace constraint {

// CSP1: e(alpha, beta) and (Is(y), Ie(y)) = h(0)(beta)
void pii_node_constraints(const std::vector<Instruction *> &instructions);

// CSP2: (Is(x_i), Ie(x_i)) = h(i)(beta)
void def_use_constraints(const std::vector<Instruction *> &instructions);

// CSP3: Is(x_i) = const
void initial_constraints(const std::unordered_map<Tensor *, int64> &known_offsets);

// CSP4: NON_OVERLAP(h(0), h'(0))
void overlap_constraints(const std::vector<Instruction *> &instructions,
                         const std::unordered_map<std::string, Tensor *> &tensor_map,
                         std::vector<IntVar *> &new_vars);

} // namespace constraint

/* Helper functions for CSP4 */

LiveRange compute_live_range(Tensor *a, const std::vector<Instruction *> &instructions);
InterferenceGraph
build_interference_graph(const std::unordered_map<Tensor *, LiveRange> &live_ranges);

#endif // CONSTRAINT_H
