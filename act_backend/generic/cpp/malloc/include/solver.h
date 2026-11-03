#ifndef SOLVER_H
#define SOLVER_H

#include <cassert>
#include <memory>
#include <utility>

#include "ortools/constraint_solver/constraint_solver.h"

typedef int64_t int64;

// The solver of the candidate being allocated
extern std::unique_ptr<operations_research::Solver> solver;

#define MAKE_INT_VAR(a, b) (solver->MakeIntVar((a), (b)))
#define MAKE_INT_KNOWN(a) (solver->MakeIntVar((a), (a)))

#define MAKE_SUM(a, b) (solver->MakeSum((a), (b)))
#define MAKE_PROD(a, b) (solver->MakeProd((a), (b)))

#define MAKE_EQUALITY(a, b) (solver->MakeEquality((a), (b)))

// A value's quotient and remainder by a known positive integer, rounded down as the language does
inline std::pair<operations_research::IntVar *, operations_research::IntVar *>
floor_division(operations_research::IntExpr *a, operations_research::IntExpr *b) {
  int64 divisor = b->Min();
  assert(b->Bound() && divisor > 0);
  // The quotient's bounds, rounded down as it is
  auto down = [divisor](int64 value) { return value / divisor - (value % divisor < 0); };
  operations_research::IntVar *quotient = MAKE_INT_VAR(down(a->Min()), down(a->Max()));
  // The remainder a variable of its own, so a sum it is in is bounded by its bounds
  operations_research::IntVar *remainder =
      solver->MakeDifference(a, MAKE_PROD(quotient, divisor))->Var();
  solver->AddConstraint(solver->MakeBetweenCt(remainder, 0, divisor - 1));
  return {quotient, remainder};
}

inline operations_research::IntExpr *floor_div(operations_research::IntExpr *a,
                                               operations_research::IntExpr *b) {
  return floor_division(a, b).first;
}

inline operations_research::IntExpr *floor_mod(operations_research::IntExpr *a,
                                               operations_research::IntExpr *b) {
  return floor_division(a, b).second;
}

#endif // SOLVER_H
