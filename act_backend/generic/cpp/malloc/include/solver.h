#ifndef SOLVER_H
#define SOLVER_H

#include <algorithm>
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

// A value's quotient and remainder, rounded down as the language does
inline std::pair<operations_research::IntExpr *, operations_research::IntExpr *>
floor_division(operations_research::IntExpr *a, operations_research::IntExpr *b) {
  if (!b->Bound()) {
    // A divisor of addresses, held at 1 or more as the oracle refuses 0
    solver->AddConstraint(solver->MakeGreaterOrEqual(b, int64_t(1)));
    // The floor: a - r == q * b, 0 <= r < b, the product alone on its side for open domains
    operations_research::IntVar *quotient =
        MAKE_INT_VAR(std::min(a->Min(), int64(0)), std::max(a->Max(), int64(0)));
    operations_research::IntVar *remainder = MAKE_INT_VAR(0, INT64_MAX - 1);
    solver->AddConstraint(solver->MakeLess(remainder, b));
    solver->AddConstraint(
        MAKE_EQUALITY(solver->MakeDifference(a, remainder), MAKE_PROD(quotient, b)));
    return {quotient, remainder};
  }
  int64 divisor = b->Min();
  assert(divisor > 0);
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

// An integer's quotient rounded down, as the language's
inline int64 floor_div(int64 a, int64 b) {
  int64 quotient = a / b;
  return a % b != 0 && (a < 0) != (b < 0) ? quotient - 1 : quotient;
}

// Its remainder, of the divisor's sign as the language's; the least integer by -1 leaves 0
inline int64 floor_mod(int64 a, int64 b) {
  int64 remainder = b == -1 ? 0 : a % b;
  return remainder != 0 && (remainder < 0) != (b < 0) ? remainder + b : remainder;
}

#endif // SOLVER_H
