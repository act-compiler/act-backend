#ifndef COMMON_H
#define COMMON_H

#include "ortools/constraint_solver/constraint_solver.h"

typedef int64_t int64;

extern operations_research::Solver solver;

#define MAKE_INT_VAR(a, b) (solver.MakeIntVar((a), (b)))
#define MAKE_INT_KNOWN(a) (solver.MakeIntVar((a), (a)))

#define MAKE_SUM(a, b) (solver.MakeSum((a), (b)))

#define MAKE_EQUALITY(a, b) (solver.MakeEquality((a), (b)))

#endif // COMMON_H
