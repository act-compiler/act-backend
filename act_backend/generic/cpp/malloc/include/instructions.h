#ifndef INSTRUCTIONS_H
#define INSTRUCTIONS_H

#include <string>
#include <utility>
#include <vector>

#include "solver.h"
#include "storage.h"
#include "tensor.h"

using IntExpr = operations_research::IntExpr;
using IntVar = operations_research::IntVar;
using Constraints = std::vector<operations_research::Constraint *>;

class Instruction {
protected:
  std::string op_name;
  Tensor *y;
  std::vector<Tensor *> X;

public:
  Instruction(const std::string &op_name, Tensor *y, const std::vector<Tensor *> &X)
      : op_name(op_name), y(y), X(X) {
    if (y != nullptr)
      y->set_defined_by(this);
    for (auto *x : X) {
      if (x != nullptr)
        x->add_used_by(this);
    }
  }

  virtual ~Instruction() = default;

  virtual std::string str() = 0;

  virtual Tensor *get_lhs() {
    return y;
  }

  virtual std::vector<Tensor *> get_rhs() {
    return X;
  }
  virtual std::vector<bool> get_rhs_inplace() {
    return std::vector<bool>(X.size(), false);
  }

  virtual std::vector<std::pair<IntExpr *, IntExpr *>> get_h(int num) = 0;
  virtual Constraints get_e() = 0;

  virtual std::vector<IntVar *> get_int_var() = 0;
};

// {{INSTRUCTION_CLASSES}}

class Slice : public Instruction {
private:
  // [(start, end), ...] for each dimension of the slice
  std::vector<std::pair<int64, int64>> slices;
  std::vector<int64> rhs_sizes; // size of each dimension of rhs tensor

  std::vector<IntVar *> rhs_addrs;

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h0() {
    std::vector<std::pair<IntExpr *, IntExpr *>> h;
    for (size_t i = 0; i < slices.size(); i++) {
      IntExpr *lhs_addr_start = MAKE_SUM(rhs_addrs[i], slices[i].first);
      IntExpr *lhs_addr_end = MAKE_SUM(rhs_addrs[i], slices[i].second);
      h.push_back({lhs_addr_start, lhs_addr_end});
    }
    return h;
  }

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h1() {
    std::vector<std::pair<IntExpr *, IntExpr *>> h;
    for (size_t i = 0; i < rhs_sizes.size(); i++) {
      IntExpr *rhs_addr_start = rhs_addrs[i];
      IntExpr *rhs_addr_end = MAKE_SUM(rhs_addrs[i], rhs_sizes[i]);
      h.push_back({rhs_addr_start, rhs_addr_end});
    }
    return h;
  }

public:
  // Whether y can be this slice of X, else why not
  static bool check(Tensor *y, const std::vector<Tensor *> &X,
                    const std::vector<std::pair<int64, int64>> &slices, std::string &why) {
    std::string name = y->get_name();
    if (X.size() != 1) {
      why = "a slice takes one operand, and " + name + " has " + std::to_string(X.size());
      return false;
    }
    if (X[0]->get_storage() != y->get_storage()) {
      why = "the slice " + name + " lies in " + y->get_storage()->get_name() +
            ", and its operand " + X[0]->get_name() + " in " + X[0]->get_storage()->get_name();
      return false;
    }
    std::vector<int64> rhs_sizes = X[0]->get_sizes();
    if (slices.size() != rhs_sizes.size()) {
      why = "the slice " + name + " gives " + std::to_string(slices.size()) +
            " ranges, where it takes " + std::to_string(rhs_sizes.size());
      return false;
    }
    for (size_t i = 0; i < rhs_sizes.size(); i++) {
      auto [start, limit] = slices[i];
      std::string range = std::to_string(start) + ":" + std::to_string(limit);
      if (start < 0 || limit > rhs_sizes[i] || start >= limit) {
        why = "the slice " + name + " takes " + range + " of " + X[0]->get_name() + ", which has " +
              std::to_string(rhs_sizes[i]);
        return false;
      }
      if (y->get_sizes()[i] != limit - start) {
        why = "the slice " + name + " has " + std::to_string(y->get_sizes()[i]) +
              ", where its range " + range + " takes " + std::to_string(limit - start);
        return false;
      }
    }
    return true;
  }

  // Built once check holds
  Slice(Tensor *y, std::vector<Tensor *> X, const std::vector<std::pair<int64, int64>> &slices)
      : Instruction("slice", y, X), slices(slices) {
    assert(y != nullptr);
    assert(X[0] != nullptr);

    rhs_sizes = X[0]->get_sizes();
    for (size_t i = 0; i < rhs_sizes.size(); i++) {
      rhs_addrs.push_back(MAKE_INT_VAR(INT64_MIN, INT64_MAX));
    }
  }

  std::string str() override {
    for (auto *var : rhs_addrs) {
      assert(var->Bound());
    }
    return "";
  }

  std::vector<bool> get_rhs_inplace() override {
    return {true};
  }

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h(int num) override {
    switch (num) {
    case 0:
      return get_h0();
    case 1:
      return get_h1();
    default:
      assert(false && "Too many indices for get_h");
    }
  }

  Constraints get_e() override {
    Constraints constraints;
    return constraints;
  }

  std::vector<IntVar *> get_int_var() override {
    return rhs_addrs;
  }
};

class Concat : public Instruction {
private:
  int64 dimension;
  std::vector<int64> rhs1_sizes; // size of each dimension of rhs1 tensor
  std::vector<int64> rhs2_sizes; // size of each dimension of rhs2 tensor

  std::vector<IntVar *> rhs1_addrs;

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h0() {
    std::vector<std::pair<IntExpr *, IntExpr *>> h;
    for (size_t i = 0; i < rhs1_sizes.size(); i++) {
      if (i == (size_t)dimension) {
        IntExpr *lhs_addr_start = rhs1_addrs[i];
        IntExpr *lhs_addr_end = MAKE_SUM(rhs1_addrs[i], rhs1_sizes[i] + rhs2_sizes[i]);
        h.push_back({lhs_addr_start, lhs_addr_end});
      } else {
        IntExpr *lhs_addr_start = rhs1_addrs[i];
        IntExpr *lhs_addr_end = MAKE_SUM(rhs1_addrs[i], rhs1_sizes[i]);
        h.push_back({lhs_addr_start, lhs_addr_end});
      }
    }
    return h;
  }

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h1() {
    std::vector<std::pair<IntExpr *, IntExpr *>> h;
    for (size_t i = 0; i < rhs1_sizes.size(); i++) {
      IntExpr *rhs1_addr_start = rhs1_addrs[i];
      IntExpr *rhs1_addr_end = MAKE_SUM(rhs1_addrs[i], rhs1_sizes[i]);
      h.push_back({rhs1_addr_start, rhs1_addr_end});
    }
    return h;
  }

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h2() {
    std::vector<std::pair<IntExpr *, IntExpr *>> h;
    for (size_t i = 0; i < rhs2_sizes.size(); i++) {
      if (i == (size_t)dimension) {
        IntExpr *rhs2_addr_start = MAKE_SUM(rhs1_addrs[i], rhs1_sizes[i]);
        IntExpr *rhs2_addr_end = MAKE_SUM(rhs1_addrs[i], rhs1_sizes[i] + rhs2_sizes[i]);
        h.push_back({rhs2_addr_start, rhs2_addr_end});
      } else {
        IntExpr *rhs2_addr_start = rhs1_addrs[i];
        IntExpr *rhs2_addr_end = MAKE_SUM(rhs1_addrs[i], rhs2_sizes[i]);
        h.push_back({rhs2_addr_start, rhs2_addr_end});
      }
    }
    return h;
  }

public:
  // Whether y can be this concatenate of X, else why not
  static bool check(Tensor *y, const std::vector<Tensor *> &X, int64 dimension, std::string &why) {
    std::string name = y->get_name();
    if (X.size() != 2) {
      why = "a concatenate takes two operands, and " + name + " has " + std::to_string(X.size());
      return false;
    }
    for (auto *x : X) {
      if (x->get_storage() != y->get_storage()) {
        why = "the concatenate " + name + " lies in " + y->get_storage()->get_name() +
              ", and its operand " + x->get_name() + " in " + x->get_storage()->get_name();
        return false;
      }
    }
    // One storage, so one number of dimensions
    std::vector<int64> sizes = y->get_sizes();
    std::vector<int64> rhs1_sizes = X[0]->get_sizes();
    std::vector<int64> rhs2_sizes = X[1]->get_sizes();
    for (size_t i = 0; i < sizes.size(); i++) {
      bool joined = i == (size_t)dimension;
      if (joined ? sizes[i] != rhs1_sizes[i] + rhs2_sizes[i]
                 : (sizes[i] != rhs1_sizes[i] || rhs1_sizes[i] != rhs2_sizes[i])) {
        why = "the concatenate " + name + " has " + std::to_string(sizes[i]) + " along dimension " +
              std::to_string(i) + ", where its operands give " + std::to_string(rhs1_sizes[i]) +
              " and " + std::to_string(rhs2_sizes[i]);
        return false;
      }
    }
    return true;
  }

  // Built once check holds
  Concat(Tensor *y, std::vector<Tensor *> X, int64 dimension)
      : Instruction("concat", y, X), dimension(dimension) {
    assert(y != nullptr);
    assert(X[0] != nullptr);
    assert(X[1] != nullptr);

    rhs1_sizes = X[0]->get_sizes();
    rhs2_sizes = X[1]->get_sizes();
    for (size_t i = 0; i < rhs1_sizes.size(); i++) {
      rhs1_addrs.push_back(MAKE_INT_VAR(INT64_MIN, INT64_MAX));
    }
  }

  std::string str() override {
    for (auto *var : rhs1_addrs) {
      assert(var->Bound());
    }
    return "";
  }

  std::vector<bool> get_rhs_inplace() override {
    return {true, true};
  }

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h(int num) override {
    switch (num) {
    case 0:
      return get_h0();
    case 1:
      return get_h1();
    case 2:
      return get_h2();
    default:
      assert(false && "Too many indices for get_h");
    }
  }

  Constraints get_e() override {
    Constraints constraints;
    return constraints;
  }

  std::vector<IntVar *> get_int_var() override {
    return rhs1_addrs;
  }
};

#endif // INSTRUCTIONS_H
