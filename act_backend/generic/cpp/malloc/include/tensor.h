#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>

#include "solver.h"
#include "storage.h"

class Instruction;

class Tensor {
private:
  std::string name;

  Storage *storage;
  std::vector<int64> value;
  std::vector<int64> shape;
  std::vector<int64> sizes;
  std::vector<operations_research::IntVar *> offsets;
  bool held = false;
  Instruction *defined_by;
  std::vector<Instruction *> used_by;

public:
  enum node_type { INPUT, CONSTANT, INTERMEDIATE, OUTPUT };
  node_type type;

  // A value's tensor, held by hold once its storage's shape for it is known
  Tensor(const std::string &name, Storage *storage, const std::vector<int64> &value)
      : name(name), storage(storage), value(value) {
    defined_by = nullptr;
    used_by.resize(0);
  }

  // Held in a shape of its storage's rank, which it fits: its sizes and offsets
  void hold(const std::vector<int64> &extent) {
    assert(!held && extent.size() == storage->get_total_dims());
    held = true;
    shape = extent;
    sizes.reserve(storage->get_addressing_dims());
    offsets.reserve(storage->get_addressing_dims());
    for (size_t i = 0; i < storage->get_addressing_dims(); i++) {
      sizes.push_back(shape[i]);
      // The tensor fits: offset at most capacity - size
      int64 cap = storage->get_capacity(i)->Max();
      int64 max_off = cap - sizes[i];
      offsets.push_back(MAKE_INT_VAR(0, max_off));
    }
  }

  bool is_held() const {
    return held;
  }

  std::string get_name() const {
    return name;
  }
  Storage *get_storage() const {
    return storage;
  }

  // Its value's shape, as its line gives it
  std::vector<int64> get_value() const {
    return value;
  }
  // Its extent along every dimension, the unit ones too
  std::vector<int64> get_shape() const {
    return shape;
  }
  std::vector<int64> get_sizes() const {
    return sizes;
  }
  std::vector<operations_research::IntVar *> get_offsets() const {
    return offsets;
  }

  Instruction *get_defined_by() const {
    return defined_by;
  }

  const std::vector<Instruction *> &get_used_by() const {
    return used_by;
  }

  void set_defined_by(Instruction *instr) {
    assert(defined_by == nullptr && "a tensor is defined by one instruction");
    defined_by = instr;
  }

  void add_used_by(Instruction *instr) {
    used_by.push_back(instr);
  }
};

// Tensors in the order of their names
struct ByName {
  bool operator()(const Tensor *a, const Tensor *b) const {
    return a->get_name() < b->get_name();
  }
};

#endif // TENSOR_H
