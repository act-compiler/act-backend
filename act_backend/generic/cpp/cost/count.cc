// The default cost model: the number of instructions in a candidate

#include <fstream>
#include <string>

extern "C" int act_cost(const char *asm_path, double *cost) {
  std::ifstream asm_file(asm_path);
  if (!asm_file) {
    return 1;
  }

  // An instruction is a line that calls the API
  double count = 0;
  std::string line;
  while (std::getline(asm_file, line)) {
    size_t start = line.find_first_not_of(" \t");
    if (start != std::string::npos && line.compare(start, 4, "api.") == 0) {
      count++;
    }
  }

  *cost = count;
  return 0;
}
