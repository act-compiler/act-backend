/* A cost model that evaluates nothing */

int act_cost(const char *asm_path, double *cost) {
  (void)asm_path;
  (void)cost;
  return 5;
}
