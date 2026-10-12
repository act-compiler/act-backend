/* A cost model that prices nothing */

int act_cost(const char *asm_path, double *cost) {
  (void)asm_path;
  (void)cost;
  return 5;
}
