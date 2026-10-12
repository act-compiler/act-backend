/* A cost model for the tests: it prefers more instructions, the opposite of the default.
   Its first call takes 200 ms, so the search goes on to price the candidates after the first.
   ACT_TEST_COST makes it misbehave: "fail" returns non-zero and "nan" writes no number. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char closed_path[4096];

/* What a wrapper closes at exit, here a mark beside the first candidate */
static void close_model(void) {
  FILE *mark = fopen(closed_path, "w");
  if (mark) {
    fclose(mark);
  }
}

int act_cost(const char *asm_path, double *cost) {
  const char *mode = getenv("ACT_TEST_COST");
  if (mode && strcmp(mode, "fail") == 0) {
    return 3;
  }
  if (mode && strcmp(mode, "nan") == 0) {
    *cost = NAN;
    return 0;
  }

  if (!closed_path[0]) {
    snprintf(closed_path, sizeof closed_path, "%s.closed", asm_path);
    atexit(close_model);
    usleep(200000);
  }

  FILE *asm_file = fopen(asm_path, "r");
  if (!asm_file) {
    return 1;
  }

  /* An instruction is a line that calls the API */
  double count = 0;
  char line[4096];
  while (fgets(line, sizeof line, asm_file)) {
    if (strncmp(line + strspn(line, " \t"), "api.", 4) == 0) {
      count++;
    }
  }
  fclose(asm_file);

  /* The logarithm keeps the order and is a call into libm */
  *cost = -log2(1 + count);
  return 0;
}
