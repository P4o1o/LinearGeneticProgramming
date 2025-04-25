#include "prob.h"

void random_seed_init(void) {
    uint64_t epoch = time(NULL);
    uint64_t cpuclock = clock();
	srand(epoch ^ cpuclock);
}