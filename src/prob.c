#include "prob.h"

const uint64_t NUMBER_OF_THREADS = NUMBER_OF_OMP_THREADS;


void random_init_wrapper(uint32_t seed, uint32_t thread_num) {
    random_init(seed, thread_num);
}
 
uint32_t random_wrapper() {
    return get_MT19937(&random_engines[RANDOM_ENGINE_INDEX]);
}

void random_init_all(uint32_t seed) {
    random_init(seed, 0);
	for(uint64_t i = 1; i < NUMBER_OF_OMP_THREADS; i++){
		uint32_t seed = random();
		random_init(seed, i);
	}
}