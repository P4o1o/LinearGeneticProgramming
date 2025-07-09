#ifndef PSB2_H_INCLUDED
#define PSB2_H_INCLUDED

#include "genetics.h"

struct LGPInput vector_distance(const struct InstructionSet *const instr_set, const uint64_t vector_len, const uint64_t instances);

struct LGPInput bouncing_balls(const struct InstructionSet *const instr_set, const uint64_t instances);

struct LGPInput dice_game(const struct InstructionSet *const instr_set, const uint64_t instances);

struct LGPInput shopping_list(const struct InstructionSet *const instr_set, const uint64_t num_of_items, const uint64_t instances);

struct LGPInput snow_day(const struct InstructionSet *const instr_set, const uint64_t instances);

#endif
