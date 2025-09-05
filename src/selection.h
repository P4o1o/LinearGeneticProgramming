#ifndef SELECTION_H_INCLUDED
#define SELECTION_H_INCLUDED

#include "genetics.h"
#include "fitness.h"

struct FitnessSharingParams{ // parameters for selections based on fitness sharing
    const double alpha;
    const double beta;
    const double sigma;
    union {
        const uint64_t size;
        const double val;
    }select_factor;
};

union SelectionParams{
    const uint64_t size; // size of the tournament, size of the elite for elitism, size of sampling in roulette_selection
    const double val; // percentual_elitism
    const struct FitnessSharingParams fs_params; // fitness_sharing_tournament, fitness_sharing_roulette
};

typedef void (*selection_fn)(struct Population*, const union SelectionParams *const);

#define SELECTION_NUM 4

#define SELECTION_MACRO \
    SELECTION(tournament) \
    SELECTION(fitness_sharing_tournament) \
    SELECTION(elitism) \
    SELECTION(fitness_sharing_elitism) \
    SELECTION(percentual_elitism) \
    SELECTION(fitness_sharing_percentual_elitism) \
    SELECTION(roulette) \
    SELECTION(fitness_sharing_roulette)

#define SELECTION(NAME) \
    void NAME##_MINIMIZE(struct Population* initial, const union SelectionParams *const params); \
    void NAME##_MAXIMIZE(struct Population* initial, const union SelectionParams *const params);
SELECTION_MACRO
#undef SELECTION

struct Selection{
    const selection_fn type[FITNESS_TYPE_NUM];
};

#define SELECTION(NAME) \
    extern const struct Selection NAME;
SELECTION_MACRO
#undef SELECTION

typedef void (*multi_selection_fn)(struct MultiPopulation*, const union SelectionParams *const);

struct MultiSelection{
    const multi_selection_fn function;
};

struct DoubleCouple{
    const double val[2];
};

#endif
