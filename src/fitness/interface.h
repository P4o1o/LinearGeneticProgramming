#ifndef FITNESS_INTERFACE_H_INCLUDED
#define FITNESS_INTERFACE_H_INCLUDED

#include "../genetics.h"

typedef double *(*point_distances)(const struct LGPInput *const in);

union FitnessFactor{
	const double threshold; // used in threshold_accuracy
	const double alpha; // used in length_penalized_mse and clock_penalized_mse and conditional_value_at_risk
	const double beta; // used in f_beta_score
	const double delta; // used in huber_loss
	const double quantile; // used in pinball_loss
	const double tolerance; // used in binary_cross_entropy
	const double sigma; // used in gaussian_log_likelyhood
	const double *perturbation_vector; // used in adversarial_perturbation_sensivity
	struct {
		const uint64_t num_clusters;    // for clustering metrics that need k
		double *distance_table; // distances table between input points
	} clustering;
};

struct FitnessParams{
	const uint64_t start;
	const uint64_t end;
	union FitnessFactor fact;
};

typedef double (*fitness_fn)(const struct LGPInput *const, const struct Program *const, const uint64_t, const struct FitnessParams *const params);

enum FitnessType{
	MINIMIZE = 0,
	MAXIMIZE = 1,


	FITNESS_TYPE_NUM
};

enum MultiFitnessType{
	LEXIOGRAPHIC = 0,
	PARETO = 1,


	MULTI_FITNESS_TYPE_NUM
};


enum FitnessDataType{
	FITNESS_FLOAT = 0,
    FITNESS_INT = 1,
    FITNESS_SIGN_BIT = 2, // used for binary classification
	FITNESS_PROB = 3 // used for probabilistic outputs
};

union FitnessStepResult{
	double total_f64;
	uint64_t total_u64;
	double *vect_f64;
	struct {
		double *means;
		double *real_vals;
		double ss_res;
		uint64_t count;
	}r_2;
	struct {
		double *sum_x;
		double *sum_y;
		double *sum_xy;
		double *sum_x2;
		double *sum_y2;
	} pearson;
	struct {
		union Memblock *result;
		union Memblock *actual;
		uint64_t len;
	} info;
	struct {
		uint64_t true_pos;
		uint64_t false_pos;
		uint64_t false_neg;
		uint64_t true_neg;
	} confusion;
	union {
		uint64_t *assignments;
		uint64_t single_assignment;
	} clustering;
};


typedef union FitnessStepResult (*fitness_step)(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
typedef union FitnessStepResult (*fitness_init_acc)(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
typedef int (*fitness_combine)(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const uint64_t clocks, const uint64_t input_num, const struct FitnessParams *const params);
typedef double (*fitness_finalize)(const union FitnessStepResult * const result, const struct LGPInput *const in, const uint64_t ressize, const uint64_t prog_size, const uint64_t input_num, const struct FitnessParams *const params);


struct Fitness{
	const fitness_fn fn;
	const fitness_step step;
	const fitness_combine combine;
	const fitness_finalize finalize;
	const fitness_init_acc init_acc;
	const enum FitnessType type;
	enum FitnessDataType data_type;
	const char *name; // name of the fitness function, used for printing
};

struct MultiFitness {
	const struct Fitness *functions; // array di fitness
    const struct FitnessParams *params;  // parametri per ogni fitness
    const uint64_t size;                // numero di fitness
};

double *eval_multifitness(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct MultiFitness * const fitness);

double eval_fitness(
    const struct LGPInput *const in,
    const struct Program *const prog,
    const uint64_t max_clock,
    const struct FitnessParams * const params,
    const fitness_step step,
    const fitness_combine combine,
    const fitness_finalize finalize,
    const fitness_init_acc init_acc
);

void free_distance_table(struct FitnessParams *const params);

#endif