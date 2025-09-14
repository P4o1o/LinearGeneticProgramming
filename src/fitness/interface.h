#ifndef FITNESS_INTERFACE_H_INCLUDED
#define FITNESS_INTERFACE_H_INCLUDED

#include "../genetics.h"

union FitnessFactor{
	const double threshold; // used in threshold_accuracy
	const double alpha; // used in length_penalized_mse and clock_penalized_mse and conditional_value_at_risk
	const double beta; // used in f_beta_score
	const double delta; // used in huber_loss
	const double quantile; // used in pinball_loss
	const double tolerance; // used in binary_cross_entropy
	const double sigma; // used in gaussian_log_likelyhood
	const double *perturbation_vector; // used in adversarial_perturbation_sensivity
	union {
		const uint64_t num_clusters;    // for clustering metrics that need k
		const double fuzziness;         // for fuzzy clustering metrics
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
		// For silhouette, calinski-harabasz, davies-bouldin, inertia
		struct {
			double *centroids;     // k * dim matrix of cluster centers  
			double *distances;     // temporary distance storage
			uint64_t *assignments; // cluster assignments per point
			uint64_t k;           // number of clusters
			uint64_t dim;         // feature dimensions
		} general;
		// For dunn index (min/max distance tracking)
		struct {
			double min_inter_cluster;
			double max_intra_cluster;
			uint64_t *assignments;
			uint64_t k;
		} dunn;
		// For adjusted rand index
		struct {
			uint64_t *confusion_matrix;  // k x k confusion matrix
			uint64_t *cluster_counts;    // size k
			uint64_t *true_counts;       // size k  
			uint64_t k;
		} rand_index;
		// For fuzzy clustering
		struct {
			double *memberships;  // n x k membership matrix
			uint64_t n;          // number of points
			uint64_t k;          // number of clusters
		} fuzzy;
	} clustering;
};


typedef union FitnessStepResult (*fitness_step)(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
typedef union FitnessStepResult (*fitness_init_acc)(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
typedef int (*fitness_combine)(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
typedef double (*fitness_finalize)(const union FitnessStepResult * const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);


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

inline double eval_fitness(
    const struct LGPInput *const in,
    const struct Program *const prog,
    const uint64_t max_clock,
    const struct FitnessParams * const params,
    const fitness_step step,
    const fitness_combine combine,
    const fitness_finalize finalize,
    const fitness_init_acc init_acc
){
    ASSERT(prog->size > 0);
    ASSERT(in->ram_size > 0);
    ASSERT(in->input_num > 0);
    struct VirtualMachine vm;
    vm.program = prog->content;
    vm.ram = malloc(sizeof(union Memblock) * in->ram_size);
    if (vm.ram == NULL) {
        MALLOC_FAIL_THREADSAFE(sizeof(union Memblock) * in->ram_size);
    }
    uint64_t result_size = params->end - params->start;
    ASSERT(result_size <= in->ram_size);
    union FitnessStepResult accumulator = init_acc(in->input_num, result_size, params);
    for(uint64_t i = 0; i < in->input_num; i++){
        memset(&(vm.core), 0, sizeof(struct Core));
        memset(vm.ram, 0, sizeof(union Memblock) * in->ram_size);
        vm.rom = &(in->memory[(in->rom_size + in->res_size)* i]);
        uint64_t clocks = run_vm(&vm, max_clock);
        union Memblock *result = &vm.ram[params->start];
        union Memblock *actual = &in->memory[(in->rom_size + in->res_size)* i + in->rom_size + params->start];
        union FitnessStepResult step_res = step(result, actual, result_size, params);
        if(! combine(&accumulator, &step_res, params, clocks)){
            break;
        }
    }
    free(vm.ram);
    return finalize(&accumulator, params, result_size, in->input_num, prog->size);
}

#endif