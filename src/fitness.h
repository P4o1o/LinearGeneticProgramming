#ifndef FITNESS_H_INCLUDED
#define FITNESS_H_INCLUDED

#include "genetics.h"

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

union FitnessFactor{
	const double threshold; // used in threshold_accuracy
	const double alpha; // used in length_penalized_mse and clock_penalized_mse and conditional_value_at_risk
	const double beta; // used in f_beta_score
	const double delta; // used in huber_loss
	const double quantile; // used in pinball_loss
	const double tolerance; // used in binary_cross_entropy
	const double sigma; // used in gaussian_log_likelyhood
	const double *perturbation_vector; // used in adversarial_perturbation_sensivity
};

struct FitnessParams{
	const uint64_t start;
	const uint64_t end;
	union FitnessFactor fact;
};

typedef double (*fitness_fn)(const struct LGPInput *const, const struct Program *const, const uint64_t, const struct FitnessParams *const params);

double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double rmse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double length_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double clock_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double mae(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double symmetric_mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double logcosh(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double worst_case_error(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double huber_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double r_squared(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double pinball_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double pearson_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double strict_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double binary_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double strict_binary_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double strict_threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double balanced_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double g_mean(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double f1_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double f_beta_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double binary_cross_entropy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double gaussian_log_likelihood(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double brier_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double matthews_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double hinge_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double cohens_kappa(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double adversarial_perturbation_sensitivity(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);
double conditional_value_at_risk(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const struct FitnessParams *const params);


enum FitnessType{
	MINIMIZE = 0,
	MAXIMIZE = 1,


	FITNESS_TYPE_NUM
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
};


typedef union FitnessStepResult (*fitness_step)(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
typedef union FitnessStepResult (*fitness_init_acc)(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
typedef int (*fitness_combine)(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
typedef double (*fitness_finalize)(const union FitnessStepResult * const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);

// INIT_ACC FUNCTION PROTOTYPES
union FitnessStepResult init_acc_i64(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
union FitnessStepResult init_acc_f64(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
union FitnessStepResult init_acc_confusion(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
union FitnessStepResult init_acc_r_2(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
union FitnessStepResult pearson_init_acc(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);
union FitnessStepResult vect_f64_init_acc(const uint64_t inputnum, const uint64_t ressize, const struct FitnessParams *const params);

// STEP FUNCTION PROTOTYPES
union FitnessStepResult quadratic_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult absolute_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult absolute_percent_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult return_info(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult symmetric_absolute_percent_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult logcosh_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult huber_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult pinball_error(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult exact_match(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult binary_sign_match(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult threshold_match(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult binary_classification_confusion(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult cross_entropy_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult gaussian_likelihood_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult brier_score_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);
union FitnessStepResult hinge_loss_step(const union Memblock *const result, const union Memblock *const actual, const uint64_t len, const struct FitnessParams *const params);

// COMBINE FUNCTION PROTOTYPES
int sum_float(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int sum_float_clock_pen(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int sum_uint64(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int sum_confusion(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int max_float(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int strict_sample_match(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int r_squared_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);
int pearson_combine(union FitnessStepResult *accumulator, const union FitnessStepResult *const step_result, const struct FitnessParams *const params, const uint64_t clocks);

// FINALIZE FUNCTION PROTOTYPES
double mean_input_and_dim(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double sqrt_mean_input_and_dim(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double percent_mean_input_and_dim(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double mean_input_and_dim_length_pen(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double rate_per_input(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double rate_per_sample(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double confusion_accuracy(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double confusion_f1_score(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double confusion_f_beta_score(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double confusion_balanced_accuracy(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double confusion_g_mean(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double confusion_matthews_correlation(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double confusion_cohens_kappa(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double negative_mean_input_and_dim(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double max_over_ressize(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double r_squared_finalize(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double pearson_finalize(const union FitnessStepResult * const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);
double value_at_risk_finalize(const union FitnessStepResult *const result, const struct FitnessParams *const params, const uint64_t inputnum, const uint64_t ressize, const uint64_t prog_size);

// UTILITY FUNCTION PROTOTYPES
int compare_doubles(const void *a, const void *b);

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

double *eval_multifitness(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, struct MultiFitness * const fitness);

extern const struct Fitness MSE;
extern const struct Fitness RMSE;
extern const struct Fitness LENGTH_PENALIZED_MSE;
extern const struct Fitness CLOCK_PENALIZED_MSE;
extern const struct Fitness MAE;
extern const struct Fitness MAPE;
extern const struct Fitness SYMMETRIC_MAPE;
extern const struct Fitness LOGCOSH;
extern const struct Fitness WORST_CASE_ERROR;
extern const struct Fitness HUBER_LOSS;
extern const struct Fitness R_SQUARED;
extern const struct Fitness PINBALL_LOSS;
extern const struct Fitness PEARSON_CORRELATION;
extern const struct Fitness ACCURACY;
extern const struct Fitness STRICT_ACCURACY;
extern const struct Fitness BINARY_ACCURACY;
extern const struct Fitness STRICT_BINARY_ACCURACY;
extern const struct Fitness THRESHOLD_ACCURACY;
extern const struct Fitness STRICT_THRESHOLD_ACCURACY;
extern const struct Fitness BALANCED_ACCURACY;
extern const struct Fitness G_MEAN;
extern const struct Fitness F1_SCORE;
extern const struct Fitness F_BETA_SCORE;
extern const struct Fitness BINARY_CROSS_ENTROPY;
extern const struct Fitness GAUSSIAN_LOG_LIKELIHOOD;
extern const struct Fitness MATTHEWS_CORRELATION;
extern const struct Fitness BRIER_SCORE;
extern const struct Fitness HINGE_LOSS;
extern const struct Fitness COHENS_KAPPA;
extern const struct Fitness ADVERSARIAL_PERTURBATION_SENSITIVITY;
extern const struct Fitness CONDITIONAL_VALUE_AT_RISK;

#endif
