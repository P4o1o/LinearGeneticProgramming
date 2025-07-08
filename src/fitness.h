#ifndef FITNESS_H_INCLUDED
#define FITNESS_H_INCLUDED

#include "genetics.h"

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

union FitnessParams{
	const double threshold; // used in threshold_accuracy
	const double alpha; // used in lenght_penalized_mse and clock_penalized_mse
	const double beta; // used in f_beta_score
	const double delta; // used in huber_loss
	const double quantile; // used in pinball_loss
	const double tollerance; // used in binary_cross_entropy
	const double sigma; // used in gaussian_log_likelyhood
	const double *perturbation_vector; // used in adversarial_perturbation_sensivity
};

typedef double (*fitness_fn)(const struct LGPInput *const, const struct Program *const, const uint64_t, const union FitnessParams *const params);

double mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double rmse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double lenght_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);
double clock_penalized_mse(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);
double mae(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double symmetric_mape(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double logcosh(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double worst_case_error(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double huber_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);
double r_squared(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double pinball_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);
double pearson_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double threshold_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);
double balanced_accuracy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double g_mean(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double f1_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double f_beta_score(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);
double binary_cross_entropy(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);
double gaussian_log_likelihood(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);
double matthews_correlation(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double hinge_loss(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);
double cohens_kappa(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, UNUSED_ATTRIBUTE const union FitnessParams *const params);
double adversarial_perturbation_sensitivity(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);
double conditional_value_at_risk(const struct LGPInput *const in, const struct Program *const prog, const uint64_t max_clock, const union FitnessParams *const params);


enum FitnessType{
	MINIMIZE = 0,
	MAXIMIZE = 1,


	FITNESS_TYPE_NUM
};

struct Fitness{
	const fitness_fn fn;
	const enum FitnessType type;
	const char *name; // name of the fitness function, used for printing
};

extern const struct Fitness MSE;
extern const struct Fitness RMSE;
extern const struct Fitness LENGHT_PENALIZED_MSE;
extern const struct Fitness CLOCK_PENALIZED_MSE;
extern const struct Fitness MAE;
extern const struct Fitness MAPE;
extern const struct Fitness SYMMETRIC_MAPE;
extern const struct Fitness LOGCOSH;
extern const struct Fitness WORST_CASE_ERROR;
extern const struct Fitness HUBER_LOSS;
extern const struct Fitness RSQUARED;
extern const struct Fitness PINBALL_LOSS;
extern const struct Fitness PEARSON_CORRELATION;
extern const struct Fitness ACCURACY;
extern const struct Fitness THRESHOLD_ACCURACY;
extern const struct Fitness BALANCED_ACCURACY;
extern const struct Fitness G_MEAN;
extern const struct Fitness F1_SCORE;
extern const struct Fitness F_BETA_SCORE;
extern const struct Fitness BINARY_CROSS_ENTROPY;
extern const struct Fitness GAUSSIAN_LOG_LIKELIHOOD;
extern const struct Fitness MATTHEWS_CORRELATION;
extern const struct Fitness HINGE_LOSS;
extern const struct Fitness COHENS_KAPPA;
extern const struct Fitness ADVERSARIAL_PERTURBATION_SENSITIVITY;
extern const struct Fitness CONDITIONAL_VALUE_AT_RISK;

#endif
