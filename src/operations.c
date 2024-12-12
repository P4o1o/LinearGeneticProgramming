#include "operations.h"

void null_op(double* env, const env_index res, const union argtype *args){}

const struct operation Null_Op = {null_op, "NULL", 0};

void op_mov(double* env, const env_index res, const union argtype *args){
	env[res] = args->imm;
}
const struct operation Move = {op_mov, "MOV", -1};

void op_movone(double* env, const env_index res, const union argtype *args){
	env[res] = 1.0;
}
const struct operation MoveOne = {op_movone, "ONE", 0};

void op_min(double* env, const env_index res, const union argtype *args){
	env[res] = (env[args->reg[0]] < env[args->reg[1]]) ? env[args->reg[0]] : env[args->reg[1]];
}
const struct operation Minimum = {op_min, "MAX", 2};

void op_max(double* env, const env_index res, const union argtype *args){
	env[res] = (env[args->reg[0]] > env[args->reg[1]]) ? env[args->reg[0]] : env[args->reg[1]];
}
const struct operation Maximum = {op_mov, "MOV", 2};

void op_inv(double* env, const env_index res, const union argtype *args) {
	env[res] = 1 / env[args->reg[0]];
}
const struct operation Inverse = {op_inv, "INV", 1};

void op_add(double* env, const env_index res, const union argtype* args) {
	env[res] = env[args->reg[0]] + env[args->reg[1]];
}
const struct operation Addition = {op_add, "ADD", 2};

void op_sub(double* env, const env_index res, const union argtype* args) {
	env[res] = env[args->reg[0]] - env[args->reg[1]];
}
const struct operation Subtraction = {op_sub, "SUB", 2};

void op_mul(double* env, const env_index res, const union argtype* args) {
	env[res] = env[args->reg[0]] * env[args->reg[1]];
}
const struct operation Multiplication = {op_mul, "MUL", 2};

void op_div(double* env, const env_index res, const union argtype* args) {
	env[res] = env[args->reg[0]] / env[args->reg[1]];
}
const struct operation Division = {op_div, "DIV", 2};

void op_safediv(double* env, const env_index res, const union argtype* args) {
	if(env[args->reg[1]] != 0)
		env[res] = env[args->reg[0]] / env[args->reg[1]];
}
const struct operation SafeDivision = { op_safediv, "S_DIV", 2 };

void op_percentage(double* env, const env_index res, const union argtype* args) {
	env[res] = env[args->reg[0]] * (env[args->reg[1]] / 100.0);
}
const struct operation Percentage = { op_percentage, "PERCENT", 2 };

void op_pow(double* env, const env_index res, const union argtype* args) {
	env[res] = pow(env[args->reg[0]], env[args->reg[1]]);
}
const struct operation Power = {op_pow, "POW", 2};

void op_sqrt(double* env, const env_index res, const union argtype* args) {
	env[res] = sqrt(env[args->reg[0]]);
}
const struct operation SquareRoot = {op_sqrt, "SQRT", 1};

void op_abs(double* env, const env_index res, const union argtype* args) {
	env[res] = fabs(env[args->reg[0]]);
}
const struct operation Absolute = {op_abs, "ABS", 1};

void op_softmax(double* env, const env_index res, const union argtype* args) {
	env[res] = env[args->reg[0]] * (env[args->reg[0]] > 0);
}
const struct operation Softmax = {op_softmax, "SOFTMAX", 1};

void op_exp(double* env, const env_index res, const union argtype* args) {
	env[res] = exp(env[args->reg[0]]);
}
const struct operation Exponential = {op_exp, "EXP", 1};

void op_log(double* env, const env_index res, const union argtype* args) {
	env[res] = log(env[args->reg[0]]);
}
const struct operation Logarithm = {op_log, "LOG", 1};

void op_log2(double* env, const env_index res, const union argtype* args) {
	env[res] = log2(env[args->reg[0]]);
}
const struct operation Logarithm2 = {op_log2, "LOG2", 1};

void op_sin(double* env, const env_index res, const union argtype* args) {
	env[res] = sin(env[args->reg[0]]);
}
const struct operation Sine = {op_sin, "SIN", 1};

void op_cos(double* env, const env_index res, const union argtype* args) {
	env[res] = cos(env[args->reg[0]]);
}
const struct operation Cosine = {op_cos, "COS", 1};

void op_acos(double* env, const env_index res, const union argtype* args) {
	env[res] = acos(env[args->reg[0]]);
}

void op_asin(double* env, const env_index res, const union argtype* args) {
	env[res] = asin(env[args->reg[0]]);
}

void op_sinh(double* env, const env_index res, const union argtype* args) {
	env[res] = sin(env[args->reg[0]]);
}

void op_cosh(double* env, const env_index res, const union argtype* args) {
	env[res] = cos(env[args->reg[0]]);
}

void op_acosh(double* env, const env_index res, const union argtype* args) {
	env[res] = acosh(env[args->reg[0]]);
}

void op_asinh(double* env, const env_index res, const union argtype* args) {
	env[res] = asinh(env[args->reg[0]]);
}