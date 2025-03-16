#include "operations.h"

void null_op(struct virtual_env* env, const env_index res, const union argtype *args){}

const struct operation Null_Op = {null_op, "NULL", 0, 1};

void op_mov(struct virtual_env* env, const env_index res, const union argtype *args){
	env->freg[res] = args->imm;
}
const struct operation Move = {op_mov, "MOV", -1, 0};

void op_movz(struct virtual_env* env, const env_index res, const union argtype *args){
	if(env->flag & 1)
		env->freg[res] = env->freg[args->reg[0]];
}
const struct operation MoveZ = {op_movz, "MOV-Z", 1, 0};

void op_movl(struct virtual_env* env, const env_index res, const union argtype *args){
	if((env->flag & 11) == 0)
		env->freg[res] = env->freg[args->reg[0]];;
}
const struct operation MoveL = {op_movl, "MOV-L", 1, 0};

void op_movle(struct virtual_env* env, const env_index res, const union argtype *args){
	if((env->flag & 10) == 0)
		env->freg[res] = env->freg[args->reg[0]];;
}
const struct operation MoveLE = {op_movle, "MOV-LE", 1, 0};

void op_movg(struct virtual_env* env, const env_index res, const union argtype *args){
	if(env->flag & 10)
		env->freg[res] = env->freg[args->reg[0]];
}
const struct operation MoveG = {op_movg, "MOV-G", 1, 0};

void op_movge(struct virtual_env* env, const env_index res, const union argtype *args){
	if(env->flag & 11)
		env->freg[res] = env->freg[args->reg[0]];
}
const struct operation MoveGE = {op_movge, "MOV-GE", 1, 0};

void op_movone(struct virtual_env* env, const env_index res, const union argtype *args){
	env->freg[res] = 1.0;
}
const struct operation MoveOne = {op_movone, "ONE", 0, 0};

void op_min(struct virtual_env* env, const env_index res, const union argtype *args){
	env->freg[res] = (env->freg[args->reg[0]] < env->freg[args->reg[1]]) ? env->freg[args->reg[0]] : env->freg[args->reg[1]];
}
const struct operation Minimum = {op_min, "MAX", 2, 0};

void op_max(struct virtual_env* env, const env_index res, const union argtype *args){
	env->freg[res] = (env->freg[args->reg[0]] > env->freg[args->reg[1]]) ? env->freg[args->reg[0]] : env->freg[args->reg[1]];
}
const struct operation Maximum = {op_mov, "MOV", 2, 0};

void op_inv(struct virtual_env* env, const env_index res, const union argtype *args) {
	env->freg[res] = 1 / env->freg[args->reg[0]];
}
const struct operation Inverse = {op_inv, "INV", 1, 0};

void op_add(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = env->freg[args->reg[0]] + env->freg[args->reg[1]];
}
const struct operation Addition = {op_add, "ADD", 2, 0};

void op_sub(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = env->freg[args->reg[0]] - env->freg[args->reg[1]];
}
const struct operation Subtraction = {op_sub, "SUB", 2, 0};

void op_mul(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = env->freg[args->reg[0]] * env->freg[args->reg[1]];
}
const struct operation Multiplication = {op_mul, "MUL", 2, 0};

void op_div(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = env->freg[args->reg[0]] / env->freg[args->reg[1]];
}
const struct operation Division = {op_div, "DIV", 2, 0};

void op_safediv(struct virtual_env* env, const env_index res, const union argtype* args) {
	if(env->freg[args->reg[1]] != 0)
		env->freg[res] = env->freg[args->reg[0]] / env->freg[args->reg[1]];
}
const struct operation SafeDivision = { op_safediv, "S_DIV", 2, 0};

void op_percentage(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = env->freg[args->reg[0]] * (env->freg[args->reg[1]] / 100.0);
}
const struct operation Percentage = { op_percentage, "PERCENT", 2, 0};

void op_pow(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = pow(env->freg[args->reg[0]], env->freg[args->reg[1]]);
}
const struct operation Power = {op_pow, "POW", 2, 0};

void op_sqrt(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = sqrt(env->freg[args->reg[0]]);
}
const struct operation SquareRoot = {op_sqrt, "SQRT", 1, 0};

void op_abs(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = fabs(env->freg[args->reg[0]]);
}
const struct operation Absolute = {op_abs, "ABS", 1, 0};

void op_softmax(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = env->freg[args->reg[0]] * (env->freg[args->reg[0]] > 0);
}
const struct operation Softmax = {op_softmax, "SOFTMAX", 1, 0};

void op_exp(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = exp(env->freg[args->reg[0]]);
}
const struct operation Exponential = {op_exp, "EXP", 1, 0};

void op_log(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = log(env->freg[args->reg[0]]);
}
const struct operation Logarithm = {op_log, "LOG", 1, 0};

void op_log2(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = log2(env->freg[args->reg[0]]);
}
const struct operation Logarithm2 = {op_log2, "LOG2", 1, 0};

void op_sin(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = sin(env->freg[args->reg[0]]);
}
const struct operation Sine = {op_sin, "SIN", 1, 0};

void op_cos(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = cos(env->freg[args->reg[0]]);
}
const struct operation Cosine = {op_cos, "COS", 1, 0};

void op_acos(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = acos(env->freg[args->reg[0]]);
}
const struct operation ArcCosine = {op_acos, "ARCCOS", 1, 0};

void op_asin(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = asin(env->freg[args->reg[0]]);
}
const struct operation ArcSine = {op_asin, "ARCSIN", 1, 0};

void op_sinh(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = sinh(env->freg[args->reg[0]]);
}
const struct operation SineH = {op_sinh, "SINH", 1, 0};

void op_cosh(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = cosh(env->freg[args->reg[0]]);
}
const struct operation CosineH = {op_cosh, "COSH", 1, 0};

void op_acosh(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = acosh(env->freg[args->reg[0]]);
}
const struct operation ArcCosineH = {op_acosh, "ARCCOSH", 1, 0};

void op_asinh(struct virtual_env* env, const env_index res, const union argtype* args) {
	env->freg[res] = asinh(env->freg[args->reg[0]]);
}
const struct operation ArcSineH = {op_asinh, "ARCSINH", 1, 0};

void op_cmp(struct virtual_env* env, const env_index res, const union argtype* args){
	double cmp = env->freg[args->reg[0]] - env->freg[args->reg[1]];
	env->flag |= (cmp == 0);  // 01 => 0; 10 => +; 00 => -
	env->flag |= ((cmp > 0) << 1);
}
const struct operation Compare = {op_cmp, "CMP", 2, 1};