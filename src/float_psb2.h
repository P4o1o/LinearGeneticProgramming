#ifndef FLOAT_PSB2_H_INCLUDED
#define FLOAT_PSB2_H_INCLUDED

#include "genetics.h"
#include <stdio.h>

// FUNCTIONS FOR GENERATE A genetic_input FOR A PSB2 PROBLEM ON FLOATS

struct genetic_input vector_distance(const struct genetic_env *genv, const length_t vector_len, const length_t instances);
/*
Given two n-dimensional vectors of floats, return the Euclidean distance between the two vectors in n-dimensional space.
*/

struct genetic_input bouncing_balls(const struct genetic_env *genv, const length_t instances);
/*
Given a starting height and a height after the first bounce of a dropped ball, calculate the bounciness index (height of first bounce / starting height).
Then, given a number of bounces, use the bounciness index to calculate the total distance that the ball travels across those bounces.
*/

struct genetic_input dice_game(const struct genetic_env *genv, const length_t instances);
/*
Peter has an n sided die and Colin has an m sided die. If they both roll their dice at the same time, return the probability that Peter rolls strictly higher than Colin.
*/

struct genetic_input shopping_list(const struct genetic_env *genv, env_index num_of_items, const length_t instances);
/*
Given a vector of floats representing the prices of various shopping goods and another vector of floats representing the percent discount of each of those goods,
return the total price of the shopping trip after applying the discount to each item.
*/

struct genetic_input snow_day(const struct genetic_env *genv, const length_t instances);
/*
Given an integer representing a number of hours and 3 floats representing how much snow is on the ground, the rate of snow fall, and the proportion of snow melting per hour,
return the amount of snow on the ground after the amount of hours given. Each hour is considered a discrete event of adding snow and then melting, not a continuous process.
*/

#endif