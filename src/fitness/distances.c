#include "distances.h"
#include <math.h>


double *euclidean_distances(const struct LGPInput *const in){
    double *res = malloc(sizeof(double) * in->input_num * in->input_num);
    if(res == NULL){
        MALLOC_FAIL_THREADSAFE(sizeof(double) * in->input_num * in->input_num);
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        for(uint64_t j = 0; j < in->input_num; j++){
            if(i == j){
                res[i * in->input_num + j] = 0.0;
            }else if(i > j){
                res[i * in->input_num + j] = res[j * in->input_num + i];
            }else{
                double sum = 0.0;
                for(uint64_t k = 0; k < in->rom_size; k++){
                    double diff = in->memory[(in->rom_size + in->res_size) * i + k].f64 - in->memory[(in->rom_size + in->res_size) * j + k].f64;
                    sum += diff * diff;
                }
                res[i * in->input_num + j] = sqrt(sum);
            }
        }
    }
    return res;
}

double *manhattan_distances(const struct LGPInput *const in){
    double *res = malloc(sizeof(double) * in->input_num * in->input_num);
    if(res == NULL){
        MALLOC_FAIL_THREADSAFE(sizeof(double) * in->input_num * in->input_num);
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        for(uint64_t j = 0; j < in->input_num; j++){
            if(i == j){
                res[i * in->input_num + j] = 0.0;
            }else if(i > j){
                res[i * in->input_num + j] = res[j * in->input_num + i];
            }else{
                double sum = 0.0;
                for(uint64_t k = 0; k < in->rom_size; k++){
                    double diff = fabs(in->memory[(in->rom_size + in->res_size) * i + k].f64 - in->memory[(in->rom_size + in->res_size) * j + k].f64);
                    sum += diff;
                }
                res[i * in->input_num + j] = sum;
            }
        }
    }
    return res;
}

double *chebyshev_distances(const struct LGPInput *const in){
    double *res = malloc(sizeof(double) * in->input_num * in->input_num);
    if(res == NULL){
        MALLOC_FAIL_THREADSAFE(sizeof(double) * in->input_num * in->input_num);
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        for(uint64_t j = 0; j < in->input_num; j++){
            if(i == j){
                res[i * in->input_num + j] = 0.0;
            }else if(i > j){
                res[i * in->input_num + j] = res[j * in->input_num + i];
            }else{
                double max_diff = 0.0;
                for(uint64_t k = 0; k < in->rom_size; k++){
                    double diff = fabs(in->memory[(in->rom_size + in->res_size) * i + k].f64 - in->memory[(in->rom_size + in->res_size) * j + k].f64);
                    if(diff > max_diff) max_diff = diff;
                }
                res[i * in->input_num + j] = max_diff;
            }
        }
    }
    return res;
}

double *cosine_distances(const struct LGPInput *const in){
    double *res = malloc(sizeof(double) * in->input_num * in->input_num);
    if(res == NULL){
        MALLOC_FAIL_THREADSAFE(sizeof(double) * in->input_num * in->input_num);
    }
    for(uint64_t i = 0; i < in->input_num; i++){
        for(uint64_t j = 0; j < in->input_num; j++){
            if(i == j){
                res[i * in->input_num + j] = 0.0;
            }else if(i > j){
                res[i * in->input_num + j] = res[j * in->input_num + i];
            }else{
                double dot_product = 0.0;
                double norm_i = 0.0;
                double norm_j = 0.0;
                
                for(uint64_t k = 0; k < in->rom_size; k++){
                    double val_i = in->memory[(in->rom_size + in->res_size) * i + k].f64;
                    double val_j = in->memory[(in->rom_size + in->res_size) * j + k].f64;
                    dot_product += val_i * val_j;
                    norm_i += val_i * val_i;
                    norm_j += val_j * val_j;
                }
                
                norm_i = sqrt(norm_i);
                norm_j = sqrt(norm_j);
                
                if(norm_i > 0.0 && norm_j > 0.0){
                    double cosine_sim = dot_product / (norm_i * norm_j);
                    // Clamp to [-1, 1] to handle floating point errors
                    cosine_sim = fmax(-1.0, fmin(1.0, cosine_sim));
                    res[i * in->input_num + j] = 1.0 - cosine_sim; // Convert similarity to distance
                }else{
                    res[i * in->input_num + j] = 1.0; // Maximum distance if one vector is zero
                }
            }
        }
    }
    return res;
}