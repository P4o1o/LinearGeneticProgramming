#include "float_psb2.h"

void save_problem_data(char*filename, struct genetic_input problem){
	FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Failed to open file %s\n", filename);
        exit(-1);
    }
	for (size_t i = 0; i < problem.input_size; i++) {
        fwrite(problem.data[i].x, sizeof(double), problem.x_len, fp);
        fwrite(&(problem.data[i].y), sizeof(double), 1, fp);
    }
    fclose(fp);
}

struct single_input* load_data(const char* filename, size_t x_len, size_t in_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Failed to open file %s\n", filename);
        exit(-1);
    }
    
    struct single_input* data = malloc(in_size * sizeof(struct single_input));
    
    for (size_t i = 0; i < in_size; i++) {
        data[i].x = malloc(x_len * sizeof(double));
        fread(data[i].x, sizeof(double), x_len, fp);
        fread(&data[i].y, sizeof(double), 1, fp);
    }
    
    fclose(fp);
    return data;
}

struct genetic_input vector_distance(const struct genetic_env *genv, const length_t vector_len, const length_t instances) {

	struct genetic_input in;
	in.input_size = instances;
	in.x_len = vector_len * 2;
	in.genv = *genv;
	random_seed_init();
	in.data = malloc(in.input_size * sizeof(struct single_input));
	for (length_t i = 0; i < in.input_size; i++) {
		in.data[i].x = malloc(sizeof(double) * in.x_len);
		in.data[i].y = 0.0;
		for (length_t j = 0; j < vector_len; j++) {
			in.data[i].x[j] = RAND_DBL_BOUNDS(-100.0, 100.0);
			in.data[i].x[vector_len + j] = RAND_DBL_BOUNDS(-100.0, 100.0);
			double diff = in.data[i].x[j] - in.data[i].x[vector_len + j];
			in.data[i].y += diff * diff;
		}
		in.data[i].y = sqrt(in.data[i].y);
	}
	return in;
}

struct genetic_input bouncing_balls(const struct genetic_env *genv, const length_t instances) {

	struct genetic_input in;
	in.input_size = instances;
	in.x_len = 3;
	in.genv = *genv;
	random_seed_init();
	in.data = malloc(in.input_size * sizeof(struct single_input));
	for (length_t i = 0; i < in.input_size; i++) {
		in.data[i].x = malloc(sizeof(double) * in.x_len);
		in.data[i].x[0] = RAND_DBL_BOUNDS(1.01, 100.0);
		in.data[i].x[1] = RAND_DBL_BOUNDS(1.0, in.data[i].x[0] - 0.01);
		length_t rimbalzi = RAND_BOUNDS(1, 20);
		in.data[i].x[2] = (double)rimbalzi;
		in.data[i].y = in.data[i].x[0];
		double r = in.data[i].x[1] / in.data[i].x[0];
		for (length_t j = 1; j < rimbalzi; j++) {
			in.data[i].y += 2.0 * in.data[i].x[0] * pow(r, j);
		}
	}
	return in;
}

struct genetic_input dice_game(const struct genetic_env *genv, const length_t instances) {
	struct genetic_input in;
	in.input_size = instances;
	in.x_len = 2;
	in.genv = *genv;
	random_seed_init();
	in.data = malloc(in.input_size * sizeof(struct single_input));
	for (length_t i = 0; i < in.input_size; i++) {
		in.data[i].x = malloc(sizeof(double) * in.x_len);
		in.data[i].x[0] = (double) RAND_BOUNDS(1, 1000);
		in.data[i].x[1] = (double) RAND_BOUNDS(1, 1000);
		in.data[i].y = (in.data[i].x[0] > in.data[i].x[1]) ? 1.0 - ((in.data[i].x[1] + 1.0) / (2.0 * in.data[i].x[0])) : (in.data[i].x[0] - 1.0) / (2.0 * in.data[i].x[1]); // https://www.karlin.mff.cuni.cz/~nagy/NMSA202/dice1.pdf pg.70
	}
	return in;
}

struct genetic_input shopping_list(const struct genetic_env *genv, env_index num_of_items, const length_t instances) {
	struct genetic_input in;
	in.input_size = instances;
	in.x_len = 2 * num_of_items;
	in.genv = *genv;
	random_seed_init();
	in.data = malloc(in.input_size * sizeof(struct single_input));
	for (length_t i = 0; i < in.input_size; i++) {
		in.data[i].x = malloc(sizeof(double) * in.x_len);
		in.data[i].y = 0;
		for (env_index j = 0; j < num_of_items; j++) {
			in.data[i].x[j] = RAND_DBL_BOUNDS(0.0, 50.0);
			in.data[i].x[num_of_items + j] = RAND_DBL_BOUNDS(0.0, 100.0);
			in.data[i].y += in.data[i].x[j] * (1.0 - (in.data[i].x[num_of_items + j] / 100.0));
		}
	}
	return in;
}

struct genetic_input snow_day(const struct genetic_env *genv, const length_t instances) {
	struct genetic_input in;
	in.input_size = instances;
	in.x_len = 4;
	in.genv = *genv;
	random_seed_init();
	in.data = malloc(in.input_size * sizeof(struct single_input));
	for (length_t i = 0; i < in.input_size; i++) {
		in.data[i].x = malloc(sizeof(double) * in.x_len);
		length_t ore = RAND_BOUNDS(0, 20);
		in.data[i].x[0] = (double)ore;
		in.data[i].x[1] = RAND_DBL_BOUNDS(0.0, 20.0);
		in.data[i].x[2] = RAND_DBL_BOUNDS(0.0, 10.0);
		in.data[i].x[3] = RAND_DBL_BOUNDS(0.0, 1.0);
		in.data[i].y = in.data[i].x[1];
		for (length_t j = 0; j < ore; j++) {
			in.data[i].y += in.data[i].x[2];
			in.data[i].y *= (1.0 - in.data[i].x[3]);
		}
	}
	return in;
}