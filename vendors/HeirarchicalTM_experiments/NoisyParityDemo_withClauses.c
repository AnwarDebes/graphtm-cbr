#include "MultiClassTsetlinMachine.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define NUMBER_OF_EXAMPLES 20000

int X_train[NUMBER_OF_EXAMPLES][LITERALS];
int y_train[NUMBER_OF_EXAMPLES];

int X_test[NUMBER_OF_EXAMPLES][LITERALS];
int y_test[NUMBER_OF_EXAMPLES];

void read_file(void)
{
	FILE * fp;
	char * line = NULL;
	size_t len = 0;

	const char *s = " ";
	char *token = NULL;

	fp = fopen("NoisyParityTrainingData.txt", "r");
	if (fp == NULL) {
		printf("Error opening\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < NUMBER_OF_EXAMPLES; i++) {
		getline(&line, &len, fp);

		token = strtok(line, s);
		for (int j = 0; j < FEATURES; j++) {
			X_train[i][j] = atoi(token);
			X_train[i][j + FEATURES] = 1 - X_train[i][j];
			token=strtok(NULL,s);
		}
		y_train[i] = atoi(token);
	}
	fclose(fp);

	fp = fopen("NoisyParityTestingData.txt", "r");
	if (fp == NULL) {
		printf("Error opening\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < NUMBER_OF_EXAMPLES; i++) {
		getline(&line, &len, fp);

		token = strtok(line, s);
		for (int j = 0; j < FEATURES; j++) {
			X_test[i][j] = atoi(token);
			X_test[i][j + FEATURES] = 1 - X_test[i][j];

			token=strtok(NULL,s);
		}
		y_test[i] = atoi(token);
	}
	fclose(fp);
}


static inline int action(int state)
{
	return state > NUMBER_OF_STATES;
}

void print_clauses(struct TsetlinMachine *tm)
{
	for (int i = 0; i < CLAUSES; i++) {
		printf("Clause %d:\n", i);

		for (int j = 0; j < ROOT_FACTORS; j++) {
			printf("  (");

			for (int k = 0; k < INTERIOR_ALTERNATIVES; k++) {
				if (k > 0) {
					printf(" ∨ ");
				}
				printf("(");

				for (int l = 0; l < INTERIOR_FACTORS; l++) {
					if (l > 0) {
						printf(" ∧ ");
					}
					printf("(");

					for (int m = 0; m < LEAF_ALTERNATIVES; m++) {
						if (m > 0) {
							printf(" ∨ ");
						}

						int first_lit = 1;
						for (int n = 0; n < LEAF_FACTORS; n++) {
							int feature = j * INTERIOR_FACTORS * LEAF_FACTORS + l * LEAF_FACTORS + n;
							int state = (*tm).ta_state[i][j][k][l][m][n];

							if (action(state) == 1) {
								if (!first_lit) {
									printf("∧");
								}
								printf("X%d", feature);
								first_lit = 0;
							}

							state = (*tm).ta_state[i][j][k][l][m][n + LEAF_FACTORS];
							if (action(state) == 1) {
								if (!first_lit) {
									printf("∧");
								}
								printf("~X%d", feature);
								first_lit = 0;
							}
						}
					}
					printf(")");
				}
				printf(")");
			}
			printf(")");
			if (j < ROOT_FACTORS - 1) {
				printf(" ∧\n");
			}
		}
		printf("\n");
	}
}

void export_clauses_json(struct MultiClassTsetlinMachine *mc_tm, const char *filename)
{
	FILE *fp = fopen(filename, "w");
	if (fp == NULL) {
		printf("Error opening file for writing\n");
		return;
	}
	
	fprintf(fp, "{");
	for (int class_idx = 0; class_idx < 2; class_idx++) {
		struct TsetlinMachine *tm = mc_tm->tsetlin_machines[class_idx];
		fprintf(fp, "\"class%d\": {", class_idx);
		
		for (int clause = 0; clause < CLAUSES; clause++) {
			fprintf(fp, "\"%d\": {\"type\": \"AND\",\"children\": [", clause);
		
			for (int j = 0; j < ROOT_FACTORS; j++) {
				fprintf(fp, "{\"type\": \"OR\",\"children\": [");
				for (int k = 0; k < INTERIOR_ALTERNATIVES; k++) {
					fprintf(fp, "{\"type\": \"AND\",\"children\": [");
					for (int l = 0; l < INTERIOR_FACTORS; l++) {
						fprintf(fp, "{\"type\": \"OR\",\"children\": [");
						for (int m = 0; m < LEAF_ALTERNATIVES; m++) {
							fprintf(fp, "{\"type\": \"AND\",\"children\": [");
							int positive_includes[LEAF_FACTORS], negative_includes[LEAF_FACTORS], positive_count = 0, negative_count = 0;
							for (int n = 0; n < LEAF_FACTORS; n++) {
								int feature = j * INTERIOR_FACTORS * LEAF_FACTORS + l * LEAF_FACTORS + n;
								int state = (*tm).ta_state[clause][j][k][l][m][n];
								if (action(state) == 1) {
									positive_includes[positive_count++] = feature;
								}

								state = (*tm).ta_state[clause][j][k][l][m][n + LEAF_FACTORS];
								if (action(state) == 1) {
									negative_includes[negative_count++] = feature;
								}
							}

							for (int p = 0; p < positive_count; p++) {
								fprintf(fp, "\"X%d\"%s", positive_includes[p], (p < positive_count - 1 || negative_count > 0) ? "," : "");
							}

							for (int n = 0; n < negative_count; n++) {
								fprintf(fp, "\"~X%d\"%s", negative_includes[n], (n < negative_count - 1) ? "," : "");
							}
							fprintf(fp, "]}%s", (m < LEAF_ALTERNATIVES - 1) ? "," : "");
						}
						fprintf(fp, "]}%s", (l < INTERIOR_FACTORS - 1) ? "," : "");
					}
					fprintf(fp, "]}%s", (k < INTERIOR_ALTERNATIVES - 1) ? "," : "");
				}
				fprintf(fp, "]}%s", (j < ROOT_FACTORS - 1) ? "," : "");
			}
			fprintf(fp, "]}%s", (clause < CLAUSES - 1) ? "," : "");
		}
		fprintf(fp, "}%s", (class_idx < 1) ? "," : "");
	}
	fprintf(fp, "}\n");
	fclose(fp);
}


int main(void)
{	
	srand(time(NULL));

	printf("Creating data...\n");
	// create_data();
	read_file();

	struct MultiClassTsetlinMachine *mc_tsetlin_machine = CreateMultiClassTsetlinMachine();

	float accuracy_sum = 0.0;
	for (int i = 0; i < 1; i++) {
		printf("==================================================\n");
		printf("RUN %d\n", i+1);
		printf("Init TM...\n");

		mc_tm_initialize(mc_tsetlin_machine);

		printf("Training TM...\n");
		clock_t start_total = clock();
		mc_tm_fit(mc_tsetlin_machine, X_train, y_train, NUMBER_OF_EXAMPLES, 1, 32.1);
		clock_t end_total = clock();
		double time_used = ((double) (end_total - start_total)) / CLOCKS_PER_SEC;
		printf("Training Done in: %f sec\n", time_used);

		printf("\nExporting clause trees to JSON...\n");
		export_clauses_json(mc_tsetlin_machine, "clauses.json");

		float accuracy = mc_tm_evaluate(mc_tsetlin_machine, X_test, y_test, NUMBER_OF_EXAMPLES);
		accuracy_sum += accuracy;
		printf("Accuracy: %f\n", accuracy);
		printf("Average accuracy: %f\n", accuracy_sum/(i+1));
		printf("\n==================================================\n");
	}

	return 0;
}
