#include "MultiClassTsetlinMachine.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <string.h>

#define NUMBER_OF_EXAMPLES 20000
#define NUMBER_OF_EXAMPLES_TEST 1000

/*gcc -Wall -O3 -ffast-math -o additionDemo additionDemo.c MultiClassTsetlinMachine.c TsetlinMachine.c*/

int X_train[NUMBER_OF_EXAMPLES][LITERALS];
int y_train[NUMBER_OF_EXAMPLES];

int X_test[NUMBER_OF_EXAMPLES][LITERALS];
int y_test[NUMBER_OF_EXAMPLES];

int X_test_trainbased[NUMBER_OF_EXAMPLES][LITERALS];
int y_test_trainbased[NUMBER_OF_EXAMPLES];

int X_test_holdout[10][LITERALS];
int y_test_holdout[10];

int singledata[1][LITERALS];


void read_file(void)
{
    printf("starting to Read...");
	FILE * fp;

	char * line = NULL;
	size_t len = 0;

	const char *s = " ";
	char *token = NULL;

	fp = fopen("AdditionTrainingData.txt", "r");
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
    printf("Done Reading Training...");
    printf("starting to Read Testing...");
	fp = fopen("AdditionTestingData_all.txt", "r");
	if (fp == NULL) {
		printf("Error opening\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < NUMBER_OF_EXAMPLES_TEST; i++) {
		getline(&line, &len, fp);

		token = strtok(line, s);
		for (int j = 0; j < FEATURES; j++) {
			X_test[i][j] = atoi(token);
			X_test[i][j + FEATURES] = 1 - X_test[i][j];

			token=strtok(NULL,s);
		}
		y_test[i] = atoi(token);
	}

    fp = fopen("AdditionTestingData_trainbased.txt", "r");
	if (fp == NULL) {
		printf("Error opening\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < NUMBER_OF_EXAMPLES_TEST; i++) {
		getline(&line, &len, fp);

		token = strtok(line, s);
		for (int j = 0; j < FEATURES; j++) {
			X_test_trainbased[i][j] = atoi(token);
			X_test_trainbased[i][j + FEATURES] = 1 - X_test_trainbased[i][j];

			token=strtok(NULL,s);
		}
		y_test_trainbased[i] = atoi(token);
	}

    fp = fopen("AdditionTestingData_limited.txt", "r");
	if (fp == NULL) {
		printf("Error opening\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < 10; i++) {
		getline(&line, &len, fp);

		token = strtok(line, s);
		for (int j = 0; j < FEATURES; j++) {
			X_test_holdout[i][j] = atoi(token);
			X_test_holdout[i][j + FEATURES] = 1 - X_test_holdout[i][j];

			token=strtok(NULL,s);
		}
		y_test_holdout[i] = atoi(token);
	}
}

void GetSlicedSingle(int A[][LITERALS], int B[1][LITERALS], int sliced_mrow)
{
    int row, col;
    for(row=0; row < NUMBER_OF_EXAMPLES_TEST; row++)
    {
        for (col=0; col < LITERALS; col++)
        {
            if (row==sliced_mrow) B[0][col]=A[row][col];
        }
    }
}


int write_to_file(const char *content)
{
    const char *filename = "basic_parameters_check.txt";

    FILE *output_file = fopen(filename, "a");  // open for appending

    if (output_file != NULL) {
        fputs(content, output_file);  // write content
        fclose(output_file);

        printf("Content successfully written to %s\n", filename);
        return 0;  // success
    } else {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        return 1;  // error
    }
}


int main(void)
{	
    printf("Here for Addition...");
	srand(time(NULL));

	read_file();
    struct MultiClassTsetlinMachine *mc_tsetlin_machine = CreateMultiClassTsetlinMachine();

	float accuracy_sum = 0.0;
    float trainbased_accuracy_sum = 0.0;
    float limited_accuracy_sum =0.0;
    printf("Init mc TM");
    printf("starting...");
	for (int i = 0; i < 21; i++) {
        printf("Episode %d\n",i);
        fflush(stdout);
		mc_tm_initialize(mc_tsetlin_machine);
		clock_t start_total = clock();
        float s= 2.1;
        int trainepochs = 100;
		mc_tm_fit(mc_tsetlin_machine, X_train, y_train, NUMBER_OF_EXAMPLES, trainepochs, s);
		clock_t end_total = clock();
		double time_used = ((double) (end_total - start_total)) / CLOCKS_PER_SEC;

		printf("RUN %d TIME: %f\n", i+1, time_used);
		float accuracy = mc_tm_evaluate(mc_tsetlin_machine, X_test, y_test, NUMBER_OF_EXAMPLES_TEST);
		accuracy_sum += accuracy;
        float trainaccuracy = mc_tm_evaluate(mc_tsetlin_machine, X_test_trainbased, y_test_trainbased, NUMBER_OF_EXAMPLES_TEST);
		trainbased_accuracy_sum += trainaccuracy;
        float limitedaccuracy = mc_tm_evaluate(mc_tsetlin_machine, X_test_holdout, y_test_holdout, 10);
		limited_accuracy_sum += limitedaccuracy;

        if (i%10 == 0){
    		printf("Accuracy on All Test: %f\n", accuracy);
    		printf("Average accuracy on All Test: %f\n", accuracy_sum/(i+1));
    
            printf("Accuracy on Train-based Test: %f\n", trainaccuracy);
    		printf("Average accuracy on Train-based Test: %f\n", trainbased_accuracy_sum/(i+1));
    
            printf("Accuracy on Holdout-based Test: %f\n", limitedaccuracy);
    		printf("Average accuracy on HoldOut-based Test: %f\n", limited_accuracy_sum/(i+1));
    
            
    		printf("\n");
        }

        if (i==0){
            char buffer[752];

        snprintf(buffer, sizeof(buffer),"\nParamters: CLASSES:%d, CLAUSES:%d, T:%d, s:%f, Training epochs:%d, [LEAF_FACTORS]:%d,[ROOT_FACTORS]:%d,[INTERIOR_ALTERNATIVES]:%d,[INTERIOR_FACTORS]:%d,[LEAF_ALTERNATIVES]:%d,[LITERALS_PER_GROUP]:%d\nEpisode 0 :Accuracy overall:%f,Accuracy train_based:%f, Accuracy holdout:%f",CLASSES,CLAUSES,THRESHOLD,s,trainepochs, LEAF_FACTORS,ROOT_FACTORS,INTERIOR_ALTERNATIVES,INTERIOR_FACTORS,LEAF_ALTERNATIVES,LITERALS_PER_GROUP, accuracy, trainaccuracy, limitedaccuracy);
        write_to_file(buffer);}
        if (i==20){
            char buffer[752];

        snprintf(buffer, sizeof(buffer),"\nParamters: CLASSES:%d, CLAUSES:%d, T:%d, s:%f, Training epochs:%d, [LEAF_FACTORS]:%d,[ROOT_FACTORS]:%d,[INTERIOR_ALTERNATIVES]:%d,[INTERIOR_FACTORS]:%d,[LEAF_ALTERNATIVES]:%d,[LITERALS_PER_GROUP]:%d\nEpisode 20 :Accuracy overall:%f,Accuracy train_based:%f, Accuracy holdout:%f",CLASSES,CLAUSES,THRESHOLD,s,trainepochs, LEAF_FACTORS,ROOT_FACTORS,INTERIOR_ALTERNATIVES,INTERIOR_FACTORS,LEAF_ALTERNATIVES,LITERALS_PER_GROUP, accuracy, trainaccuracy, limitedaccuracy);
        write_to_file(buffer);}
        /*int singledataindex = 4;
        '''GetSlicedSingle(X_test,singledata,singledataindex);'''
        '''int singley[1] = {y_test[singledataindex]};'''
        '''int result = mc_tm_evaluatesingle(mc_tsetlin_machine, singledata, singley);'''*/
            
	}

	return 0;

}
