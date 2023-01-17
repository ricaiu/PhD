#include <iostream>
#include <bits/stdc++.h>
#include <cstring>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "SFMT.h"
#include <math.h>
#include <omp.h>
#include <cctype>
#include <array>
#include <vector>
#include <random>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

#define BUFFER_SIZE 1000

int main(int argc, char *argv[]) {

/*
argc is the argument counter
argv is the string array containing all the arguments passed after the .out.
argv[0] is ./main.out, argv[argc-1] is the last
Scheme:
./main.out  data_path/parameters.txt simulation test [> stoud (optional)]
*/
  int num_threads = 10;
// test is the test Number
  char* simulation = argv[argc-2];
  char* test = argv[argc-1];
  printf("Starting simulation: %s and test: %s\n", simulation,test);

  /*  loading parameter from parameter.txt  */
  ifstream input_param;
  input_param.open(argv[1]);
  if(!input_param.is_open()){
    cout << "Parameters file not found" << endl;
    exit(1);}
  string parameter_strings[4];
  double parameter_numbers[4];
  for(int i=0; i<4 ; i++){
    getline(input_param,parameter_strings[i]);
  }
  for(int i=0; i<4; i++){
    parameter_numbers[i] = stod(parameter_strings[i]);
  }




  const double sigma = parameter_numbers[0];
  const int q = parameter_numbers[1];
  const int steps = parameter_numbers[2];
  const int termalization = parameter_numbers[3];

  string rounded_sigma = to_string(sigma).substr(0, to_string(sigma).find(".")+3);
  string temperatures_path_name = "./data/sigma_" + rounded_sigma +
                                  "/simulation_" + simulation  + "/";
  string sizes_file_name =  "./data/sigma_" + rounded_sigma +
                            "/simulation_" + simulation +"/Ls_test_"+test+".txt";
  string output_path_name = "./data/sigma_" + rounded_sigma +
                            "/simulation_" + simulation  + "/";

  printf("sigma = %f \n",sigma);
  printf("steps = %d \n",steps);
  printf("q = %d \n",q);
  printf("termalization = %d \n",termalization);
  printf("Temperatures path: %s\n", temperatures_path_name.c_str());
  printf("Sizes file: %s\n", sizes_file_name.c_str());
  printf("output path: %s\n", output_path_name.c_str());
  fflush(stdout);



  FILE  *sizes_file;

  // double* my_buffer;
	// my_buffer = (double *) malloc(sizeof(double)*BUFFER_SIZE);

  int initialization = time(NULL);
  initialization += (50*atoi(test)*atoi(test));
  srand (initialization);
  clock_t now_time, tStart = clock();

  int c;    //this must be an int (we need it to get number of sizes)
  int number_of_L = 0;//number of sizes
  int index_L = 0;//temporary index to store sizes

  double lower = 1/sqrt(2) ; //this is the lower bound of the distances extracted

  double  alpha = -1/sigma; //this is the exponent for the distance extraction

  //for time computing via omp_get_wtime()
  double start; 
  double end;

  //open the file containing the sizes
  sizes_file = fopen(sizes_file_name.c_str(),"r");



  //get the number of sizes
  for (c = getc(sizes_file); c != EOF; c = getc(sizes_file))
    if (c == '\n') // Increment count if this character is newline
      number_of_L = number_of_L + 1;

  printf("Number of sizes: %d\n", number_of_L);

  //reput the FILE puntactor at the start of the file
  rewind(sizes_file);

  //create the size array and put the sizes into it
  int Ls[number_of_L];


  while (fscanf(sizes_file, "%d", &Ls[index_L])!=EOF){
    index_L ++;
  }

  fclose( sizes_file );




  //start the loop over the sizes
  for(int sizes = 0; sizes < number_of_L; sizes ++){

    int tmp, L, number_of_T = 0, index_T = 0;
    string  temperatures_file_name;
    FILE *temperatures_file;
    L = Ls[sizes];


    // open the temperature files
    temperatures_file_name = temperatures_path_name + "L_"+to_string(L)+"/Ts_test_"+test+".txt";
    printf("Temperatures FILE: %s\n", temperatures_file_name.c_str());
    temperatures_file = fopen(temperatures_file_name.c_str(),"r");
    // gets the number of temperatures
    for (tmp = getc(temperatures_file); tmp != EOF; tmp = getc(temperatures_file))
      if (tmp == '\n') // Increment count if this character is newline
        number_of_T = number_of_T + 1;

    printf("Number of temperatures: %d\n", number_of_T);
    //put the puntactor of the file at the start of this one
    rewind(temperatures_file);
    //create a temperatures array in which we store the temperatures
    double Ts[number_of_T];

    while (fscanf(temperatures_file, "%lf", &Ts[index_T])!=EOF){
      index_T ++;
    }
    fclose( temperatures_file );


    // START THE LOOP OVER TEMPERATURES

    //set the number of threads as number_of_T
    num_threads = number_of_T;


    /* defining main random seed */

    /* inizialing seeds for different threads */

    printf("\n> Seed estratti per i diversi threads: \n\n");

    int seeds[num_threads];
    bool seed_check_flag = true;

    while (seed_check_flag){
      for(int i = 0; i<num_threads ; i++){
        seeds[i] = (rand()%100000);
      }
      /* checking diversity of seeds */

      for (int i = 0; i < (num_threads-1); i++ ) {
        for (int j = (i+1); j < num_threads; j++ ) {
          if ( seeds[i] == seeds[j] ) { seed_check_flag = false; }
        }
      }
      if (!seed_check_flag){
         printf("\n> Flag di controllo sui seeds : false \n");
         seed_check_flag = true;
      }
      else{
        /*Print the seeds*/
        for(int i = 0; i<num_threads ; i++){
          if ( (i+1) == 0) { printf("\t %d", seeds[i]); }
          else if ( (i+1) % 4 == 0) { printf("\t %d \n", seeds[i]); }
          else { printf("\t %d", seeds[i]); }
        }
        printf("\n");
        seed_check_flag = false;       
      }
    }
    






    //for(int temperature = 0; temperature < number_of_T; temperature ++)
    tStart = clock();
 
    start = omp_get_wtime(); 
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
      int temperature = omp_get_thread_num();

      //initialize random generator
      sfmt_t sfmt;
      sfmt_init_gen_rand(&sfmt, seeds[temperature]);

      int center_x, center_y, neighs_sum = 0, energy;
      double T,r, unif, theta, magne = 0;
      double filt[7];
      string output_file_name, seed_file_name, configuration_file_name;
      FILE * data_output_file, * seed_file, * configuration_file;
      int lattice[L][L];

      T = Ts[temperature];

      //save in the a file seed.txt the seed

      seed_file_name = output_path_name+"L_"+to_string(L)+"/test_"+test+
                        "/seeds/T_"+to_string(T)+".txt";
      seed_file = fopen(seed_file_name.c_str(), "w");
      fprintf(seed_file,"%d",seeds[temperature]);
      fclose(seed_file);

      // cold start!!
      for (int i = 0; i < L; i++){
         for (int j = 0; j < L; j++){
              lattice[i][j] = 1;
         }
      }



      output_file_name = output_path_name+"L_"+
                          to_string(L)+"/test_"+test+"/T_"+to_string(T)+"_m.bin";
      printf("Output FILE: %s\n", output_file_name.c_str());
      data_output_file = fopen( output_file_name.c_str() ,"wb");

      fflush(stdout);

      filt[0] = 1/(  1+exp(  2*(1/T)*(-3)  )  );
      filt[1] = 1/(  1+exp(  2*(1/T)*(-2)  )  );
      filt[2] = 1/(  1+exp(  2*(1/T)*(-1)  )  );
      filt[3] = 1/(  1+exp(  2*(1/T)*(0)   )  );
      filt[4] = 1/(  1+exp(  2*(1/T)*(1)   )  );
      filt[5] = 1/(  1+exp(  2*(1/T)*(2)   )  );
      filt[6] = 1/(  1+exp(  2*(1/T)*(3)   )  );



    

    //int count = 0;
    //TERMALIZATION
    for (int termo = 0; termo < termalization; termo++) {
      for (int spinflip = 0; spinflip < pow(L,2); spinflip++) {
        neighs_sum = 0;
        center_x = (int)(L*sfmt_genrand_res53(&sfmt))%L;
        center_y = (int)(L*sfmt_genrand_res53(&sfmt))%L;
        for (int j = 0; j < q; j++){
            unif = 1-sfmt_genrand_res53(&sfmt);
            r  = lower*exp(alpha*log(unif));//lower*pow(unif,alpha);

            if (r <= (int) L/2){
              unif = sfmt_genrand_res53(&sfmt);
              theta = unif*(2*M_PI);
              neighs_sum += lattice[(center_x+(int)round(r*cos(theta))+L)%L]
                                  [(center_y+(int)round(r*sin(theta))+L)%L];
            }
            

        }
        energy = lattice[center_x][center_y]*neighs_sum;
        unif = sfmt_genrand_res53(&sfmt);

        if (energy == -3){
          if (unif < filt[0]){
              lattice[center_x][center_y]*=-1;
          }
        }
        else if (energy == -2){
          if (unif < filt[1]){
              lattice[center_x][center_y]*=-1;
          }
        }
        else if (energy == -1){
          if (unif < filt[2]){
              lattice[center_x][center_y]*=-1;
          }
        }
        else if (energy == 0){
          if (unif < filt[3]){
              lattice[center_x][center_y]*=-1;
          }
        }
        else if (energy == 1){
          if (unif < filt[4]){
              lattice[center_x][center_y]*=-1;
          }
        }
        else if (energy == 2){
          if (unif < filt[5]){
              lattice[center_x][center_y]*=-1;
          }
        }
        else if (energy == 3){
          if (unif < filt[6]){
              lattice[center_x][center_y]*=-1;
          }
        }
      }
    }

    for (int step = 0; step < steps; step++){

        for (int spinflip = 0; spinflip < pow(L,2); spinflip++) {
          neighs_sum = 0;
          center_x = (int)(L*sfmt_genrand_res53(&sfmt))%L;
          center_y = (int)(L*sfmt_genrand_res53(&sfmt))%L;
          for (int j = 0; j < q; j++){
              unif = 1-sfmt_genrand_res53(&sfmt);
              r  = lower*exp(alpha*log(unif));//lower*pow(unif,alpha);


              if (r <= (int) L/2){
                unif = sfmt_genrand_res53(&sfmt);
                theta = unif*(2*M_PI);
                neighs_sum += lattice[(center_x+(int)round(r*cos(theta))+L)%L]
                                    [(center_y+(int)round(r*sin(theta))+L)%L];
              }


          }
          energy = lattice[center_x][center_y]*neighs_sum;
          unif = sfmt_genrand_res53(&sfmt);
          
          
          if (energy == -3){
            if (unif < filt[0]){
                lattice[center_x][center_y]*=-1;
            }
          }
          else if (energy == -2){
            if (unif < filt[1]){
                lattice[center_x][center_y]*=-1;
            }
          }
          else if (energy == -1){
            if (unif < filt[2]){
                lattice[center_x][center_y]*=-1;
            }
          }
          else if (energy == 0){
            if (unif < filt[3]){
                lattice[center_x][center_y]*=-1;
            }
          }
          else if (energy == 1){
            if (unif < filt[4]){
                lattice[center_x][center_y]*=-1;
            }
          }
          else if (energy == 2){
            if (unif < filt[5]){
                lattice[center_x][center_y]*=-1;
            }
          }
          else if (energy == 3){
            if (unif < filt[6]){
                lattice[center_x][center_y]*=-1;
            }
          }
        }

        magne = 0;
        for (int k = 0; k < L; k++){
            for (int l = 0; l < L; l++){
                magne += lattice[k][l];
            }
        }

        magne = magne/pow(L,2);

        fwrite(&magne,sizeof(double), 1, data_output_file);


    }

    //free( my_buffer );
    fclose( data_output_file );

    configuration_file_name = output_path_name+"L_"+to_string(L)+"/test_"+test+
                      "/last_configuration/T_"+to_string(T)+".bin";
    configuration_file = fopen(configuration_file_name.c_str(), "wb");
    for (int k = 0; k < L; k++){
      for (int l = 0; l < L; l++){
              fwrite(&lattice[k][l],sizeof(int), 1, configuration_file);
            }
        }
    
    fclose( configuration_file);

    }

    now_time = clock();
    printf("End of simulation for L: %d\n",L);
    printf("Time taken: %.2fs\n", (double)(now_time - tStart)/CLOCKS_PER_SEC);
    end = omp_get_wtime(); 
    printf("Time (per-thread) taken: %.2fs\n", (double)(end - start));
    fflush(stdout);

  }




  return 0;

  // // my_buffer[count] = magne;
  // // count ++;
  // // if(count == BUFFER_SIZE){
  // //     fwrite( my_buffer, sizeof(double), BUFFER_SIZE, data_output_file);
  // //     fflush( data_output_file );
  // //     count = 0;
  // // }

}
