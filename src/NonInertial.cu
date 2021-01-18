#include <iostream>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>


int N = 100;
float a = 2.0; // diameter
float b = 2.0; // diameter
float k = 10.0;
float mu = 1.0;
float mu_r = 1.0;
float dt = 1.0/300.0;
float Dr = 0.0;
float Dt = 0.0;
float v0 = 1.0;
float L = 1./0.;
float TotalTime = 30.0;
float tau = 0.0;
float alpha = 0.0;
float packing = 0.5;
int StepsBetweenSaves = 30;
int silent = 0;

unsigned long seed = 123456789;

#include "../include/NonInertial.cuh"

void help(){
  std::cout << "Options are: \n";
  std::cout << "-N                                           (int) Number of particles, DEFAULT 100\n";
  std::cout << "-T                                           (float) Max time, DEFAULT = 30.0, [seconds]\n";
  std::cout << "-dt                                          (float) Time step, DEFAULT = 1.0/300.0,[seconds]\n";
  std::cout << "-mur                                         (float) rotational mobility coefficient, DEFAULT = 0.0001\n";
  std::cout << "-a                                           (float) Particle major axis, DEFAULT = 2.0, this defines the length scale\n";
  std::cout << "-b                                           (float) Particle minor axis, DEFAULT = 2.0, this defines the length scale\n";
  std::cout << "Note: (a,b) = (2,2) implies a unit circle\n";
  std::cout << "-k                                           (float) Spring constant, DEFAULT = 10.0\n";
  std::cout << "-mu                                          (float) accel = f(v,x,t) + mu * sum(collision_forces(x,t)), DEFAULT = 1.0\n";
  std::cout << "-Dr                                          (float) rotational diffusion, DEFAULT = 0.0,[rad]^2[s]^-1\n";
  std::cout << "-Dt                                          (float) translational diffusion, DEFAULT = 0.00, [r]^2[s]^-1\n";
  std::cout << "-v                                           (float) v0, DEFAULT = 10.0 ,[r][s]^-1\n";
  std::cout << "--initial-packing-fraction                   (float) density of random intial condition, DEFAULT = 0.5\n";
  std::cout << "--box-length                                 (float) length of periodic box, DEFAULT inf => no box\n";
  std::cout << "-tau                                         (float) reorientation coefficient, DEFAULT = 0.0\n";
  std::cout << "-alpha                                       (float) reorientation exponent, DEFAULT = 0.0\n";
  std::cout << "--save-every                                 (int) save state every --save-every time steps, DEFAULT = 10\n";
  std::cout << "--random-seed                                (unsigned long) DEFAULT = 31415926535897\n";
  std::cout << "-silent                                      suppress cout DEFAULT = 0 (don't suppress)\n";
}

int main(int argc, char ** argv){
  if ( (argc+1) % 2 == 0 && argc >= 1){
    // should have -OptionName Option pairs, + the program name
    for (int i = 1; i+1 < argc; i+=2){
      std::string OptionName = argv[i];
      std::string Option = argv[i+1];
      if (OptionName == "-h"){
        help();
        return 0;
      }
      else if (OptionName == "-N"){
        N = std::stoi(Option);
      }
      else if (OptionName == "-T"){
        TotalTime = std::stod(Option);
      }
      else if (OptionName == "-dt"){
        dt = std::stod(Option);
      }
      else if (OptionName == "-mur"){
        mu_r = std::stod(Option);
      }
      else if (OptionName == "-a"){
        a = std::stod(Option);
      }
      else if (OptionName == "-b"){
        b = std::stod(Option);
      }
      else if (OptionName == "-k"){
        k = std::stod(Option);
      }
      else if (OptionName == "-mu"){
        mu = std::stod(Option);
      }
      else if (OptionName == "-Dr"){
        Dr = std::stod(Option);
      }
      else if (OptionName == "-Dt"){
        Dt = std::stod(Option);
      }
      else if (OptionName == "-v"){
        v0 = std::stod(Option);
      }
      else if (OptionName == "--initial-packing-fraction"){
        packing = std::stod(Option);
      }
      else if (OptionName == "--box-length"){
        L = std::stod(Option);
      }
      else if (OptionName == "-tau"){
        tau = std::stod(Option);
      }
      else if (OptionName == "-alpha"){
        alpha = std::stod(Option);
      }
      else if (OptionName == "--save-every"){
        StepsBetweenSaves = std::stoi(Option);
      }
      else if (OptionName == "--random-seed"){
        seed = std::stoi(Option);
      }
      else if (OptionName == "-silent"){
        silent = std::stoi(Option);
      }
    }
  }
  else{
    std::cout << "Incomplete options\n";
    std::cout << "Options should be given in pairs, e.g -N 100\n";
    help();
    return 0;
  }
  if (silent == 0){
    std::cout << "#######################################\n";
    std::cout << "Parameters Set: \n";
    std::cout << "N                                      " << N << std::endl;
    std::cout << "T                                      " << TotalTime << std::endl;
    std::cout << "dt                                     " << dt << std::endl;
    std::cout << "a                                      " << a << std::endl;
    std::cout << "b                                      " << b << std::endl;
    std::cout << "force strength                         " << k << std::endl;
    std::cout << "mobility                               " << mu << std::endl;
    std::cout << "rotational mobility                    " << mu_r << std::endl;
    std::cout << "rotation-diffusion coefficient         " << Dr << std::endl;
    std::cout << "translation diffusion coefficient      " << Dt << std::endl;
    std::cout << "self propulsion speed                  " << v0 << std::endl;
    std::cout << "intial packing-fraction                " << packing << std::endl;
    std::cout << "box length                             " << L << std::endl;
    std::cout << "tau                                    " << tau << std::endl;
    std::cout << "alpha                                  " << alpha << std::endl;
    std::cout << "save every                             " << StepsBetweenSaves << std::endl;
    std::cout << "random seed                            " << seed << std::endl;
    std::cout << "#######################################\n";
  }

  float * X; // Positions
  float * O; // orientations, theta
  float * Trajectories; // will store the answers

  X = new float [N*2];
  O = new float [N];

  if (tau != 0.0){
    // require this for density each step
    if (silent == 0){
      std::cout << "Warning: tau != 0.0 requires density calculation each step.\n";
      std::cout << "You will loose the benifit of not copying from the device each step\n";
    }
  }

  int total_steps = int(ceil(TotalTime/dt));
  Trajectories = new float [total_steps/StepsBetweenSaves*N*4]; // x,y,o,density for each N and t

  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> uniform_real(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);

  double l = std::sqrt((N*M_PI*a*b)/packing);
  if (std::isnan(L)){
    l = sqrt((N*M_PI*a*b)/0.5);
  }
  for (int i = 0; i < N; i++){
    // initialise positions with packing fraction = packing
    X[i*2] = uniform_real(generator)*l;
    X[i*2+1] = uniform_real(generator)*l;
    // random normal oritentations
    O[i] = normal(generator)*2.0*M_PI;

    Trajectories[0*N*4 + 4*i + 0] = X[i*2];
    Trajectories[0*N*4 + 4*i + 1] = X[i*2+1];
    Trajectories[0*N*4 + 4*i + 2] = O[i];
    Trajectories[0*N*4 + 4*i + 3] = 0.0;
  }

  TakeSteps(X,O,Trajectories,N,total_steps,StepsBetweenSaves,dt,k,mu,mu_r,
    a,b,Dt,Dr,v0,tau,alpha,L);

  if (silent == 0){
    std::cout << "Simulation done, saving data...\n";
  }
  //set up output to save data.
  std::ostringstream namestring;
  namestring << "trajectories.txt";
  std::string str1 = namestring.str();
  std::ofstream output(str1.c_str());

  clock_t start;
	start = clock();

  for (int t = 0; t < total_steps/StepsBetweenSaves; t++){
    for (int i = 0; i < N; i++){
      output << Trajectories[t*N*4 + 4*i + 0] << ", ";
      output << Trajectories[t*N*4 + 4*i + 1] << ", ";
      output << Trajectories[t*N*4 + 4*i + 2] << ", ";
      output << Trajectories[t*N*4 + 4*i + 3];
      output << std::endl;
    }
  }

  float time = (clock()-start)/(float)CLOCKS_PER_SEC;
  float rounded_down = floorf(time * 100) / 100;

  if (silent == 0){
    std::cout << "Saving data took: " << rounded_down << " s\n";
  }

  std::free(X);
  std::free(O);
  std::free(Trajectories);
  return 0;
}
