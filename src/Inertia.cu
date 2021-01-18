#include <iostream>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_occupancy.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
using namespace std;

#include "../include/Inertia.cuh"

int N = 100;
float r = 1.0;
float k = 10.0;
float mu = 1.0;
float dt = 1.0/300.0;
float Dr = 1.0;
float Dt = 0.01;
float v0 = 10.0;
float packing = 0.4;
float TotalTime = 30.0;
int StepsBetweenSaves = 10;
float xi_t = 1.0;
float xi_r = 1.0;
float M = 1.0;
float J = 1.0;
float alpha = 0.0;
float tau = 0.0;

unsigned long seed = 31415926535897;

void help(){
  cout << "Options are: \n";
  cout << "-N                                           (int) Number of particles, DEFAULT 100\n";
  cout << "-T                                           (float) Max time, DEFAULT = 30.0, [seconds]\n";
  cout << "-dt                                          (float) Time step, DEFAULT = 1.0/300.0,[seconds]\n";
  cout << "-M                                           (float) Mass, DEFAULT = 0.0001, [kg]\n";
  cout << "-J                                           (float) Moment of inertia, DEFAULT = 0.0001, [kg][m]^2\n";
  cout << "-translational_friction                      (float) translational friction coefficient, DEFAULT = 0.0001\n";
  cout << "-rotational_friction                         (float) rotational friction coefficient, DEFAULT = 0.0001\n";
  cout << "-radius                                      (float) Particle radius, DEFAULT = 1.0, this defines the length scale\n";
  cout << "--force-strength                             (float) Spring constant, DEFAULT = 10.0\n";
  cout << "--chemical-potential                         (float) accel = f(v,x,t) + mu * sum(collision_forces(x,t)), DEFAULT = 1.0\n";
  cout << "--rotational-diffusion-coefficient           (float) rotational diffusion, DEFAULT = 1.0,[rad]^2[s]^-1\n";
  cout << "--translational-diffusion-coefficient        (float) translational diffusion, DEFAULT = 0.01, [r]^2[s]^-1\n";
  cout << "--self-propulsion-speed                      (float) v0, DEFAULT = 10.0 ,[r][s]^-1\n";
  cout << "--initial-packing-fraction                   (float) density of random intial condition, DEFAULT = 0.4\n";
  cout << "-tau                                         (float) reorientation coefficient, DEFAULT = 0.0\n";
  cout << "-alpha                                       (float) reorientation exponent, DEFAULT = 0.0\n";
  cout << "--save-every                                 (int) save state every --save-every time steps, DEFAULT = 10\n";
  cout << "--random-seed                                (unsigned long) DEFAULT = 31415926535897\n";
}

int main(int argc, char ** argv){
  if ( (argc+1) % 2 == 0 && argc >= 1){
    // should have -OptionName Option pairs, + the program name
    for (int i = 1; i+1 < argc; i+=2){
      string OptionName = argv[i];
      string Option = argv[i+1];
      if (OptionName == "-h"){
        help();
        return 0;
      }
      else if (OptionName == "-N"){
        N = stoi(Option);
      }
      else if (OptionName == "-T"){
        TotalTime = stod(Option);
      }
      else if (OptionName == "-dt"){
        dt = stod(Option);
      }
      else if (OptionName == "-M"){
        M = stod(Option);
      }
      else if (OptionName == "-J"){
        J = stod(Option);
      }
      else if (OptionName == "-translational_friction"){
        xi_t = stod(Option);
      }
      else if (OptionName == "-rotational_friction"){
        xi_r = stod(Option);
      }
      else if (OptionName == "-radius"){
        r = stod(Option);
      }
      else if (OptionName == "--force-strength"){
        k = stod(Option);
      }
      else if (OptionName == "--chemical-potential"){
        mu = stod(Option);
      }
      else if (OptionName == "--rotational-diffusion-coefficient"){
        Dr = stod(Option);
      }
      else if (OptionName == "--translational-diffusion-coefficient"){
        Dt = stod(Option);
      }
      else if (OptionName == "--self-propulsion-speed"){
        v0 = stod(Option);
      }
      else if (OptionName == "--initial-packing-fraction"){
        packing = stod(Option);
      }
      else if (OptionName == "-tau"){
        tau = stod(Option);
      }
      else if (OptionName == "-alpha"){
        alpha = stod(Option);
      }
      else if (OptionName == "--save-every"){
        StepsBetweenSaves = stoi(Option);
      }
      else if (OptionName == "--random-seed"){
        seed = stoi(Option);
      }
    }
  }
  else{
    cout << "Incomplete options\n";
    cout << "Options should be given in pairs, e.g -N 100\n";
    help();
    return 0;
  }
  cout << "#####################################################################\n";
  cout << "Parameters Set: \n";
  cout << "N                                      " << N << endl;
  cout << "T                                      " << TotalTime << endl;
  cout << "dt                                     " << dt << endl;
  cout << "M                                      " << M << endl;
  cout << "J                                      " << J << endl;
  cout << "translational friction                 " << xi_t << endl;
  cout << "rotational friction                    " << xi_r << endl;
  cout << "radius                                 " << r << endl;
  cout << "force strength                         " << k << endl;
  cout << "chemical potential                     " << mu << endl;
  cout << "rotation-diffusion coefficient         " << Dr << endl;
  cout << "translation diffusion coefficient      " << Dt << endl;
  cout << "self propulsion speed                  " << v0 << endl;
  cout << "intial packing-fraction                " << packing << endl;
  cout << "tau                                    " << tau << endl;
  cout << "alpha                                  " << alpha << endl;
  cout << "save every                             " << StepsBetweenSaves << endl;
  cout << "random seed                            " << seed << endl;
  cout << "####################################################################\n";

  float * X; // Positions
  float * V; // velocities
  float * O; // orientations, theta
  float * W; // angular speed, omega
  float * Trajectories; // will store the answers

  X = new float [N*2];
  V = new float [N*2];
  O = new float [N];
  W = new float [N];

  int total_steps = int(ceil(TotalTime/dt));
  Trajectories = new float [total_steps/StepsBetweenSaves*N*6]; // x,y,vx,vy,o,w for each N and t

  default_random_engine generator(seed);
  uniform_real_distribution<double> uniform_real(0.0, 1.0);
  normal_distribution<double> normal(0.0, 1.0);

  int L = sqrt((N*M_PI*r*r)/packing);
  for (int i = 0; i < N; i++){
    // initialise positions with packing fraction = packing
    X[i*2] = uniform_real(generator)*L;
    X[i*2+1] = uniform_real(generator)*L;
    // random normal oritentations
    O[i] = normal(generator)*2.0*M_PI;
    V[i*2] = v0*cos(O[i]);
    V[i*2+1] = v0*sin(O[i]);
    // random normal angular speeds
    W[i] = normal(generator);

    Trajectories[0*N*6 + 6*i + 0] = X[i*2];
    Trajectories[0*N*6 + 6*i + 1] = X[i*2+1];
    Trajectories[0*N*6 + 6*i + 2] = V[i*2];
    Trajectories[0*N*6 + 6*i + 3] = V[i*2+1];
    Trajectories[0*N*6 + 6*i + 4] = O[i];
    Trajectories[0*N*6 + 6*i + 5] = W[i];
  }

  TakeSteps(X,V,O,W,Trajectories,N,total_steps,StepsBetweenSaves,dt,M,J,xi_t,xi_r,k,mu,
    r,Dt,Dr,v0,tau,alpha);

  cout << "Simulation done, saving data...\n";

  //set up output to save data.
  ostringstream namestring;
  namestring << "trajectories.txt";
  string str1 = namestring.str();
  ofstream output(str1.c_str());

  for (int t = 0; t < total_steps/StepsBetweenSaves; t++){
    for (int i = 0; i < N; i++){
      output << Trajectories[t*N*6 + 6*i + 0] << ", ";
      output << Trajectories[t*N*6 + 6*i + 1] << ", ";
      output << Trajectories[t*N*6 + 6*i + 2] << ", ";
      output << Trajectories[t*N*6 + 6*i + 3] << ", ";
      output << Trajectories[t*N*6 + 6*i + 4] << ", ";
      output << Trajectories[t*N*6 + 6*i + 5];
      output << endl;
    }
  }

  free(X);
  free(V);
  free(O);
  free(W);
  free(Trajectories);
  return 0;
}
