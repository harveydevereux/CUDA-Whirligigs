#ifndef NONINERTIAL_CUH
#define NONINERTIAL_CUH

#include "../include/density.h"

// __device__ void EllipseRangeParameter(float * ret, float theta_a, float theta_b, float r1, float r2, float chi, float sigma_0){
// 	ret = sigma_0/sqrt(1.0-(chi/2.0)*(
//     pow(r1*cos(theta_a)+r2*sin(theta_a) + r1*cos(theta_b)+r2*sin(theta_b),2.0) / (1+chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))+
//     pow(r1*cos(theta_a)+r2*sin(theta_a) - r1*cos(theta_b)-r2*sin(theta_b),2.0) / (1-chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))
// 	));
// }
//
// __device__ void EllipsePotential(float * ret, float sigma_ij, float r, float k, float sigma_0){
// 	ret = 0.5*k*pow(sigma_ij-r/sigma_0,2.0);
// }

__global__ void ForcesEllipse(float * X, float * theta, float * F, int P, float dt,float k, float a, float b){
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  int COL = blockIdx.y*blockDim.y + threadIdx.y;
	float chi = ( pow(a/b,2) -1 )/( pow(a/b,2) +1);
	float sigma_0 = 2*a;
	if (ROW < P && COL < P){
		float Rx = X[COL*2] - X[ROW*2];
		float Ry = X[COL*2+1] - X[ROW*2+1];
		float dist = sqrt(Rx*Rx+Ry*Ry);

		float theta_a = theta[ROW];
		float theta_b = theta[COL];
		float r1 = Rx/dist;
		float r2 = Ry/dist;
		float sigma_ij = sigma_0/sqrt(1.0-(chi/2.0)*(pow(r1*cos(theta_a)+r2*sin(theta_a) + r1*cos(theta_b)+r2*sin(theta_b),2.0) / (1+chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))+pow(r1*cos(theta_a)+r2*sin(theta_a) - r1*cos(theta_b)-r2*sin(theta_b),2.0) / (1-chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))));
		if (dist <= sigma_ij){
			F[ROW*2] -= k*(sigma_ij - dist) * r1 / (sigma_0*sigma_0);
			F[ROW*2+1] -= k*(sigma_ij - dist) * r2 / (sigma_0*sigma_0);
		}
	}
}


__global__ void TorqueEllipse(float * X, float * theta, float * T, int P, float dt,float k,float a, float b, float dtheta=0.01){
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  int COL = blockIdx.y*blockDim.y + threadIdx.y;
	float chi = ( pow(a/b,2) -1 )/( pow(a/b,2) +1);
	float sigma_0 = 2*a;
  if (ROW < P && COL < P){
		float Rx = X[COL*2] - X[ROW*2];
		float Ry = X[COL*2+1] - X[ROW*2+1];
		float dist = sqrt(Rx*Rx+Ry*Ry);
		float theta_a = theta[ROW];
		float theta_b = theta[COL];
		float r1 = Rx/dist;
		float r2 = Ry/dist;
		float sigma_ij = sigma_0/sqrt(1.0-(chi/2.0)*(
	    pow(r1*cos(theta_a)+r2*sin(theta_a) + r1*cos(theta_b)+r2*sin(theta_b),2.0) / (1+chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))+
	    pow(r1*cos(theta_a)+r2*sin(theta_a) - r1*cos(theta_b)-r2*sin(theta_b),2.0) / (1-chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))
		));
		if (dist <= sigma_ij){
			theta_a = theta_a + dtheta;
			float dsigma_ij = sigma_0/sqrt(1.0-(chi/2.0)*(
				    pow(r1*cos(theta_a)+r2*sin(theta_a) + r1*cos(theta_b)+r2*sin(theta_b),2.0) / (1+chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))+
				    pow(r1*cos(theta_a)+r2*sin(theta_a) - r1*cos(theta_b)-r2*sin(theta_b),2.0) / (1-chi*(cos(theta_a)*cos(theta_b) + sin(theta_a)*sin(theta_b)))
					));
			T[ROW] -= ( 0.5*k*pow( (dsigma_ij-dist)/sigma_0 ,2.0) - 0.5*k*pow( (sigma_ij-dist)/sigma_0 ,2.0) )/dtheta;
		}
  }
}

__global__ void Forces(float * X, float * F, int P, float dt,float k, float mu, float r){
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  int COL = blockIdx.y*blockDim.y + threadIdx.y;
  if (ROW < P && COL < P){
    // collision forces
    if (ROW != COL){
      float Rx = X[COL*2] - X[ROW*2];
      float Ry = X[COL*2+1] - X[ROW*2+1];
      float dist = sqrt(Rx*Rx+Ry*Ry);
      if (dist < 2.0*r){
        F[ROW*2] += -1.0*k*(2.0*r-dist)*(Rx/dist);
        F[ROW*2+1] += -1.0*k*(2.0*r-dist)*(Ry/dist);
      }
    }
  }
}

__global__ void COM(float * X, int N, float COM_x, float COM_y){
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  if (ROW < N){
    COM_x += X[ROW*2];
    COM_y += X[ROW*2+1];
  }
}


__global__ void GenerateNRandomUniforms(float* numbers, unsigned long seed, float min, float max, int N) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {

        curandState state;

        curand_init(seed, i, 0, &state);

        numbers[i] = min + curand_uniform(&state)*max;
    }
}

__global__ void Reorientation(float * X, float * theta, double * density, float * V, int N, float COM_x, float COM_y, float dt, float tau, float alpha, float mur){
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  if (ROW < N){
		float Rx = X[ROW*2] - COM_x;
		float Ry = X[ROW*2+1] - COM_y;
		float R = sqrt(Rx*Rx+Ry*Ry);
		Rx = Rx/R;
		Ry = Ry/R;

		float Vx = V[ROW*2]; //cos(theta[ROW]);
		float Vy = V[ROW*2+1]; //sin(theta[ROW]);

		float v = sqrt(Vx*Vx+Vy*Vy);

		if (v != 0.0){
			Vx = Vx/v;
			Vy = Vy/v;

			// "cross product"
			float kappa = Rx*Vy - Ry*Vx;

			theta[ROW] += dt*kappa*tau*pow(density[ROW],alpha);
		}
  }
}

__global__ void Step(float * X, float * theta, float * F, float * T, float k, float mu, float mur, float Dr, float Dt, float dt, float v0, float L, float P){
  // calculate force terms (inc random) and forward propulsion
  // X      1d flattened array of positions
  // theta  1d array of angles
  // r      particle radius
  // k      harmonic force constant
  // P      is the number of particles
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  int COL = blockIdx.y*blockDim.y + threadIdx.y;
  if (ROW < P && COL < P){
    curandState state;
    curand_init(clock64(), ROW, 0, &state);
		// self propulsion
		X[ROW*2] += v0*cos(theta[ROW])*dt;
		X[ROW*2+1] += v0*sin(theta[ROW])*dt;
		// force
		X[ROW*2] += mu*dt*F[ROW*2];
		X[ROW*2+1] += mu*dt*F[ROW*2+1];
		// torque
		theta[ROW] += mur*dt*T[ROW];
		// reset force and torque
		F[ROW*2] = 0.0;
		F[ROW*2+1] = 0.0;
		T[ROW] = 0.0;
    // random component on angle
    theta[ROW] += curand_normal(&state)*sqrt(2.0*Dr*dt);
    // random force
    X[ROW*2] += curand_normal(&state)*sqrt(2.0*Dt*dt);
    X[ROW*2+1] += curand_normal(&state)*sqrt(2.0*Dt*dt);

		if (isinf(L) == false){
			if (X[ROW*2] < 0.0){
				X[ROW*2] = L+X[ROW*2];
			}
			if (X[ROW*2+1] < 0.0){
				X[ROW*2+1] = L+X[ROW*2+1];
			}
			if (X[ROW*2] > L){
				X[ROW*2] = X[ROW*2]-L;
			}
			if (X[ROW*2+1] > L){
				X[ROW*2+1] = X[ROW*2+1]-L;
			}
		}
  }
}

void TakeSteps(float * X, float * theta, float * Trajectories,
               int N, int total_steps, int StepsBetweenSaves, float dt,
							 float k, float mu, float mur, float a, float b, float Dt, float Dr, float v0,
							 float tau, float alpha, float L){
  float * d_X;
  float * d_theta;
	float * d_F;
	double * d_density;
	float * d_T;
	float * d_V;
	// for daluanator
	std::vector<double> coords(2*N,0.0);

	float * F = new float [N*2];
	double * density = new double[N];
	float * V = new float [N*2];
	for (int i = 0; i < N; i++){
		F[i*2] = 0.0;
		F[i*2+1] = 0.0;
		density[i] = 0.0;
		V[i] = 0.0;
	}


  size_t memX = N*2*sizeof(float);
  size_t memT = N*sizeof(float);

  cudaMalloc(&d_X,memX);
  cudaMalloc(&d_theta,memT);
	cudaMalloc(&d_F,memX);
	cudaMalloc(&d_density,N*sizeof(double));
	cudaMalloc(&d_T,memT);
	cudaMalloc(&d_V,memX);

  cudaMemcpy(d_X,X,memX,cudaMemcpyHostToDevice);
  cudaMemcpy(d_theta,theta,memT,cudaMemcpyHostToDevice);
	cudaMemcpy(d_F,F,memX,cudaMemcpyHostToDevice);
	cudaMemcpy(d_V,V,memX,cudaMemcpyHostToDevice);

	dim3 ForcesthreadsPerBlock(8, 8);
  int bx = (N + ForcesthreadsPerBlock.x - 1)/ForcesthreadsPerBlock.x;
  int by = (N + ForcesthreadsPerBlock.y - 1)/ForcesthreadsPerBlock.y;
  dim3 ForcesblocksPerGrid(bx,by);

	int blockSize;   // The launch configurator returned block size
	int minGridSize; // The minimum grid size needed to achieve the
									 // maximum occupancy for a full device launch
	int gridSize;    // The actual grid size needed, based on input size

	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
																			Step, 0, N);
	// Round up according to array size
	gridSize = (N + blockSize - 1) / blockSize;

	int Position = 0;
	int barWidth = 70.0;
	clock_t start;
	start = clock();
	float COM_x;
	float COM_y;
	if (silent == 0){
		std::cout << "Total Time Steps: " << total_steps/StepsBetweenSaves << std::endl;
	}
	for (int s = 0; s < total_steps/StepsBetweenSaves; s++){
		for (int i = 0; i < N; i++){
			float l = 2*a;
			if (a != b){
				l = 4*a;
			}
			coords[2*i] = X[2*i]/(l);
			coords[2*i+1] = X[2*i+1]/(l);
		}
		for (int t = 0; t < StepsBetweenSaves; t++){
			COM_x = 0.0;
			COM_y = 0.0;
			COM<<<ForcesblocksPerGrid,ForcesthreadsPerBlock>>>(d_X,N,COM_x,COM_y);
			cudaDeviceSynchronize();
			COM_x = COM_x / N;
			COM_y = COM_y / N;
			ForcesEllipse<<<ForcesblocksPerGrid,ForcesthreadsPerBlock>>>(d_X,d_theta,d_F,N,dt,k,a,b);
			cudaDeviceSynchronize();
			TorqueEllipse<<<ForcesblocksPerGrid,ForcesthreadsPerBlock>>>(d_X,d_theta,d_T,N,dt,k,a,b,0.01);
			cudaDeviceSynchronize();
			if (tau != 0.0){
				// compute delaunay (cpu)
				delaunator::Delaunator d(coords);
				// compute density (cpu)
		    for (int i = 0; i < N; i++){
						density[i] = 0.0;
		        density[i] = WeightedDTFELocalDensity(d,coords,i); // *a^2 for length scale
		    }
				cudaMemcpy(d_density,density,N*sizeof(double),cudaMemcpyHostToDevice);
				// apply re-orientation
				 Reorientation<<<ForcesblocksPerGrid,ForcesthreadsPerBlock>>>(d_X,d_theta,d_density,d_V,N,COM_x,COM_y,dt,tau,alpha,mur);
				 cudaDeviceSynchronize();
			}
			Step<<<gridSize,blockSize>>>(d_X,d_theta,d_F,d_T,k,mu,mur,Dr,Dt,dt,v0,L,N);
			cudaDeviceSynchronize();
			if (tau != 0.0){
				// keep track for density calcs
				cudaMemcpy(X,d_X,N*2*sizeof(float),cudaMemcpyDeviceToHost);
				cudaMemcpy(theta,d_theta,N*sizeof(float),cudaMemcpyDeviceToHost);
				// keep track of inst. velocity for reorientation term
				for (int j = 0; j < N; j++){
					V[j*2] = X[j*2] - coords[j*2];
					V[j*2+1] = X[j*2+1] - coords[j*2+1];
					//std::cout << V[j*2] << ", " << V[j*2+1] << std::endl;
				}
				cudaMemcpy(d_V,V,N*2*sizeof(float),cudaMemcpyHostToDevice);
			}
		}
		if (tau == 0.0){
			cudaMemcpy(X,d_X,N*2*sizeof(float),cudaMemcpyDeviceToHost);
			cudaMemcpy(theta,d_theta,N*sizeof(float),cudaMemcpyDeviceToHost);
		}
		// update Trajectories
		for (int i = 0; i < N; i++){
			Trajectories[s*N*4 + 4*i + 0] = X[i*2];
			Trajectories[s*N*4 + 4*i + 1] = X[i*2+1];
			Trajectories[s*N*4 + 4*i + 2] = theta[i];
			Trajectories[s*N*4 + 4*i + 3] = density[i];
		}
		if (silent == 0){
			std::cout << "[";
		}
		Position = barWidth*float(s)/float(total_steps/StepsBetweenSaves);
		if (silent == 0){
			for (int i = 0; i < barWidth; i++){
				if (i < Position) std::cout << "=";
				else if (i == Position) std::cout << ">";
				else std::cout << " ";
			}
		}
		float time = (clock()-start)/(float)CLOCKS_PER_SEC;
		float rounded_down = floorf(time * 100) / 100;
		if (silent == 0){
			std::cout << "]" << int(100*float(s)/float(total_steps/StepsBetweenSaves)) << " % | " << rounded_down << "s\r";
			std::cout.flush();
		}
	}
	if (silent == 0){
		std::cout.flush();
		std::cout << std::endl;
	}

  cudaFree(d_X);
  cudaFree(d_theta);
	cudaFree(d_F);
	cudaFree(d_density);
	cudaFree(d_T);
	cudaFree(d_V);

	free(F);
	free(density);
	free(V);
  return;
}

#endif
