#ifndef INERTIA_CUH
#define INERTIA_CUH

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

__global__ void Step(float * X, float * V, float * O, float * W, float * F, int P, float dt,
                     float M, float J, float xi_t, float xi_r, float k, float mu, float r, float Dt,
                     float Dr, float v0, float tau, float alpha, float COM_x, float COM_y){
  int ROW = blockIdx.x*blockDim.x + threadIdx.x;
  float ax = 0.0;
  float ay = 0.0;
  float ao = 0.0;
  float turning = 0.0;
  float R = 0.0;
  float norm_V = 0.0;
  float cross = 0.0;

	float I1 = xi_t / M;
	float I2 = xi_r / J;
  if (ROW < P){
    curandState state;
    curand_init(clock64(), ROW, 0, &state);
    // acceleration
    ax = I1 * (v0*cos(O[ROW]) + curand_normal(&state)*sqrt(2.0*Dt/dt) - V[ROW*2]) + (mu/M)*F[ROW*2];
    ay = I1 * (v0*sin(O[ROW]) + curand_normal(&state)*sqrt(2.0*Dt/dt) - V[ROW*2+1]) + (mu/M)*F[ROW*2+1];
    // angular acceleration
    ao = I2 * (curand_normal(&state)*sqrt(2.0*Dr/dt) - W[ROW]);
    // reorientation term
    R = (COM_x - X[ROW*2])*(COM_x - X[ROW*2]) + (COM_y - X[ROW*2+1])*(COM_y - X[ROW*2+1]);
    R = sqrt(R);
    norm_V = V[ROW*2]*V[ROW*2] + V[ROW*2+1]*V[ROW*2+1];
    norm_V = sqrt(norm_V);

    cross = V[ROW*2]*(COM_y - X[ROW*2+1]) - V[ROW*2+1]*(COM_x - X[ROW*2]);
    cross = cross / (R*norm_V);
    turning = (1.0/J)*tau*pow(R,alpha)*cross;
    // x update
    X[ROW*2] = X[ROW*2] + dt*V[ROW*2];
    X[ROW*2+1] = X[ROW*2+1] + dt*V[ROW*2+1];
    // V update
    V[ROW*2] = V[ROW*2] + dt*ax;
    V[ROW*2+1] = V[ROW*2+1] + dt*ay;
    // O update
    O[ROW] = O[ROW] + dt*W[ROW];
    // W update
    W[ROW] = W[ROW] + dt*ao + dt*turning;
    // reset forces
    F[ROW*2] = 0.0;
    F[ROW*2+1] = 0.0;
  }
}
void TakeSteps(float * X, float * V, float * O, float * W, float * Trajectories,
               int N, int total_steps, int StepsBetweenSaves, float dt, float M,
							 float J, float xi_t, float xi_r, float k, float mu, float r,
							 float Dt, float Dr, float v0,float tau, float alpha){
  float * d_X;
  float * d_V;
  float * d_O;
  float * d_W;
  float * d_F;

  float * F = new float [N*2];
  for (int i = 0; i < N; i++){
    F[i*2] = 0.0;
    F[i*2+1] = 0.0;
  }

  cudaMalloc(&d_X,N*2*sizeof(float));
  cudaMalloc(&d_V,N*2*sizeof(float));
  cudaMalloc(&d_F,N*2*sizeof(float));
  cudaMalloc(&d_O,N*sizeof(float));
  cudaMalloc(&d_W,N*sizeof(float));

  cudaMemcpy(d_X,X,N*2*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_V,V,N*2*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_F,F,N*2*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_O,O,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_W,W,N*sizeof(float),cudaMemcpyHostToDevice);

  dim3 ForcesthreadsPerBlock(8, 8);
  int bx = (N + ForcesthreadsPerBlock.x - 1)/ForcesthreadsPerBlock.x;
  int by = (N + ForcesthreadsPerBlock.y - 1)/ForcesthreadsPerBlock.y;
  dim3 ForcesblocksPerGrid(bx,by);

  cout << bx << ", " << by << endl;

  int blockSize;   // The launch configurator returned block size
  int minGridSize; // The minimum grid size needed to achieve the
                   // maximum occupancy for a full device launch
  int gridSize;    // The actual grid size needed, based on input size

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                                      Step, 0, N);
  // Round up according to array size
  gridSize = (N + blockSize - 1) / blockSize;

  cout << gridSize << ", " << blockSize << endl;

  int Position = 0;
  int barWidth = 70.0;
  clock_t start;
  start = clock();
  float COM_x;
  float COM_y;
  cout << "Total Time Steps: " << total_steps/StepsBetweenSaves << endl;
  for (int s = 0; s < total_steps/StepsBetweenSaves; s++){
    for (int t = 0; t < StepsBetweenSaves; t++){
      COM_x = 0.0;
      COM_y = 0.0;
      COM<<<ForcesblocksPerGrid,ForcesthreadsPerBlock>>>(d_X,N,COM_x,COM_y);
      cudaDeviceSynchronize();
      COM_x = COM_x / N;
      COM_y = COM_y / N;
      Forces<<<ForcesblocksPerGrid,ForcesthreadsPerBlock>>>(d_X,d_F,N,dt,k,mu,r);
      cudaDeviceSynchronize();
      Step<<<gridSize,blockSize>>>(d_X,d_V,d_O,d_W,d_F,N,dt,M,J,xi_t,xi_r,k,mu,r,Dt,Dr,v0,tau,alpha,COM_x,COM_y);
      cudaDeviceSynchronize();
    }
    cudaMemcpy(X,d_X,N*2*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(V,d_V,N*2*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(O,d_O,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(W,d_W,N*sizeof(float),cudaMemcpyDeviceToHost);
    // update Trajectories
    for (int i = 0; i < N; i++){
      Trajectories[s*N*6 + 6*i + 0] = X[i*2];
      Trajectories[s*N*6 + 6*i + 1] = X[i*2+1];
      Trajectories[s*N*6 + 6*i + 2] = V[i*2];
      Trajectories[s*N*6 + 6*i + 3] = V[i*2+1];
      Trajectories[s*N*6 + 6*i + 4] = O[i];
      Trajectories[s*N*6 + 6*i + 5] = W[i];
    }
    cout << "[";
    Position = barWidth*float(s)/float(total_steps/StepsBetweenSaves);
    for (int i = 0; i < barWidth; i++){
      if (i < Position) cout << "=";
      else if (i == Position) cout << ">";
      else cout << " ";
    }
    float time = (clock()-start)/(float)CLOCKS_PER_SEC;
    float rounded_down = floorf(time * 100) / 100;
    cout << "]" << int(100*float(s)/float(total_steps/StepsBetweenSaves)) << " % | " << rounded_down << "s\r";
    cout.flush();
  }
  cout.flush();
  cout << endl;

  cudaFree(d_X);
  cudaFree(d_V);
  cudaFree(d_O);
  cudaFree(d_W);
  cudaFree(d_F);

  free(F);
  return;
}

#endif
