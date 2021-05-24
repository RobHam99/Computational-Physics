#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
using namespace std;

int main()
{
  chrono::time_point<std::chrono::system_clock> start, end;
  start = chrono::system_clock::now();

  float a[3], w=2.*M_PI, m=1., x[3] = {-1., 0., 0.}, v[3] = {0., 2*M_PI, 0.}, x_full[3], v_full[3], v_half[3], dt=0.0001, t0=0., t1=10.;
  int i, ii, j = 0;

  ofstream file;
  file.open("x.csv");
  file << x[0] << "," << x[1] << "," << x[2] << "\n";

  ofstream file2;
  file2.open("v.csv");
  file2 << v[0] << "," << v[1] << "," << v[2] << "\n";

  for(i = 0; i < 3; i++) {
     a[i] = -w*w * x[i] / (sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])*sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])*sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]));
  }
  cout << setprecision(9) << a[0] << " " << a[1] << " " << a[2];

  while(t0 < t1) {
      for(i = 0; i < 3; i++) {
        v_half[i] = v[i] + a[i] * dt/2;
        x_full[i] = x[i] + v_half[i] * dt;

        a[i] = -w*w * x_full[i] / (sqrt(x_full[0]*x_full[0] + x_full[1]*x_full[1] + x_full[2]*x_full[2])*sqrt(x_full[0]*x_full[0] + x_full[1]*x_full[1] + x_full[2]*x_full[2])*sqrt(x_full[0]*x_full[0] + x_full[1]*x_full[1] + x_full[2]*x_full[2]));
        v_full[i] = v_half[i] + a[i] * dt/2;

        x[i] = x_full[i];
        v[i] = v_full[i];

      }
      file << x[0] << "," << x[1] << "," << x[2] << "\n";
      file2 << v[0] << "," << v[1] << "," << v[2] << "\n";

      t0 += dt;
      j += 1;
  }
  end = chrono::system_clock::now();
  cout << "Time in microseconds: " << chrono::duration_cast<chrono::microseconds>(end - start).count();

  return 0;
}
