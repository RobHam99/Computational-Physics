#include <iostream>
#include <cmath>
#include <fstream>

int main(){

  double k1, l1, k2, l2, t0 = 0, t1 = 80,
  dt = 0.001, v = 0, x = 1, zeta = 0.07, w0 = 1, A = 1, omega = 2.5;

  // files for t and x
  std::ofstream t_file;
  t_file.open("t.txt");


  std::ofstream x_file;
  x_file.open("x.txt");


  while (t0 < t1) {
    t_file << t0 << "\n";
    x_file << x << "\n";

    k1 = dt * v;
    l1 = dt * (-2 * zeta * w0 * v - w0*w0 * x + A * sin(omega*t0));

    k2 = dt * (v + 0.5 * l1);
    l2 = dt * (-2 * zeta * w0 * (v + 0.5 * l1) - w0*w0 * (x + 0.5 * k1) + A * sin(omega * (t0 + 0.5 * dt)));

    x += k2;
    v += l2;
    t0 += dt;

  };



  return 0;
}
