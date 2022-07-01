vector diagSPD_EQ(real alpha, real rho, real L, int M) {
  vector[M] one_to_M2;
  for (m in 1:M) one_to_M2[m] = m^2;
  return alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * one_to_M2);
}
vector diagSPD_Matern32(real alpha, real rho, real L, int M) {
  vector[M] one_to_M2;
  for (m in 1:M) one_to_M2[m] = m^2;
   return 2*alpha * (sqrt(3)/rho)^1.5 * inv((sqrt(3)/rho)^2 + (pi()/2/L)^2 * one_to_M2);
}
vector diagSPD_periodic(real alpha, real rho, int M) {
  real a = 1/rho^2;
  vector[M] q;
  for (m in 1:M) q[m] = sqrt(alpha^2 * 2 / exp(a) * modified_bessel_first_kind(m, a));
  return append_row(q,q);
}
matrix PHI_EQ(int N, int M, real L, vector x) {
  vector[M] one_to_M;
  for (m in 1:M) one_to_M[m] = m;
  return sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), one_to_M))/sqrt(L);
}
matrix PHI_periodic(int N, int M, real w0, vector x) {
  vector[M] one_to_M;
  matrix[N,M] mw0x;
  for (m in 1:M) one_to_M[m] = m;
  mw0x = diag_post_multiply(rep_matrix(w0*x, M), one_to_M);
  return append_col(cos(mw0x), sin(mw0x));
}
