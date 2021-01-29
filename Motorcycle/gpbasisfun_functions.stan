vector diagSPD_EQ(real alpha, real rho, real L, int M) {
  return sqrt((alpha^2) * sqrt(2*pi()) * rho * exp(-0.5*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2));
}
/* real spd_Matt(real alpha, real rho, real w) { */
/*   real S = 4*alpha^2 * (sqrt(3)/rho)^3 * 1/((sqrt(3)/rho)^2 + w^2)^2; */
/*   return sqrt(S); */
/* } */
vector diagSPD_periodic(real alpha, real rho, int M) {
  real a = 1/rho^2;
  int one_to_M[M];
  for (m in 1:M) one_to_M[m] = m;
  vector[M] q = sqrt(alpha^2 * 2 / exp(a) * to_vector(modified_bessel_first_kind(one_to_M, a)));
  return append_row(q,q);
}
matrix PHI_EQ(int N, int M, real L, vector x) {
  return sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
}
matrix PHI_periodic(int N, int M, real w0, vector x) {
  matrix[N,M] mw0x = diag_post_multiply(rep_matrix(w0*x, M), linspaced_vector(M, 1, M));
  return append_col(cos(mw0x), sin(mw0x));
}
