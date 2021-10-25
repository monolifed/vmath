#include "vmath.h"

#include <stdio.h>

#define VMATH_IMPLEMENTATION
//#define VMATH_NOSTDMEM
//#define VMATH_DOUBLEPREC
#include "vmath.h"

#define SEPVEC4(V) V[0], V[1], V[2], V[3]
#define SEPVEC3(V) V[0], V[1], V[2]
#define SEPVEC2(V) V[0], V[1]

#define FMTVEC4 "%+0.2f, %+0.2f, %+0.2f, %+0.2f"

#define PRINTMAT4R(M, R) printf("[" FMTVEC4 "]\n", SEPVEC4(M[R]))

#define PRINTMAT4(M) \
	PRINTMAT4R(M, 0); PRINTMAT4R(M, 1); \
	PRINTMAT4R(M, 2); PRINTMAT4R(M, 3)

int main()
{
	mat4 m, n;
	mat4_eq_roty(m, 1.0);
	PRINTMAT4(m);
	printf("\n");
	mat4_inverse(n, m);
	PRINTMAT4(n);
	printf("\n");
	mat4_eq_roty(m, -1.0);
	PRINTMAT4(m);
	printf("\n");
	return 0;
}
