#include "vmath_struct.h"

#include <stdio.h>

#define VMATH_IMPLEMENTATION
//#define VMATH_NOSTDMEM
//#define VMATH_DOUBLEPREC
#include "vmath_struct.h"




#define SEPVECP4(V) V->x, V->y, V->z, V->w
#define SEPVEC4(V)  V.x, V.y, V.z, V.w
#define SEPVECP3(V) V->x, V->y, V->z
#define SEPVEC3(V)  V.x, V.y, V.z
#define SEPVECP2(V) V->x, V->y
#define SEPVEC2(V)  V.x, V.y

#define FMTVEC4 "%+0.2f, %+0.2f, %+0.2f, %+0.2f"

#define PRINTMAT4R(M, R) printf("[" FMTVEC4 "]\n", SEPVEC4(M.R))

#define PRINTMAT4(M) \
	PRINTMAT4R(M, a); PRINTMAT4R(M, b); \
	PRINTMAT4R(M, c); PRINTMAT4R(M, d)

int main()
{
	mat4 m, n;
	mat4_eq_roty(&m, 1.0);
	PRINTMAT4(m);
	printf("\n");
	mat4_inverse(&n, &m);
	PRINTMAT4(n);
	printf("\n");
	mat4_eq_roty(&m, -1.0);
	PRINTMAT4(m);
	printf("\n");
	return 0;
}
