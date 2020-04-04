#ifndef _VMATH_INCLUDE_
#define _VMATH_INCLUDE_

#ifdef VMATH_STATICDEF
#define VMATHDEF static
#else
#define VMATHDEF extern
#endif

#ifdef VMATH_DOUBLEPREC
typedef double scalar;
#else
typedef float  scalar;
#endif

// plane: x,y,z is unit normal
//        w is (signed) distance in normal direction
// quat: x,y,z is i,j,k components, w is real component
typedef struct vec4_s
{
	scalar x, y, z, w;
} vec4, quat, plane;

typedef struct vec3_s
{
	scalar x, y, z;
} vec3;

typedef struct vec2_s
{
	scalar x, y;
} vec2;


// a,b,c,d are rows of the matrix
// i.e. row major matrix
// However we use v' = Mv
// i.e. pre-multiplied
typedef struct mat4x4_s
{
	vec4 a, b, c, d;
} mat4, mat4x4;

typedef struct mat3x4_s
{
	vec4 a, b, c; // d = {0, 0, 0, 1}
} mat3x4;

typedef struct mat3x3_s
{
	vec3 a, b, c;
} mat3, mat3x3;

typedef struct mat2x2_s
{
	vec2 a, b;
} mat2, mat2x2;


// vec4 common
VMATHDEF scalar vec4_dot(vec4 *u, vec4 *v); // return u . v
VMATHDEF scalar vec4_lensqr(vec4 *v); // return |v|^2
VMATHDEF scalar vec4_len(vec4 *v); // return |v|
VMATHDEF scalar vec4_len1(vec4 *v); // return d1(v, 0)
VMATHDEF scalar vec4_distsqr(vec4 *u, vec4 *v); // return |u - v|^2
VMATHDEF scalar vec4_dist(vec4 *u, vec4 *v); // return |u - v|
VMATHDEF scalar vec4_dist1(vec4 *u, vec4 *v); // return d1(u, v)
VMATHDEF void vec4_zero(vec4 *v); // v = 0
VMATHDEF void vec4_makeunit(vec4 *v); // v = v / |v|
VMATHDEF void vec4_setcomp(vec4 *v, scalar x, scalar y, scalar z, scalar w); // v = {x, y, z, w} (set components)
VMATHDEF void vec4_setarr(vec4 *v, scalar *u); // v = {u} (set array)
VMATHDEF void vec4_copy(vec4 *r, vec4 *v); // r = v
VMATHDEF void vec4_from_vec2(vec4 *r, vec2 *v); // r = {v, 0, 0}
VMATHDEF void vec4_from_vec3(vec4 *r, vec3 *v); // r = {v, 0}
VMATHDEF void vec4_from_mat2(vec4 *v, mat2 *m); // r = {m.a, m.b}
VMATHDEF void vec4_smul(vec4 *r, scalar s, vec4 *v); // r = s * v (scalar multiplication)
VMATHDEF void vec4_negate(vec4 *r, vec4 *v); // r = -v
VMATHDEF void vec4_normalize(vec4 *r, vec4 *v); // r = v / |v| (possible div. by 0)
VMATHDEF void vec4_add(vec4 *r, vec4 *u, vec4 *v); // r = u + v
VMATHDEF void vec4_sub(vec4 *r, vec4 *u, vec4 *v); // r = u - v
VMATHDEF void vec4_tmul(vec4 *r, vec4 *u, vec4 *v); // r = u * v (term-wise multiplication)
VMATHDEF void vec4_ma(vec4 *r, vec4 *u, scalar s, vec4 *v); // r = u + t * v (multiply add)
VMATHDEF void vec4_combine(vec4 *r, scalar s, vec4 *u, scalar t, vec4 *v); // r = s * u + t * v (linear combination)
VMATHDEF void vec4_lerp(vec4 *r, vec4 *u, vec4 *v, scalar t); // r = (1 - t) * u + t * v
VMATHDEF void vec4_project(vec4 *r, vec4 *v, vec4 *n); // r = projection of v wrt (unit) n
VMATHDEF void vec4_reflect(vec4 *r, vec4 *v, vec4 *n); // r = reflection of v wrt (unit) n

// vec3 common
VMATHDEF scalar vec3_dot(vec3 *u, vec3 *v);
VMATHDEF scalar vec3_lensqr(vec3 *v);
VMATHDEF scalar vec3_len(vec3 *v);
VMATHDEF scalar vec3_len1(vec3 *v);
VMATHDEF scalar vec3_distsqr(vec3 *u, vec3 *v);
VMATHDEF scalar vec3_dist(vec3 *u, vec3 *v);
VMATHDEF scalar vec3_dist1(vec3 *u, vec3 *v);
VMATHDEF void vec3_zero(vec3 *v);
VMATHDEF void vec3_makeunit(vec3 *v);
VMATHDEF void vec3_setcomp(vec3 *v, scalar x, scalar y, scalar z);
VMATHDEF void vec3_setarr(vec3 *v, scalar *u);
VMATHDEF void vec3_copy(vec3 *r, vec3 *v);
VMATHDEF void vec3_from_vec2(vec3 *r, vec2 *v);
VMATHDEF void vec3_from_vec4(vec3 *r, vec4 *v);
VMATHDEF void vec3_smul(vec3 *r, scalar s, vec3 *v);
VMATHDEF void vec3_negate(vec3 *r, vec3 *v);
VMATHDEF void vec3_normalize(vec3 *r, vec3 *v);
VMATHDEF void vec3_add(vec3 *r, vec3 *u, vec3 *v);
VMATHDEF void vec3_sub(vec3 *r, vec3 *u, vec3 *v);
VMATHDEF void vec3_tmul(vec3 *r, vec3 *u, vec3 *v);
VMATHDEF void vec3_ma(vec3 *r, vec3 *u, scalar t, vec3 *v);
VMATHDEF void vec3_combine(vec3 *r, scalar s, vec3 *u, scalar t, vec3 *v);
VMATHDEF void vec3_lerp(vec3 *r, vec3 *u, vec3 *v, scalar t);
VMATHDEF void vec3_project(vec3 *r, vec3 *v, vec3 *n);
VMATHDEF void vec3_reflect(vec3 *r, vec3 *v, vec3 *n);

// vec2 common
VMATHDEF scalar vec2_dot(vec2 *u, vec2 *v);
VMATHDEF scalar vec2_lensqr(vec2 *v);
VMATHDEF scalar vec2_len(vec2 *v);
VMATHDEF scalar vec2_len1(vec2 *v);
VMATHDEF scalar vec2_distsqr(vec2 *u, vec2 *v);
VMATHDEF scalar vec2_dist(vec2 *u, vec2 *v);
VMATHDEF scalar vec2_dist1(vec2 *u, vec2 *v);
VMATHDEF void vec2_zero(vec2 *v);
VMATHDEF void vec2_makeunit(vec2 *v);
VMATHDEF void vec2_setcomp(vec2 *v, scalar x, scalar y);
VMATHDEF void vec2_setarr(vec2 *v, scalar *u);
VMATHDEF void vec2_copy(vec2 *r, vec2 *v);
VMATHDEF void vec2_from_vec3(vec2 *r, vec3 *v);
VMATHDEF void vec2_from_vec4(vec2 *r, vec4 *v);
VMATHDEF void vec2_smul(vec2 *r, scalar s, vec2 *v);
VMATHDEF void vec2_negate(vec2 *r, vec2 *v);
VMATHDEF void vec2_normalize(vec2 *r, vec2 *v);
VMATHDEF void vec2_add(vec2 *r, vec2 *u, vec2 *v);
VMATHDEF void vec2_sub(vec2 *r, vec2 *u, vec2 *v);
VMATHDEF void vec2_tmul(vec2 *r, vec2 *u, vec2 *v);
VMATHDEF void vec2_ma(vec2 *r, vec2 *u, scalar t, vec2 *v);
VMATHDEF void vec2_combine(vec2 *r, scalar s, vec2 *u, scalar t, vec2 *v);
VMATHDEF void vec2_lerp(vec2 *r, vec2 *u, vec2 *v, scalar t);
VMATHDEF void vec2_project(vec2 *r, vec2 *v, vec2 *n);
VMATHDEF void vec2_reflect(vec2 *r, vec2 *v, vec2 *n);

// vec other
VMATHDEF scalar vec2_cross(vec2 *u, vec2 *v); // scalar since first 2 components are always 0
VMATHDEF void vec3_cross(vec3 *r, vec3 *u, vec3 *v); // r = u x v
VMATHDEF void vec4_cross(vec4 *r, vec4 *u, vec4 *v); // r = {vec3(u) x vec3(v), 0}
VMATHDEF void plane_from_points(plane *p, vec3 *a, vec3 *b, vec3 *c);
VMATHDEF void vec3_plane_project(vec3 *r, vec3 *v, plane *p); // same as vec3_project (p has unit normal)
VMATHDEF void vec3_plane_reflect(vec3 *r, vec3 *v, plane *p); // same as vec3_reflect
VMATHDEF void plane_makeunit(plane *p); // make normal of p unit
VMATHDEF void plane_normalize(plane *r, plane *p); // normalize the normal 

// mat4 common
VMATHDEF void mat4_zero(mat4 *m); // m = 0
VMATHDEF void mat4_id(mat4 *m); // m = I
VMATHDEF void mat4_t(mat4 *m); // m = m^T
VMATHDEF void mat4_eq_rotx(mat4 *m, scalar a); // m = R_x (rotation matrix around x with angle a)
VMATHDEF void mat4_eq_roty(mat4 *m, scalar a); // m = R_y
VMATHDEF void mat4_eq_rotz(mat4 *m, scalar a); // m = R_z
VMATHDEF void mat4_eq_rotaxis(mat4 *m, vec3 *u, scalar a); // m = R_u (rotation matrix around v with angle a)
VMATHDEF void mat4_eq_scale(mat4 *m, scalar sx, scalar sy, scalar sz); // m = scale matrix
VMATHDEF void mat4_eq_shear(mat4 *m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy); // xy: of x along y
VMATHDEF void mat4_eq_trans(mat4 *m, scalar tx, scalar ty, scalar tz); // m = translation matrix
VMATHDEF void mat4_setarr(mat4 *m, scalar *s); // m = [s]
VMATHDEF void mat4_copy(mat4 *r, mat4 *m); // r = m
VMATHDEF void mat4_from_mat3(mat4 *r, mat3 *m);
VMATHDEF void mat4_from_mat3x4(mat4 *r, mat3x4 *m);
VMATHDEF void mat4_setrows(mat4 *m, vec4 *ra, vec4 *rb, vec4 *rc, vec4 *rd); // m = [ra,rb,rc,rd]
VMATHDEF void mat4_setcols(mat4 *m, vec4 *cx, vec4 *cy, vec4 *cz, vec4 *cw); // m = [cx,cy,cz,cw]^T
VMATHDEF void mat4_setrowa(mat4 *m, vec4 *v); // row a of m = v
VMATHDEF void mat4_setrowb(mat4 *m, vec4 *v);
VMATHDEF void mat4_setrowc(mat4 *m, vec4 *v);
VMATHDEF void mat4_setrowd(mat4 *m, vec4 *v);
VMATHDEF void mat4_setcolx(mat4 *m, vec4 *v); // col x of m = v
VMATHDEF void mat4_setcoly(mat4 *m, vec4 *v);
VMATHDEF void mat4_setcolz(mat4 *m, vec4 *v);
VMATHDEF void mat4_setcolw(mat4 *m, vec4 *v);
VMATHDEF void mat4_rowa(vec4 *r, mat4 *m); // r = row a of m
VMATHDEF void mat4_rowb(vec4 *r, mat4 *m);
VMATHDEF void mat4_rowc(vec4 *r, mat4 *m);
VMATHDEF void mat4_rowd(vec4 *r, mat4 *m);
VMATHDEF void mat4_colx(vec4 *r, mat4 *m); // r = col x of m
VMATHDEF void mat4_coly(vec4 *r, mat4 *m);
VMATHDEF void mat4_colz(vec4 *r, mat4 *m);
VMATHDEF void mat4_colw(vec4 *r, mat4 *m);
VMATHDEF void mat4_smul(mat4 *r, scalar s, mat4 *m); // r = s * m  (scalar multiplication)
VMATHDEF void mat4_mulrows(mat4 *r, mat4 *m, scalar a, scalar b, scalar c, scalar d); // multiply rows with scalars
VMATHDEF void mat4_mulrowv(mat4 *r, mat4 *m, vec4 *v); // multiply rows with components of v
VMATHDEF void mat4_negate(mat4 *r, mat4 *m); // r = -m
VMATHDEF void mat4_add(mat4 *r, mat4 *f, mat4 *g); // r = f + g
VMATHDEF void mat4_sub(mat4 *r, mat4 *f, mat4 *g); // r = f - g
VMATHDEF void mat4_tmul(mat4 *r, mat4 *f, mat4 *g); // r = f * g (term-wise multiplication)
// VMATHDEF void _vec4_mul_mat4(vec4 *r, vec4 *v, mat4 *m); // r = vm (r != v, used internally)
VMATHDEF void mat4_mul(mat4 *r, mat4 *f, mat4 *g); // r = f * g
VMATHDEF void mat4_ma(mat4 *r, mat4 *f, scalar t, mat4 *g); // r = f + t * g
VMATHDEF void mat4_combine(mat4 *r, scalar s, mat4 *f, scalar t, mat4 *g); // r = s * f + t * g
VMATHDEF void mat4_lerp(mat4 *r, mat4 *f, mat4 *g, scalar t); // r = (1 - t) * f + t * g
VMATHDEF void vec4_mul_mat4(vec4 *r, vec4 *v, mat4 *m); // r = vm
VMATHDEF void mat4_mul_vec4(vec4 *r, mat4 *m, vec4 *v); // r = mv
VMATHDEF void mat4_mul_vec3(vec3 *r, mat4 *m, vec3 *v); // r = mv (affine)
VMATHDEF void mat4_transpose(mat4 *r, mat4 *m); // r = m^T
VMATHDEF void mat4_inverse(mat4 *r, mat4 *m); // r = inverse(m) (r != m)
VMATHDEF void mat4_vtrace(vec4 *v, mat4 *m); // return trace (as) vector
VMATHDEF scalar mat4_trace(mat4 *m); // return trace
VMATHDEF scalar mat4_det3(mat4 *m); // return determinant of inner 3x3
VMATHDEF scalar mat4_det(mat4 *m); // return determinant of m

// mat3x4 common
VMATHDEF void mat3x4_zero(mat3x4 *m);
VMATHDEF void mat3x4_id(mat3x4 *m);
VMATHDEF void mat3x4_t(mat3x4 *m);
VMATHDEF void mat3x4_eq_rotx(mat3x4 *m, scalar a);
VMATHDEF void mat3x4_eq_roty(mat3x4 *m, scalar a);
VMATHDEF void mat3x4_eq_rotz(mat3x4 *m, scalar a);
VMATHDEF void mat3x4_eq_rotaxis(mat3x4 *m, vec3 *u, scalar a);
VMATHDEF void mat3x4_eq_scale(mat3x4 *m, scalar sx, scalar sy, scalar sz);
VMATHDEF void mat3x4_eq_shear(mat3x4 *m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy);
VMATHDEF void mat3x4_eq_trans(mat3x4 *m, scalar tx, scalar ty, scalar tz);
VMATHDEF void mat3x4_setarr(mat3x4 *m, scalar *s);
VMATHDEF void mat3x4_copy(mat3x4 *r, mat3x4 *m);
VMATHDEF void mat3x4_from_mat3(mat3x4 *r, mat3 *m);
VMATHDEF void mat3x4_from_mat4(mat3x4 *r, mat4 *m);
VMATHDEF void mat3x4_setrows(mat3x4 *m, vec4 *ra, vec4 *rb, vec4 *rc);
VMATHDEF void mat3x4_setcols(mat3x4 *m, vec3 *cx, vec3 *cy, vec3 *cz, vec3 *cw);
VMATHDEF void mat3x4_setrowa(mat3x4 *m, vec4 *v);
VMATHDEF void mat3x4_setrowb(mat3x4 *m, vec4 *v);
VMATHDEF void mat3x4_setrowc(mat3x4 *m, vec4 *v);
VMATHDEF void mat3x4_setcolx(mat3x4 *m, vec3 *v);
VMATHDEF void mat3x4_setcoly(mat3x4 *m, vec3 *v);
VMATHDEF void mat3x4_setcolz(mat3x4 *m, vec3 *v);
VMATHDEF void mat3x4_setcolw(mat3x4 *m, vec3 *v);
VMATHDEF void mat3x4_rowa(vec4 *r, mat3x4 *m); 
VMATHDEF void mat3x4_rowb(vec4 *r, mat3x4 *m); 
VMATHDEF void mat3x4_rowc(vec4 *r, mat3x4 *m); 
VMATHDEF void mat3x4_colx(vec3 *r, mat3x4 *m);
VMATHDEF void mat3x4_coly(vec3 *r, mat3x4 *m);
VMATHDEF void mat3x4_colz(vec3 *r, mat3x4 *m);
VMATHDEF void mat3x4_colw(vec3 *r, mat3x4 *m);
VMATHDEF void mat3x4_smul(mat3x4 *r, scalar s, mat3x4 *m);
VMATHDEF void mat3x4_mulrows(mat3x4 *r, mat3x4 *m, scalar a, scalar b, scalar c);
VMATHDEF void mat3x4_mulrowv(mat3x4 *r, mat3x4 *m, vec3 *v);
VMATHDEF void mat3x4_negate(mat3x4 *r, mat3x4 *m);
VMATHDEF void mat3x4_add(mat3x4 *r, mat3x4 *f, mat3x4 *g);
VMATHDEF void mat3x4_sub(mat3x4 *r, mat3x4 *f, mat3x4 *g);
VMATHDEF void mat3x4_tmul(mat3x4 *r, mat3x4 *f, mat3x4 *g);
// VMATHDEF void _vec4_mul_mat3x4(vec4 *r, vec4 *v, mat3x4 *m);
VMATHDEF void mat3x4_mul(mat3x4 *r, mat3x4 *f, mat3x4 *g);
VMATHDEF void mat3x4_ma(mat3x4 *r, mat3x4 *f, scalar t, mat3x4 *g);
VMATHDEF void mat3x4_combine(mat3x4 *r, scalar s, mat3x4 *f, scalar t, mat3x4 *g);
VMATHDEF void mat3x4_lerp(mat3x4 *r, mat3x4 *f, mat3x4 *g, scalar t);
VMATHDEF void vec4_mul_mat3x4(vec4 *r, vec4 *v, mat3x4 *m);
VMATHDEF void mat3x4_mul_vec4(vec4 *r, mat3x4 *m, vec4 *v);
VMATHDEF void mat3x4_mul_vec3(vec3 *r, mat3x4 *m, vec3 *v);
VMATHDEF void mat3x4_transpose(mat3x4 *r, mat3x4 *m);
VMATHDEF void mat3x4_inverse(mat3x4 *r, mat3x4 *m);
VMATHDEF void mat3x4_vtrace(vec3 *v, mat3x4 *m);
VMATHDEF scalar mat3x4_trace(mat3x4 *m);
VMATHDEF scalar mat3x4_det(mat3x4 *m);

// mat3 common
VMATHDEF void mat3_zero(mat3 *m);
VMATHDEF void mat3_id(mat3 *m);
VMATHDEF void mat3_t(mat3 *m);
VMATHDEF void mat3_eq_rotx(mat3 *m, scalar a);
VMATHDEF void mat3_eq_roty(mat3 *m, scalar a);
VMATHDEF void mat3_eq_rotz(mat3 *m, scalar a);
VMATHDEF void mat3_eq_rot2d(mat3 *m, scalar a);
VMATHDEF void mat3_eq_rotaxis(mat3 *m, vec3 *u, scalar a);
VMATHDEF void mat3_eq_trans2d(mat3 *m, scalar tx, scalar ty);
VMATHDEF void mat3_eq_scale(mat3 *m, scalar sx, scalar sy, scalar sz);
VMATHDEF void mat3_eq_scale2d(mat3 *m, scalar sx, scalar sy);
VMATHDEF void mat3_eq_shear(mat3 *m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy);
VMATHDEF void mat3_eq_shear2d(mat3 *m, scalar x, scalar y);
VMATHDEF void mat3_setarr(mat3 *m, scalar *s);
VMATHDEF void mat3_copy(mat3 *r, mat3 *m);
VMATHDEF void mat3_from_mat4(mat3 *r, mat4 *m);
VMATHDEF void mat3_from_mat3x4(mat3 *r, mat3x4 *m);
VMATHDEF void mat3_setrows(mat3 *m, vec3 *ra, vec3 *rb, vec3 *rc);
VMATHDEF void mat3_setcols(mat3 *m, vec3 *cx, vec3 *cy, vec3 *cz);
VMATHDEF void mat3_setrowa(mat3 *m, vec3 *v);
VMATHDEF void mat3_setrowb(mat3 *m, vec3 *v);
VMATHDEF void mat3_setrowc(mat3 *m, vec3 *v);
VMATHDEF void mat3_setcolx(mat3 *m, vec3 *v);
VMATHDEF void mat3_setcoly(mat3 *m, vec3 *v);
VMATHDEF void mat3_setcolz(mat3 *m, vec3 *v);
VMATHDEF void mat3_rowa(vec3 *r, mat3 *m);
VMATHDEF void mat3_rowb(vec3 *r, mat3 *m);
VMATHDEF void mat3_rowc(vec3 *r, mat3 *m);
VMATHDEF void mat3_colx(vec3 *r, mat3 *m);
VMATHDEF void mat3_coly(vec3 *r, mat3 *m);
VMATHDEF void mat3_colz(vec3 *r, mat3 *m);
VMATHDEF void mat3_smul(mat3 *r, scalar s, mat3 *m);
VMATHDEF void mat3_mulrows(mat3 *r, mat3 *m, scalar a, scalar b, scalar c);
VMATHDEF void mat3_mulrowv(mat3 *r, mat3 *m, vec3 *v);
VMATHDEF void mat3_negate(mat3 *r, mat3 *m);
VMATHDEF void mat3_add(mat3 *r, mat3 *f, mat3 *g);
VMATHDEF void mat3_sub(mat3 *r, mat3 *f, mat3 *g);
VMATHDEF void mat3_tmul(mat3 *r, mat3 *f, mat3 *g);
// VMATHDEF void _vec3_mul_mat3(vec3 *r, vec3 *v, mat3 *m);
VMATHDEF void mat3_mul(mat3 *r, mat3 *f, mat3 *g);
VMATHDEF void mat3_ma(mat3 *r, mat3 *f, scalar t, mat3 *g);
VMATHDEF void mat3_combine(mat3 *r, scalar s, mat3 *f, scalar t, mat3 *g);
VMATHDEF void mat3_lerp(mat3 *r, mat3 *f, mat3 *g, scalar t);
VMATHDEF void vec3_mul_mat3(vec3 *r, vec3 *v, mat3 *m);
VMATHDEF void mat3_mul_vec3(vec3 *r, mat3 *m, vec3 *v);
VMATHDEF void mat3_mul_vec2(vec2 *r, mat3 *m, vec2 *v);
VMATHDEF void mat3_transpose(mat3 *r, mat3 *m);
VMATHDEF void mat3_inverse(mat3 *r, mat3 *m);
VMATHDEF void mat3_vtrace(vec3 *v, mat3 *m);
VMATHDEF scalar mat3_trace(mat3 *m);
VMATHDEF scalar mat3_det(mat3 *m);

// mat2 common
VMATHDEF void mat2_zero(mat2 *m);
VMATHDEF void mat2_id(mat2 *m);
VMATHDEF void mat2_t(mat2 *m);
VMATHDEF void mat2_eq_rot(mat2 *m, scalar a); // around origin/z
VMATHDEF void mat2_eq_scale(mat2 *m, scalar sx, scalar sy);
VMATHDEF void mat2_eq_shear(mat2 *m, scalar x, scalar y);
VMATHDEF void mat2_setarr(mat2 *m, scalar *s);
VMATHDEF void mat2_copy(mat2 *r, mat2 *m);
VMATHDEF void mat2_from_vec4(mat2 *m, vec4 *v);
VMATHDEF void mat2_setrows(mat2 *m, vec2 *a, vec2 *b);
VMATHDEF void mat2_setcols(mat2 *m, vec2 *a, vec2 *b);
VMATHDEF void mat2_setrowa(mat2 *m, vec2 *v);
VMATHDEF void mat2_setrowb(mat2 *m, vec2 *v);
VMATHDEF void mat2_setcolx(mat2 *m, vec2 *v);
VMATHDEF void mat2_setcoly(mat2 *m, vec2 *v);
VMATHDEF void mat2_rowa(vec2 *r, mat2 *m);
VMATHDEF void mat2_rowb(vec2 *r, mat2 *m);
VMATHDEF void mat2_colx(vec2 *r, mat2 *m);
VMATHDEF void mat2_coly(vec2 *r, mat2 *m);
VMATHDEF void mat2_smul(mat2 *r, scalar s, mat2 *m);
VMATHDEF void mat2_mulrows(mat2 *r, mat2 *m, scalar a, scalar b);
VMATHDEF void mat2_mulrowv(mat2 *r, mat2 *m, vec2 *v);
VMATHDEF void mat2_negate(mat2 *r, mat2 *m);
VMATHDEF void mat2_add(mat2 *r, mat2 *f, mat2 *g);
VMATHDEF void mat2_sub(mat2 *r, mat2 *f, mat2 *g);
VMATHDEF void mat2_tmul(mat2 *r, mat2 *f, mat2 *g);
// VMATHDEF void _vec2_mul_mat2(vec2 *r, vec2 *v, mat2 *m);
VMATHDEF void mat2_mul(mat2 *r, mat2 *f, mat2 *g);
VMATHDEF void mat2_ma(mat2 *r, mat2 *f, scalar t, mat2 *g);
VMATHDEF void mat2_combine(mat2 *r, scalar s, mat2 *f, scalar t, mat2 *g);
VMATHDEF void mat2_lerp(mat2 *r, mat2 *f, mat2 *g, scalar t);
VMATHDEF void vec2_mul_mat2(vec2 *r, vec2 *v, mat2 *m);
VMATHDEF void mat2_mul_vec2(vec2 *r, mat2 *m, vec2 *v);
VMATHDEF void mat2_transpose(mat2 *r, mat2 *m);
VMATHDEF void mat2_inverse(mat2 *r, mat2 *m);
VMATHDEF void mat2_vtrace(vec2 *v, mat2 *m);
VMATHDEF scalar mat2_trace(mat2 *m);
VMATHDEF scalar mat2_det(mat2 *m);

// quat common (also vec4)
VMATHDEF void quat_id(quat *v); // q = 1
VMATHDEF void quat_from_rot(quat *r, vec3 *v, scalar a); // q = rotation around axis:v with angle:a
VMATHDEF void quat_inv(quat *r, quat *v); // r = inverse of q
VMATHDEF void quat_conj(quat *r, quat *v); // r = conjugate of q
VMATHDEF void quat_from_vec3(quat *q, vec3 *v); // q = v ({x, y, z, 0})
VMATHDEF void vec3_from_quat(vec3 *v, quat *q); // v = q ({x, y, z})
VMATHDEF void quat_mul(quat *r, quat *u, quat *v); // r = u * v (quat mult.)
VMATHDEF void quat_mul_vec3(quat *r, quat *q, vec3 *v); // quat_mul with v = {x, y, z, 0}
VMATHDEF void vec3_mul_quat(quat *r, vec3 *v, quat *q);
VMATHDEF void vec3_rotate_quat(vec3 *r, quat *q, vec3 *v);
VMATHDEF void mat3_from_quat(mat3 *m, quat *q); // q is unit
VMATHDEF void quat_from_mat3(quat *q, mat3 *m);

// mat3x4 m from scale:s rotation:r translation:t
VMATHDEF void mat3x4_from_srt(mat3x4 *m, vec3 *s, quat *r, vec3 *t);
// z direction: dir
VMATHDEF void mat3_from_dir(mat3 *m, vec3 *dir);



#endif // _VMATH_INCLUDE_

// IMPLEMENTATION
#ifdef VMATH_IMPLEMENTATION


#include <math.h>

#ifndef VMATH_NOSTDMEM
#include <string.h>
#define vmemset memset
#define vmemcpy memcpy
#else
static void *vmemset(void *d, int c, unsigned n)
{
	unsigned char *dd = d;
	for (unsigned i = 0; i < n; i++)
		dd[i] = c;
	return d;
}

static void *vmemcpy(void *d, const void *s, unsigned n)
{
	unsigned char *dd = d;
	const unsigned char *ss = s;
	for (unsigned i = 0; i < n; i++)
		dd[i] = ss[i];
	return d;
}
#endif

#ifndef VMATH_DOUBLEPREC
#define VP(X) X##.0F
#define vsqrt(X) sqrtf(X)
#define vabs(X)  fabsf(X)
#define vsin(X)  sinf(X)
#define vcos(X)  cosf(X)
#else
#define VP(X) X##.0
#define vsqrt(X) sqrt(X)
#define vabs(X)  fabs(X)
#define vsin(X)  sin(X)
#define vcos(X)  cos(X)
#endif

// -------------------
// --- common vec4 ---
// -------------------
VMATHDEF scalar vec4_dot(vec4 *u, vec4 *v)
{
	return u->x * v->x + u->y * v->y + u->z * v->z + u->w * v->w;
}

VMATHDEF scalar vec4_lensqr(vec4 *v)
{
	return v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w;
}

VMATHDEF scalar vec4_len(vec4 *v)
{
	return vsqrt(v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w);
}

VMATHDEF scalar vec4_len1(vec4 *v)
{
	return vabs(v->x) + vabs(v->y) + vabs(v->z) + vabs(v->w);
}

VMATHDEF scalar vec4_distsqr(vec4 *u, vec4 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	t = u->z - v->z; d += t * t;
	t = u->w - v->w; d += t * t;
	return d;
}

VMATHDEF scalar vec4_dist(vec4 *u, vec4 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	t = u->z - v->z; d += t * t;
	t = u->w - v->w; d += t * t;
	return vsqrt(d);
}

VMATHDEF scalar vec4_dist1(vec4 *u, vec4 *v)
{
	return vabs(u->x - v->x) + vabs(u->y - v->y)
	     + vabs(u->z - v->z) + vabs(u->w - v->w);
}

VMATHDEF void vec4_zero(vec4 *v)
{
	v->x = v->y = v->z = v->w = 0;
}

VMATHDEF void vec4_makeunit(vec4 *v)
{
	scalar s = VP(1) / vsqrt(v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w);
	v->x *= s; v->y *= s; v->z *= s; v->w *= s;
}

VMATHDEF void vec4_setcomp(vec4 *v, scalar x, scalar y, scalar z, scalar w)
{
	v->x = x; v->y = y; v->z = z; v->w = w;
}

VMATHDEF void vec4_setarr(vec4 *v, scalar *u)
{
	v->x = u[0]; v->y = u[1]; v->z = u[2]; v->w = u[3];
}

VMATHDEF void vec4_copy(vec4 *r, vec4 *v)
{
	r->x = v->x; r->y = v->y;
	r->z = v->z; r->w = v->w;
}

VMATHDEF void vec4_from_vec2(vec4 *r, vec2 *v)
{
	r->x = v->x; r->y = v->y;
	r->z = r->w = 0; 
}

VMATHDEF void vec4_from_vec3(vec4 *r, vec3 *v)
{
	r->x = v->x; r->y = v->y; r->z = v->z;
	r->w = 0; 
}

VMATHDEF void vec4_from_mat2(vec4 *v, mat2 *m)
{
	v->x = m->a.x; v->y = m->a.y;
	v->z = m->b.x; v->w = m->b.y;
}

VMATHDEF void vec4_smul(vec4 *r, scalar s, vec4 *v)
{
	r->x = s * v->x; r->y = s * v->y;
	r->z = s * v->z; r->w = s * v->w;
}

VMATHDEF void vec4_negate(vec4 *r, vec4 *v)
{
	r->x = -(v->x); r->y = -(v->y);
	r->z = -(v->z); r->w = -(v->w);
}

VMATHDEF void vec4_normalize(vec4 *r, vec4 *v)
{
	scalar s = VP(1) / vsqrt(v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w);
	r->x = v->x * s; r->y = v->y * s;
	r->z = v->z * s; r->w = v->w * s;
}

VMATHDEF void vec4_add(vec4 *r, vec4 *u, vec4 *v)
{
	r->x = u->x + v->x; r->y = u->y + v->y;
	r->z = u->z + v->z; r->w = u->w + v->w;
}

VMATHDEF void vec4_sub(vec4 *r, vec4 *u, vec4 *v)
{
	r->x = u->x - v->x; r->y = u->y - v->y;
	r->z = u->z - v->z; r->w = u->w - v->w;
}

VMATHDEF void vec4_tmul(vec4 *r, vec4 *u, vec4 *v)
{
	r->x = u->x * v->x; r->y = u->y * v->y;
	r->z = u->z * v->z; r->w = u->w * v->w;
}

VMATHDEF void vec4_ma(vec4 *r, vec4 *u, scalar s, vec4 *v)
{
	r->x = u->x + s * v->x; r->y = u->y + s * v->y;
	r->z = u->z + s * v->z; r->w = u->w + s * v->w;
}

VMATHDEF void vec4_combine(vec4 *r, scalar s, vec4 *u, scalar t, vec4 *v)
{
	r->x = s * u->x + t * v->x; r->y = s * u->y + t * v->y;
	r->z = s * u->z + t * v->z; r->w = s * u->w + t * v->w;
}

VMATHDEF void vec4_lerp(vec4 *r, vec4 *u, vec4 *v, scalar t)
{
	scalar s = VP(1) - t;
	r->x = s * u->x + t * v->x; r->y = s * u->y + t * v->y;
	r->z = s * u->z + t * v->z; r->w = s * u->w + t * v->w;
}

VMATHDEF void vec4_project(vec4 *r, vec4 *v, vec4 *n)
{
	// scalar t = vec4_dot(n, v) / vec4_dot(n, n);
	vec4_ma(r, v, -vec4_dot(n, v), n);
}

VMATHDEF void vec4_reflect(vec4 *r, vec4 *v, vec4 *n)
{
	// scalar t = vec4_dot(n, v) / vec4_dot(n, n);
	vec4_ma(r, v, -VP(2) * vec4_dot(n, v), n);
}


// -------------------
// --- common vec3 ---
// -------------------
VMATHDEF scalar vec3_dot(vec3 *u, vec3 *v)
{
	return u->x * v->x + u->y * v->y + u->z * v->z;
}

VMATHDEF scalar vec3_lensqr(vec3 *v)
{
	return v->x * v->x + v->y * v->y + v->z * v->z;
}

VMATHDEF scalar vec3_len(vec3 *v)
{
	return vsqrt(v->x * v->x + v->y * v->y + v->z * v->z);
}

VMATHDEF scalar vec3_len1(vec3 *v)
{
	return vabs(v->x) + vabs(v->y) + vabs(v->z);
}

VMATHDEF scalar vec3_distsqr(vec3 *u, vec3 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	t = u->z - v->z; d += t * t;
	return d;
}

VMATHDEF scalar vec3_dist(vec3 *u, vec3 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	t = u->z - v->z; d += t * t;
	return vsqrt(d);
}

VMATHDEF scalar vec3_dist1(vec3 *u, vec3 *v)
{
	return vabs(u->x - v->x) + vabs(u->y - v->y) + vabs(u->z - v->z);
}

VMATHDEF void vec3_zero(vec3 *v)
{
	v->x = v->y = v->z = 0;
}

VMATHDEF void vec3_makeunit(vec3 *v)
{
	scalar s = VP(1) / vsqrt(v->x * v->x + v->y * v->y + v->z * v->z);
	v->x *= s; v->y *= s; v->z *= s;
}

VMATHDEF void vec3_setcomp(vec3 *v, scalar x, scalar y, scalar z)
{
	v->x = x; v->y = y; v->z = z;
}

VMATHDEF void vec3_setarr(vec3 *v, scalar *u)
{
	v->x = u[0]; v->y = u[1]; v->z = u[2];
}

VMATHDEF void vec3_copy(vec3 *r, vec3 *v)
{
	r->x = v->x; r->y = v->y; r->z = v->z;
}

VMATHDEF void vec3_from_vec2(vec3 *r, vec2 *v)
{
	r->x = v->x; r->y = v->y; r->z = 0; 
}

VMATHDEF void vec3_from_vec4(vec3 *r, vec4 *v)
{
	r->x = v->x; r->y = v->y; r->z = v->z; 
}

VMATHDEF void vec3_smul(vec3 *r, scalar s, vec3 *v)
{
	r->x = s * v->x; r->y = s * v->y; r->z = s * v->z;
}

VMATHDEF void vec3_negate(vec3 *r, vec3 *v)
{
	r->x = -(v->x); r->y = -(v->y); r->z = -(v->z);
}

VMATHDEF void vec3_normalize(vec3 *r, vec3 *v)
{
	scalar s = VP(1) / vsqrt(v->x * v->x + v->y * v->y + v->z * v->z);
	r->x = v->x * s; r->y = v->y * s; r->z = v->z * s;
}

VMATHDEF void vec3_add(vec3 *r, vec3 *u, vec3 *v)
{
	r->x = u->x + v->x; r->y = u->y + v->y; r->z = u->z + v->z;
}

VMATHDEF void vec3_sub(vec3 *r, vec3 *u, vec3 *v)
{
	r->x = u->x - v->x; r->y = u->y - v->y; r->z = u->z - v->z;
}

VMATHDEF void vec3_tmul(vec3 *r, vec3 *u, vec3 *v)
{
	r->x = u->x * v->x; r->y = u->y * v->y; r->z = u->z * v->z;
}

VMATHDEF void vec3_ma(vec3 *r, vec3 *u, scalar t, vec3 *v)
{
	r->x = u->x + t * v->x; r->y = u->y + t * v->y; r->z = u->z + t * v->z;
}

VMATHDEF void vec3_combine(vec3 *r, scalar s, vec3 *u, scalar t, vec3 *v)
{
	r->x = s * u->x + t * v->x; r->y = s * u->y + t * v->y;
	r->z = s * u->z + t * v->z;
}

VMATHDEF void vec3_lerp(vec3 *r, vec3 *u, vec3 *v, scalar t)
{
	scalar s = VP(1) - t;
	r->x = s * u->x + t * v->x; r->y = s * u->y + t * v->y;
	r->z = s * u->z + t * v->z;
}

VMATHDEF void vec3_project(vec3 *r, vec3 *v, vec3 *n)
{
	vec3_ma(r, v, -vec3_dot(n, v), n);
}

VMATHDEF void vec3_reflect(vec3 *r, vec3 *v, vec3 *n)
{
	vec3_ma(r, v, -VP(2) * vec3_dot(n, v), n);
}


// -------------------
// --- common vec2 ---
// -------------------
VMATHDEF scalar vec2_dot(vec2 *u, vec2 *v)
{
	return u->x * v->x + u->y * v->y;
}

VMATHDEF scalar vec2_lensqr(vec2 *v)
{
	return v->x * v->x + v->y * v->y;
}

VMATHDEF scalar vec2_len(vec2 *v)
{
	return vsqrt(v->x * v->x + v->y * v->y);
}

VMATHDEF scalar vec2_len1(vec2 *v)
{
	return vabs(v->x) + vabs(v->y);
}

VMATHDEF scalar vec2_distsqr(vec2 *u, vec2 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	return d;
}

VMATHDEF scalar vec2_dist(vec2 *u, vec2 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	return vsqrt(d);
}

VMATHDEF scalar vec2_dist1(vec2 *u, vec2 *v)
{
	return vabs(u->x - v->x) + vabs(u->y - v->y);
}

VMATHDEF void vec2_zero(vec2 *v)
{
	v->x = v->y = 0;
}

VMATHDEF void vec2_makeunit(vec2 *v)
{
	scalar s = VP(1) / vsqrt(v->x * v->x + v->y * v->y);
	v->x *= s; v->y *= s;
}

VMATHDEF void vec2_setcomp(vec2 *v, scalar x, scalar y)
{
	v->x = x; v->y = y;
}

VMATHDEF void vec2_setarr(vec2 *v, scalar *u)
{
	v->x = u[0]; v->y = u[1];
}

VMATHDEF void vec2_copy(vec2 *r, vec2 *v)
{
	r->x = v->x; r->y = v->y;
}

VMATHDEF void vec2_from_vec3(vec2 *r, vec3 *v)
{
	r->x = v->x; r->y = v->y;
}

VMATHDEF void vec2_from_vec4(vec2 *r, vec4 *v)
{
	r->x = v->x; r->y = v->y; 
}

VMATHDEF void vec2_smul(vec2 *r, scalar s, vec2 *v)
{
	r->x = s * v->x; r->y = s * v->y;
}

VMATHDEF void vec2_negate(vec2 *r, vec2 *v)
{
	r->x = -(v->x); r->y = -(v->y);
}

VMATHDEF void vec2_normalize(vec2 *r, vec2 *v)
{
	scalar s = VP(1) / vsqrt(v->x * v->x + v->y * v->y);
	r->x = v->x * s; r->y = v->y * s;
}

VMATHDEF void vec2_add(vec2 *r, vec2 *u, vec2 *v)
{
	r->x = u->x + v->x; r->y = u->y + v->y;
}

VMATHDEF void vec2_sub(vec2 *r, vec2 *u, vec2 *v)
{
	r->x = u->x - v->x; r->y = u->y - v->y;
}

VMATHDEF void vec2_tmul(vec2 *r, vec2 *u, vec2 *v)
{
	r->x = u->x * v->x; r->y = u->y * v->y;
}

VMATHDEF void vec2_ma(vec2 *r, vec2 *u, scalar t, vec2 *v)
{
	r->x = u->x + t * v->x; r->y = u->y + t * v->y;
}

VMATHDEF void vec2_combine(vec2 *r, scalar s, vec2 *u, scalar t, vec2 *v)
{
	r->x = s * u->x + t * v->x; r->y = s * u->y + t * v->y;
}

VMATHDEF void vec2_lerp(vec2 *r, vec2 *u, vec2 *v, scalar t)
{
	scalar s = VP(1) - t;
	r->x = s * u->x + t * v->x; r->y = s * u->y + t * v->y;
}

VMATHDEF void vec2_project(vec2 *r, vec2 *v, vec2 *n)
{
	vec2_ma(r, v, -vec2_dot(n, v), n);
}

VMATHDEF void vec2_reflect(vec2 *r, vec2 *v, vec2 *n)
{
	vec2_ma(r, v, -VP(2) * vec2_dot(n, v), n);
}


// -------------------
// ---- vec other ----
// -------------------
VMATHDEF scalar vec2_cross(vec2 *u, vec2 *v)
{
	return u->x * v->y - u->y * v->x;
}

VMATHDEF void vec3_cross(vec3 *r, vec3 *u, vec3 *v)
{
	scalar x, y, z;
	x = u->y * v->z - u->z * v->y;
	y = u->x * v->z - u->z * v->x;
	z = u->x * v->y - u->y * v->x;
	r->x = x; r->y = -y; r->z = z;
}

VMATHDEF void vec4_cross(vec4 *r, vec4 *u, vec4 *v)
{
	scalar x, y, z;
	x = u->y * v->z - u->z * v->y;
	y = u->x * v->z - u->z * v->x;
	z = u->x * v->y - u->y * v->x;
	r->x = x; r->y = -y; r->z = z; r->w = 0;
}

VMATHDEF void plane_from_points(plane *p, vec3 *a, vec3 *b, vec3 *c)
{
	vec3 n, u, v;
	vec3_sub(&u, b, a);
	vec3_sub(&v, c, a);
	vec3_cross(&n, &u, &v);
	vec3_makeunit(&n);
	p->x = n.x; p->y = n.y; p->z = n.z;
	p->w = vec3_dot(a, &n);
}

VMATHDEF void vec3_plane_project(vec3 *r, vec3 *v, plane *p)
{
	vec3 n;
	vec3_from_vec4(&n, p);
	vec3_project(r, &n, v);
}

VMATHDEF void vec3_plane_reflect(vec3 *r, vec3 *v, plane *p)
{
	vec3 n;
	vec3_from_vec4(&n, p);
	vec3_reflect(r, &n, v);
}

VMATHDEF void plane_makeunit(plane *p)
{
	scalar s = vsqrt(p->x * p->x + p->y * p->y + p->z * p->z);
	p->w *= s;
	s = VP(1) / s;
	p->x *= s; p->y *= s; p->z *= s;
}

VMATHDEF void plane_normalize(plane *r, plane *p)
{
	scalar s = vsqrt(p->x * p->x + p->y * p->y + p->z * p->z);
	r->w = s * p->w;
	s = VP(1) / s;
	r->x = s * p->x; r->y = s * p->y; r->z = s * p->z;
}


// ---------------------
// --- mat4x4 common ---
// ---------------------
VMATHDEF void mat4_zero(mat4 *m)
{
	vmemset(m, 0, sizeof *m);
}

VMATHDEF void mat4_id(mat4 *m)
{
	vmemset(m, 0, sizeof *m);
	m->a.x = m->b.y = m->c.z = m->d.w = VP(1);
}

VMATHDEF void mat4_t(mat4 *m)
{
	scalar t;
	t = m->a.y; m->a.y = m->b.x; m->b.x = t;
	t = m->a.z; m->a.z = m->c.x; m->c.x = t;
	t = m->a.w; m->a.w = m->d.x; m->d.x = t;
	
	t = m->b.z; m->b.z = m->c.y; m->c.y = t;
	t = m->b.w; m->b.w = m->d.y; m->d.y = t;
	
	t = m->c.w; m->c.w = m->d.z; m->d.z = t;
}

VMATHDEF void mat4_eq_rotx(mat4 *m, scalar a)
{
	vmemset(m, 0, sizeof *m);
	scalar c = vcos(a), s = vsin(a);
	m->a.x = m->d.w = VP(1);
	m->b.y = c; m->b.z = -s;
	m->c.y = s; m->c.z =  c;
}

VMATHDEF void mat4_eq_roty(mat4 *m, scalar a)
{
	vmemset(m, 0, sizeof *m);
	scalar c = vcos(a), s = vsin(a);
	m->b.y = m->d.w = VP(1);
	m->a.x = c; m->a.z = -s;
	m->c.x = s; m->c.z =  c;
}

VMATHDEF void mat4_eq_rotz(mat4 *m, scalar a)
{
	vmemset(m, 0, sizeof *m);
	scalar c = vcos(a), s = vsin(a);
	m->c.z = m->d.w = VP(1);
	m->a.x = c; m->a.y = -s;
	m->b.x = s; m->b.y =  c;
}

VMATHDEF void mat4_eq_rotaxis(mat4 *m, vec3 *u, scalar a)
{
	scalar s = vsin(a), c = vcos(a);
	scalar d = 1 - c;

	scalar dxx = d * u->x * u->x;
	scalar dyy = d * u->y * u->y;
	scalar dzz = d * u->z * u->z;
	scalar dxy = d * u->x * u->y;
	scalar dxz = d * u->x * u->z;
	scalar dyz = d * u->y * u->z;
	
	scalar sx = s * u->x, sy = s * u->y, sz = s * u->z;

	m->a.x = c + dxx;  m->a.y = dxy - sz; m->a.z = dxz + sy; m->a.w = 0;
	m->b.x = dxy + sz; m->b.y = c + dyy;  m->b.z = dyz - sx; m->b.w = 0;
	m->c.x = dxz - sy; m->c.y = dyz + sx; m->c.z = c + dzz;  m->c.w = 0;
	m->d.x = 0;        m->d.y = 0;        m->d.z = 0;        m->d.w = VP(1);
}

VMATHDEF void mat4_eq_scale(mat4 *m, scalar sx, scalar sy, scalar sz)
{
	vmemset(m, 0, sizeof *m);
	m->a.x = sx; m->b.y = sy; m->c.z = sz; m->d.w = VP(1);
}

VMATHDEF void mat4_eq_shear(mat4 *m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy)
{
	m->a.x = VP(1); m->a.y = xy;    m->a.z = xz;    m->a.w = 0;
	m->b.x = yx;    m->b.y = VP(1); m->b.z = yz;    m->b.w = 0;
	m->c.x = zx;    m->c.y = zy;    m->c.z = VP(1); m->c.w = 0;
	m->d.x = 0; m->d.y = 0; m->d.z = 0; m->d.w = VP(1);
}

VMATHDEF void mat4_eq_trans(mat4 *m, scalar tx, scalar ty, scalar tz)
{
	vmemset(m, 0, sizeof *m);
	m->a.x = m->b.y = m->c.z = m->d.w = VP(1);
	m->a.w = tx; m->b.w = ty; m->c.w = tz;
}

VMATHDEF void mat4_setarr(mat4 *m, scalar *s)
{
	vmemcpy(m, s, sizeof *m);
}

VMATHDEF void mat4_copy(mat4 *r, mat4 *m)
{
	vmemcpy(r, m, sizeof *m);
}

VMATHDEF void mat4_from_mat3(mat4 *r, mat3 *m)
{
	vec4_from_vec3(&r->a, &m->a);
	vec4_from_vec3(&r->b, &m->b);
	vec4_from_vec3(&r->c, &m->c);
	vec4_setcomp(&r->d, 0, 0, 0, VP(1));
}

VMATHDEF void mat4_from_mat3x4(mat4 *r, mat3x4 *m)
{
	vec4_copy(&r->a, &m->a);
	vec4_copy(&r->b, &m->b);
	vec4_copy(&r->c, &m->c);
	vec4_setcomp(&r->d, 0, 0, 0, VP(1));
}

VMATHDEF void mat4_setrows(mat4 *m, vec4 *ra, vec4 *rb, vec4 *rc, vec4 *rd)
{
	vec4_copy(&m->a, ra);
	vec4_copy(&m->b, rb);
	vec4_copy(&m->c, rc);
	vec4_copy(&m->d, rd);
}

VMATHDEF void mat4_setcols(mat4 *m, vec4 *cx, vec4 *cy, vec4 *cz, vec4 *cw)
{
	vec4_setcomp(&m->a, cx->x, cy->x, cz->x, cw->x);
	vec4_setcomp(&m->b, cx->y, cy->y, cz->y, cw->y);
	vec4_setcomp(&m->c, cx->z, cy->z, cz->z, cw->z);
	vec4_setcomp(&m->d, cx->w, cy->w, cz->w, cw->w);
}

VMATHDEF void mat4_setrowa(mat4 *m, vec4 *v) { vec4_copy(&m->a, v); }
VMATHDEF void mat4_setrowb(mat4 *m, vec4 *v) { vec4_copy(&m->b, v); }
VMATHDEF void mat4_setrowc(mat4 *m, vec4 *v) { vec4_copy(&m->c, v); }
VMATHDEF void mat4_setrowd(mat4 *m, vec4 *v) { vec4_copy(&m->d, v); }

VMATHDEF void mat4_setcolx(mat4 *m, vec4 *v)
{
	m->a.x = v->x; m->b.x = v->y; m->c.x = v->z; m->d.x = v->w;
}

VMATHDEF void mat4_setcoly(mat4 *m, vec4 *v)
{
	m->a.y = v->x; m->b.y = v->y; m->c.y = v->z; m->d.y = v->w;
}

VMATHDEF void mat4_setcolz(mat4 *m, vec4 *v)
{
	m->a.z = v->x; m->b.z = v->y; m->c.z = v->z; m->d.z = v->w;
}

VMATHDEF void mat4_setcolw(mat4 *m, vec4 *v)
{
	m->a.w = v->x; m->b.w = v->y; m->c.w = v->z; m->d.w = v->w;
}

VMATHDEF void mat4_rowa(vec4 *r, mat4 *m) { vec4_copy(r, &m->a); }
VMATHDEF void mat4_rowb(vec4 *r, mat4 *m) { vec4_copy(r, &m->b); }
VMATHDEF void mat4_rowc(vec4 *r, mat4 *m) { vec4_copy(r, &m->c); }
VMATHDEF void mat4_rowd(vec4 *r, mat4 *m) { vec4_copy(r, &m->d); }

VMATHDEF void mat4_colx(vec4 *r, mat4 *m) { vec4_setcomp(r, m->a.x, m->b.x, m->c.x, m->d.x); }
VMATHDEF void mat4_coly(vec4 *r, mat4 *m) { vec4_setcomp(r, m->a.y, m->b.y, m->c.y, m->d.y); }
VMATHDEF void mat4_colz(vec4 *r, mat4 *m) { vec4_setcomp(r, m->a.z, m->b.z, m->c.z, m->d.z); }
VMATHDEF void mat4_colw(vec4 *r, mat4 *m) { vec4_setcomp(r, m->a.w, m->b.w, m->c.w, m->d.w); }

VMATHDEF void mat4_smul(mat4 *r, scalar s, mat4 *m)
{
	vec4_smul(&r->a, s, &m->a);
	vec4_smul(&r->b, s, &m->b);
	vec4_smul(&r->c, s, &m->c);
	vec4_smul(&r->d, s, &m->d);
}

VMATHDEF void mat4_mulrows(mat4 *r, mat4 *m, scalar a, scalar b, scalar c, scalar d)
{
	vec4_smul(&r->a, a, &m->a);
	vec4_smul(&r->b, b, &m->b);
	vec4_smul(&r->c, c, &m->c);
	vec4_smul(&r->d, d, &m->d);
}

VMATHDEF void mat4_mulrowv(mat4 *r, mat4 *m, vec4 *v)
{
	vec4_smul(&r->a, v->x, &m->a);
	vec4_smul(&r->b, v->y, &m->b);
	vec4_smul(&r->c, v->z, &m->c);
	vec4_smul(&r->d, v->w, &m->d);
}

VMATHDEF void mat4_negate(mat4 *r, mat4 *m)
{
	vec4_negate(&r->a, &m->a);
	vec4_negate(&r->b, &m->b);
	vec4_negate(&r->c, &m->c);
	vec4_negate(&r->d, &m->d);
}

VMATHDEF void mat4_add(mat4 *r, mat4 *f, mat4 *g)
{
	vec4_add(&r->a, &f->a, &g->a);
	vec4_add(&r->b, &f->b, &g->b);
	vec4_add(&r->c, &f->c, &g->c);
	vec4_add(&r->d, &f->d, &g->d);
}

VMATHDEF void mat4_sub(mat4 *r, mat4 *f, mat4 *g)
{
	vec4_sub(&r->a, &f->a, &g->a);
	vec4_sub(&r->b, &f->b, &g->b);
	vec4_sub(&r->c, &f->c, &g->c);
	vec4_sub(&r->d, &f->d, &g->d);
}

VMATHDEF void mat4_tmul(mat4 *r, mat4 *f, mat4 *g)
{
	vec4_tmul(&r->a, &f->a, &g->a);
	vec4_tmul(&r->b, &f->b, &g->b);
	vec4_tmul(&r->c, &f->c, &g->c);
	vec4_tmul(&r->d, &f->d, &g->d);
}

VMATHDEF void _vec4_mul_mat4(vec4 *r, vec4 *v, mat4 *m)
{
	r->x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z + m->d.x * v->w;
	r->y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z + m->d.y * v->w;
	r->z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z + m->d.z * v->w;
	r->w = m->a.w * v->x + m->b.w * v->y + m->c.w * v->z + m->d.w * v->w;
}

VMATHDEF void mat4_mul(mat4 *r, mat4 *f, mat4 *g)
{
	mat4 m;
	_vec4_mul_mat4(&m.a, &f->a, g);
	_vec4_mul_mat4(&m.b, &f->b, g);
	_vec4_mul_mat4(&m.c, &f->c, g);
	_vec4_mul_mat4(&m.d, &f->d, g);
	vmemcpy(r, &m, sizeof m);
}

VMATHDEF void mat4_ma(mat4 *r, mat4 *f, scalar t, mat4 *g)
{
	vec4_ma(&r->a, &f->a, t, &g->a);
	vec4_ma(&r->b, &f->b, t, &g->b);
	vec4_ma(&r->c, &f->c, t, &g->c);
}

VMATHDEF void mat4_combine(mat4 *r, scalar s, mat4 *f, scalar t, mat4 *g)
{
	vec4_combine(&r->a, s, &f->a, t, &g->a);
	vec4_combine(&r->b, s, &f->b, t, &g->b);
	vec4_combine(&r->c, s, &f->c, t, &g->c);
}

VMATHDEF void mat4_lerp(mat4 *r, mat4 *f, mat4 *g, scalar t)
{
	scalar s = 1 - t;
	vec4_combine(&r->a, s, &f->a, t, &g->a);
	vec4_combine(&r->b, s, &f->b, t, &g->b);
	vec4_combine(&r->c, s, &f->c, t, &g->c);
}

VMATHDEF void vec4_mul_mat4(vec4 *r, vec4 *v, mat4 *m)
{
	scalar x, y, z, w;
	x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z + m->d.x * v->w;
	y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z + m->d.y * v->w;
	z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z + m->d.z * v->w;
	w = m->a.w * v->x + m->b.w * v->y + m->c.w * v->z + m->d.w * v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

VMATHDEF void mat4_mul_vec4(vec4 *r, mat4 *m, vec4 *v)
{
	scalar x, y, z, w;
	x = m->a.x * v->x + m->a.y * v->y + m->a.z * v->z + m->a.w * v->w;
	y = m->b.x * v->x + m->b.y * v->y + m->b.z * v->z + m->b.w * v->w;
	z = m->c.x * v->x + m->c.y * v->y + m->c.z * v->z + m->c.w * v->w;
	w = m->d.x * v->x + m->d.y * v->y + m->d.z * v->z + m->d.w * v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

VMATHDEF void mat4_mul_vec3(vec3 *r, mat4 *m, vec3 *v)
{
	scalar x, y, z;
	x = m->a.x * v->x + m->a.y * v->y + m->a.z * v->z + m->a.w;
	y = m->b.x * v->x + m->b.y * v->y + m->b.z * v->z + m->b.w;
	z = m->c.x * v->x + m->c.y * v->y + m->c.z * v->z + m->c.w;
	r->x = x; r->y = y; r->z = z;
}

VMATHDEF void mat4_transpose(mat4 *r, mat4 *m)
{
	scalar t;
	t = m->a.y; r->a.y = m->b.x; r->b.x = t;
	t = m->a.z; r->a.z = m->c.x; r->c.x = t;
	t = m->a.w; r->a.w = m->d.x; r->d.x = t;
	
	t = m->b.z; r->b.z = m->c.y; r->c.y = t;
	t = m->b.w; r->b.w = m->d.y; r->d.y = t;
	
	t = m->c.w; r->c.w = m->d.z; r->d.z = t;
	
	r->a.x = m->a.x; r->b.y = m->b.y; r->c.z = m->c.z; r->d.w = m->d.w;
}

VMATHDEF void mat4_inverse(mat4 *r, mat4 *m)
{
	scalar s0, s1, s2, s3, s4, s5;
	scalar c0, c1, c2, c3, c4, c5;
	scalar invdet;
	
	s0 = m->a.x * m->b.y - m->a.y * m->b.x;
	s1 = m->a.x * m->c.y - m->a.y * m->c.x;
	s2 = m->a.x * m->d.y - m->a.y * m->d.x;
	s3 = m->b.x * m->c.y - m->b.y * m->c.x;
	s4 = m->b.x * m->d.y - m->b.y * m->d.x;
	s5 = m->c.x * m->d.y - m->c.y * m->d.x;

	c0 = m->a.z * m->b.w - m->a.w * m->b.z;
	c1 = m->a.z * m->c.w - m->a.w * m->c.z;
	c2 = m->a.z * m->d.w - m->a.w * m->d.z;
	c3 = m->b.z * m->c.w - m->b.w * m->c.z;
	c4 = m->b.z * m->d.w - m->b.w * m->d.z;
	c5 = m->c.z * m->d.w - m->c.w * m->d.z;
	
	invdet = VP(1) / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
	
	r->a.x =  (m->b.y * c5 - m->c.y * c4 + m->d.y * c3) * invdet;
	r->a.y = -(m->a.y * c5 - m->c.y * c2 + m->d.y * c1) * invdet;
	r->a.z =  (m->a.y * c4 - m->b.y * c2 + m->d.y * c0) * invdet;
	r->a.w = -(m->a.y * c3 - m->b.y * c1 + m->c.y * c0) * invdet;
	
	r->b.x = -(m->b.x * c5 - m->c.x * c4 + m->d.x * c3) * invdet;
	r->b.y =  (m->a.x * c5 - m->c.x * c2 + m->d.x * c1) * invdet;
	r->b.z = -(m->a.x * c4 - m->b.x * c2 + m->d.x * c0) * invdet;
	r->b.w =  (m->a.x * c3 - m->b.x * c1 + m->c.x * c0) * invdet;
	
	r->c.x =  (m->b.w * s5 - m->c.w * s4 + m->d.w * s3) * invdet;
	r->c.y = -(m->a.w * s5 - m->c.w * s2 + m->d.w * s1) * invdet;
	r->c.z =  (m->a.w * s4 - m->b.w * s2 + m->d.w * s0) * invdet;
	r->c.w = -(m->a.w * s3 - m->b.w * s1 + m->c.w * s0) * invdet;
	
	r->d.x = -(m->b.z * s5 - m->c.z * s4 + m->d.z * s3) * invdet;
	r->d.y =  (m->a.z * s5 - m->c.z * s2 + m->d.z * s1) * invdet;
	r->d.z = -(m->a.z * s4 - m->b.z * s2 + m->d.z * s0) * invdet;
	r->d.w =  (m->a.z * s3 - m->b.z * s1 + m->c.z * s0) * invdet;
}

VMATHDEF void mat4_vtrace(vec4 *v, mat4 *m)
{
	v->x = m->a.x; v->y = m->b.y;
	v->z = m->c.z; v->w = m->d.w;
}

VMATHDEF scalar mat4_trace(mat4 *m)
{
	return m->a.x + m->b.y + m->c.z + m->d.w;
}

VMATHDEF scalar mat4_det3(mat4 *m)
{
	return
	  m->a.x * (m->b.y * m->c.z - m->b.z * m->c.y)
	- m->a.y * (m->b.x * m->c.z - m->b.z * m->c.x)
	+ m->a.z * (m->b.x * m->c.y - m->b.y * m->c.x);
}

VMATHDEF scalar mat4_det(mat4 *m)
{
	return 
	  (m->a.x * m->b.y - m->a.y * m->b.x) * (m->c.z * m->d.w - m->c.w * m->d.z)
	- (m->a.x * m->c.y - m->a.y * m->c.x) * (m->b.z * m->d.w - m->b.w * m->d.z)
	+ (m->a.x * m->d.y - m->a.y * m->d.x) * (m->b.z * m->c.w - m->b.w * m->c.z)
	+ (m->b.x * m->c.y - m->b.y * m->c.x) * (m->a.z * m->d.w - m->a.w * m->d.z)
	- (m->b.x * m->d.y - m->b.y * m->d.x) * (m->a.z * m->c.w - m->a.w * m->c.z)
	+ (m->c.x * m->d.y - m->c.y * m->d.x) * (m->a.z * m->b.w - m->a.w * m->b.z);
}


// ---------------------
// --- mat3x4 common ---
// ---------------------
VMATHDEF void mat3x4_zero(mat3x4 *m)
{
	vmemset(m, 0, sizeof *m);
}

VMATHDEF void mat3x4_id(mat3x4 *m)
{
	vmemset(m, 0, sizeof *m);
	m->a.x = m->b.y = m->c.z = VP(1);
}

VMATHDEF void mat3x4_t(mat3x4 *m)
{
	scalar t;
	t = m->a.y; m->a.y = m->b.x; m->b.x = t;
	t = m->a.z; m->a.z = m->c.x; m->c.x = t;
	
	t = m->b.z; m->b.z = m->c.y; m->c.y = t;
}

VMATHDEF void mat3x4_eq_rotx(mat3x4 *m, scalar a)
{
	vmemset(m, 0, sizeof *m);
	scalar c = vcos(a), s = vsin(a);
	m->a.x = VP(1);
	m->b.y =  c; m->b.z = -s;
	m->c.y =  s; m->c.z =  c;
}

VMATHDEF void mat3x4_eq_roty(mat3x4 *m, scalar a)
{
	vmemset(m, 0, sizeof *m);
	scalar c = vcos(a), s = vsin(a);
	m->b.y = VP(1);
	m->a.x =  c; m->a.z = -s;
	m->c.x =  s; m->c.z =  c;
}

VMATHDEF void mat3x4_eq_rotz(mat3x4 *m, scalar a)
{
	vmemset(m, 0, sizeof *m);
	scalar c = vcos(a), s = vsin(a);
	m->c.z = VP(1);
	m->a.x =  c; m->a.y = -s;
	m->b.x =  s; m->b.y =  c;
}

VMATHDEF void mat3x4_eq_rotaxis(mat3x4 *m, vec3 *u, scalar a)
{
	scalar s = vsin(a), c = vcos(a);
	scalar d = 1 - c;

	scalar dxx = d * u->x * u->x;
	scalar dyy = d * u->y * u->y;
	scalar dzz = d * u->z * u->z;
	scalar dxy = d * u->x * u->y;
	scalar dxz = d * u->x * u->z;
	scalar dyz = d * u->y * u->z;
	
	scalar sx = s * u->x, sy = s * u->y, sz = s * u->z;

	m->a.x = c + dxx;  m->a.y = dxy - sz; m->a.z = dxz + sy; m->a.w = 0;
	m->b.x = dxy + sz; m->b.y = c + dyy;  m->b.z = dyz - sx; m->b.w = 0;
	m->c.x = dxz - sy; m->c.y = dyz + sx; m->c.z = c + dzz;  m->c.w = 0;
}

VMATHDEF void mat3x4_eq_scale(mat3x4 *m, scalar sx, scalar sy, scalar sz)
{
	vmemset(m, 0, sizeof *m);
	m->a.x = sx; m->b.y = sy; m->c.z = sz;
}

VMATHDEF void mat3x4_eq_shear(mat3x4 *m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy)
{
	m->a.x = VP(1); m->a.y = xy;    m->a.z = xz;    m->a.w = 0;
	m->b.x = yx;    m->b.y = VP(1); m->b.z = yz;    m->b.w = 0;
	m->c.x = zx;    m->c.y = zy;    m->c.z = VP(1); m->c.w = 0;
}

VMATHDEF void mat3x4_eq_trans(mat3x4 *m, scalar tx, scalar ty, scalar tz)
{
	vmemset(m, 0, sizeof *m);
	m->a.x = m->b.y = m->c.z = VP(1);
	m->a.w = tx; m->b.w = ty; m->c.w = tz;
}

VMATHDEF void mat3x4_setarr(mat3x4 *m, scalar *s)
{
	vmemcpy(m, s, sizeof *m);
}

VMATHDEF void mat3x4_copy(mat3x4 *r, mat3x4 *m)
{
	vmemcpy(r, m, sizeof *m);
}

VMATHDEF void mat3x4_from_mat3(mat3x4 *r, mat3 *m)
{
	vec4_from_vec3(&r->a, &m->a);
	vec4_from_vec3(&r->b, &m->b);
	vec4_from_vec3(&r->c, &m->c);
}

VMATHDEF void mat3x4_from_mat4(mat3x4 *r, mat4 *m)
{
	vec4_copy(&r->a, &m->a);
	vec4_copy(&r->b, &m->b);
	vec4_copy(&r->c, &m->c);
}

VMATHDEF void mat3x4_setrows(mat3x4 *m, vec4 *ra, vec4 *rb, vec4 *rc)
{
	vec4_copy(&m->a, ra);
	vec4_copy(&m->b, rb);
	vec4_copy(&m->c, rc);
}

VMATHDEF void mat3x4_setcols(mat3x4 *m, vec3 *cx, vec3 *cy, vec3 *cz, vec3 *cw)
{
	vec4_setcomp(&m->a, cx->x, cy->x, cz->x, cw->x);
	vec4_setcomp(&m->b, cx->y, cy->y, cz->y, cw->y);
	vec4_setcomp(&m->c, cx->z, cy->z, cz->z, cw->z);
}

VMATHDEF void mat3x4_setrowa(mat3x4 *m, vec4 *v) { vec4_copy(&m->a, v); }
VMATHDEF void mat3x4_setrowb(mat3x4 *m, vec4 *v) { vec4_copy(&m->b, v); }
VMATHDEF void mat3x4_setrowc(mat3x4 *m, vec4 *v) { vec4_copy(&m->c, v); }

VMATHDEF void mat3x4_setcolx(mat3x4 *m, vec3 *v)
{
	m->a.x = v->x; m->b.x = v->y; m->c.x = v->z;
}

VMATHDEF void mat3x4_setcoly(mat3x4 *m, vec3 *v)
{
	m->a.y = v->x; m->b.y = v->y; m->c.y = v->z;
}

VMATHDEF void mat3x4_setcolz(mat3x4 *m, vec3 *v)
{
	m->a.z = v->x; m->b.z = v->y; m->c.z = v->z;
}

VMATHDEF void mat3x4_setcolw(mat3x4 *m, vec3 *v)
{
	m->a.w = v->x; m->b.w = v->y; m->c.w = v->z;
}

VMATHDEF void mat3x4_rowa(vec4 *r, mat3x4 *m) { vec4_copy(r, &m->a); }
VMATHDEF void mat3x4_rowb(vec4 *r, mat3x4 *m) { vec4_copy(r, &m->b); }
VMATHDEF void mat3x4_rowc(vec4 *r, mat3x4 *m) { vec4_copy(r, &m->c); }

VMATHDEF void mat3x4_colx(vec3 *r, mat3x4 *m) { vec3_setcomp(r, m->a.x, m->b.x, m->c.x); } //d.x = 0
VMATHDEF void mat3x4_coly(vec3 *r, mat3x4 *m) { vec3_setcomp(r, m->a.y, m->b.y, m->c.y); } //d.y = 0
VMATHDEF void mat3x4_colz(vec3 *r, mat3x4 *m) { vec3_setcomp(r, m->a.z, m->b.z, m->c.z); } //d.z = 0
VMATHDEF void mat3x4_colw(vec3 *r, mat3x4 *m) { vec3_setcomp(r, m->a.w, m->b.w, m->c.w); } //d.w = 1

VMATHDEF void mat3x4_smul(mat3x4 *r, scalar s, mat3x4 *m)
{
	vec4_smul(&r->a, s, &m->a);
	vec4_smul(&r->b, s, &m->b);
	vec4_smul(&r->c, s, &m->c);
}

VMATHDEF void mat3x4_mulrows(mat3x4 *r, mat3x4 *m, scalar a, scalar b, scalar c)
{
	vec4_smul(&r->a, a, &m->a);
	vec4_smul(&r->b, b, &m->b);
	vec4_smul(&r->c, c, &m->c);
}

VMATHDEF void mat3x4_mulrowv(mat3x4 *r, mat3x4 *m, vec3 *v)
{
	vec4_smul(&r->a, v->x, &m->a);
	vec4_smul(&r->b, v->y, &m->b);
	vec4_smul(&r->c, v->z, &m->c);
}

VMATHDEF void mat3x4_negate(mat3x4 *r, mat3x4 *m)
{
	vec4_negate(&r->a, &m->a);
	vec4_negate(&r->b, &m->b);
	vec4_negate(&r->c, &m->c);
}

VMATHDEF void mat3x4_add(mat3x4 *r, mat3x4 *f, mat3x4 *g)
{
	vec4_add(&r->a, &f->a, &g->a);
	vec4_add(&r->b, &f->b, &g->b);
	vec4_add(&r->c, &f->c, &g->c);
}

VMATHDEF void mat3x4_sub(mat3x4 *r, mat3x4 *f, mat3x4 *g)
{
	vec4_sub(&r->a, &f->a, &g->a);
	vec4_sub(&r->b, &f->b, &g->b);
	vec4_sub(&r->c, &f->c, &g->c);
}

VMATHDEF void mat3x4_tmul(mat3x4 *r, mat3x4 *f, mat3x4 *g)
{
	vec4_tmul(&r->a, &f->a, &g->a);
	vec4_tmul(&r->b, &f->b, &g->b);
	vec4_tmul(&r->c, &f->c, &g->c);
}

VMATHDEF void _vec4_mul_mat3x4(vec4 *r, vec4 *v, mat3x4 *m)
{
	r->x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z;
	r->y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z;
	r->z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z;
	r->w = m->a.w * v->x + m->b.w * v->y + m->c.w * v->z + v->w;
}

VMATHDEF void mat3x4_mul(mat3x4 *r, mat3x4 *f, mat3x4 *g)
{
	mat3x4 m;
	_vec4_mul_mat3x4(&m.a, &f->a, g);
	_vec4_mul_mat3x4(&m.b, &f->b, g);
	_vec4_mul_mat3x4(&m.c, &f->c, g);
	vmemcpy(r, &m, sizeof m);
}

VMATHDEF void mat3x4_ma(mat3x4 *r, mat3x4 *f, scalar t, mat3x4 *g)
{
	vec4_ma(&r->a, &f->a, t, &g->a);
	vec4_ma(&r->b, &f->b, t, &g->b);
	vec4_ma(&r->c, &f->c, t, &g->c);
}

VMATHDEF void mat3x4_combine(mat3x4 *r, scalar s, mat3x4 *f, scalar t, mat3x4 *g)
{
	vec4_combine(&r->a, s, &f->a, t, &g->a);
	vec4_combine(&r->b, s, &f->b, t, &g->b);
	vec4_combine(&r->c, s, &f->c, t, &g->c);
}

VMATHDEF void mat3x4_lerp(mat3x4 *r, mat3x4 *f, mat3x4 *g, scalar t)
{
	scalar s = 1 - t;
	vec4_combine(&r->a, s, &f->a, t, &g->a);
	vec4_combine(&r->b, s, &f->b, t, &g->b);
	vec4_combine(&r->c, s, &f->c, t, &g->c);
}

VMATHDEF void vec4_mul_mat3x4(vec4 *r, vec4 *v, mat3x4 *m)
{
	scalar x, y, z, w;
	x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z;
	y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z;
	z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z;
	w = m->a.w * v->x + m->b.w * v->y + m->c.w * v->z + v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

VMATHDEF void mat3x4_mul_vec4(vec4 *r, mat3x4 *m, vec4 *v)
{
	scalar x, y, z, w;
	x = m->a.x * v->x + m->a.y * v->y + m->a.z * v->z + m->a.w * v->w;
	y = m->b.x * v->x + m->b.y * v->y + m->b.z * v->z + m->b.w * v->w;
	z = m->c.x * v->x + m->c.y * v->y + m->c.z * v->z + m->c.w * v->w;
	w = v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

VMATHDEF void mat3x4_mul_vec3(vec3 *r, mat3x4 *m, vec3 *v)
{
	scalar x, y, z;
	x = m->a.x * v->x + m->a.y * v->y + m->a.z * v->z + m->a.w;
	y = m->b.x * v->x + m->b.y * v->y + m->b.z * v->z + m->b.w;
	z = m->c.x * v->x + m->c.y * v->y + m->c.z * v->z + m->c.w;
	r->x = x; r->y = y; r->z = z;
}

VMATHDEF void mat3x4_transpose(mat3x4 *r, mat3x4 *m)
{
	scalar t;
	t = m->a.y; r->a.y = m->b.x; r->b.x = t;
	t = m->a.z; r->a.z = m->c.x; r->c.x = t;
	
	t = m->b.z; r->b.z = m->c.y; r->c.y = t;
	
	r->a.x = m->a.x; r->b.y = m->b.y; r->c.z = m->c.z;
	r->a.w = m->a.w; r->b.w = m->b.w; r->c.w = m->c.w;
}

VMATHDEF void mat3x4_inverse(mat3x4 *r, mat3x4 *m)
{
	scalar s0, s1, s3;
	scalar c0, c1, c3;
	scalar invdet;
	
	s0 = m->a.x * m->b.y - m->a.y * m->b.x;
	s1 = m->a.x * m->c.y - m->a.y * m->c.x;
	s3 = m->b.x * m->c.y - m->b.y * m->c.x;
	
	c0 = m->a.z * m->b.w - m->a.w * m->b.z;
	c1 = m->a.z * m->c.w - m->a.w * m->c.z;
	c3 = m->b.z * m->c.w - m->b.w * m->c.z;
	
	invdet = VP(1) / (s0 * m->c.z - s1 * m->b.z + s3 * m->a.z);
	
	r->a.x =  (m->b.y * m->c.z - m->c.y * m->b.z) * invdet;
	r->a.y = -(m->a.y * m->c.z - m->c.y * m->a.z) * invdet;
	r->a.z =  (m->a.y * m->b.z - m->b.y * m->a.z) * invdet;
	r->a.w = -(m->a.y * c3 - m->b.y * c1 + m->c.y * c0) * invdet;
	
	r->b.x = -(m->b.x * m->c.z - m->c.x * m->b.z) * invdet;
	r->b.y =  (m->a.x * m->c.z - m->c.x * m->a.z) * invdet;
	r->b.z = -(m->a.x * m->b.z - m->b.x * m->a.z) * invdet;
	r->b.w =  (m->a.x * c3 - m->b.x * c1 + m->c.x * c0) * invdet;
	
	r->c.x =  s3 * invdet;
	r->c.y = -s1 * invdet;
	r->c.z =  s0 * invdet;
	r->c.w = -(m->a.w * s3 - m->b.w * s1 + m->c.w * s0) * invdet;
}

VMATHDEF void mat3x4_vtrace(vec3 *v, mat3x4 *m)
{
	v->x = m->a.x; v->y = m->b.y; v->z = m->c.z;
}

VMATHDEF scalar mat3x4_trace(mat3x4 *m)
{
	return m->a.x + m->b.y + m->c.z;
}

VMATHDEF scalar mat3x4_det(mat3x4 *m)
{
	return
	  m->a.x * (m->b.y * m->c.z - m->b.z * m->c.y)
	- m->a.y * (m->b.x * m->c.z - m->b.z * m->c.x)
	+ m->a.z * (m->b.x * m->c.y - m->b.y * m->c.x);
}


// ---------------------
// --- mat3x3 common ---
// ---------------------
VMATHDEF void mat3_zero(mat3 *m)
{
	vmemset(m, 0, sizeof *m);
}

VMATHDEF void mat3_id(mat3 *m)
{
	vmemset(m, 0, sizeof *m);
	m->a.x = m->b.y = m->c.z = VP(1);
}

VMATHDEF void mat3_t(mat3 *m)
{
	scalar t;
	t = m->a.y; m->a.y = m->b.x; m->b.x = t;
	t = m->a.z; m->a.z = m->c.x; m->c.x = t;
	
	t = m->b.z; m->b.z = m->c.y; m->c.y = t;
}

VMATHDEF void mat3_eq_rotx(mat3 *m, scalar a)
{
	vmemset(m, 0, sizeof *m);
	scalar c = vcos(a), s = vsin(a);
	m->a.x = VP(1);
	m->b.y =  c; m->b.z = -s;
	m->c.y =  s; m->c.z =  c;
}

VMATHDEF void mat3_eq_roty(mat3 *m, scalar a)
{
	vmemset(m, 0, sizeof *m);
	scalar c = vcos(a), s = vsin(a);
	m->b.y = VP(1);
	m->a.x =  c; m->a.z = -s;
	m->c.x =  s; m->c.z =  c;
}

VMATHDEF void mat3_eq_rotz(mat3 *m, scalar a)
{
	vmemset(m, 0, sizeof *m);
	scalar c = vcos(a), s = vsin(a);
	m->c.z = VP(1);
	m->a.x =  c; m->a.y = -s;
	m->b.x =  s; m->b.y =  c;
}

VMATHDEF void mat3_eq_rotaxis(mat3 *m, vec3 *u, scalar a)
{
	scalar s = vsin(a), c = vcos(a);
	scalar d = 1 - c;

	scalar dxx = d * u->x * u->x;
	scalar dyy = d * u->y * u->y;
	scalar dzz = d * u->z * u->z;
	scalar dxy = d * u->x * u->y;
	scalar dxz = d * u->x * u->z;
	scalar dyz = d * u->y * u->z;
	
	scalar sx = s * u->x, sy = s * u->y, sz = s * u->z;

	m->a.x = c + dxx;  m->a.y = dxy - sz; m->a.z = dxz + sy;
	m->b.x = dxy + sz; m->b.y = c + dyy;  m->b.z = dyz - sx;
	m->c.x = dxz - sy; m->c.y = dyz + sx; m->c.z = c + dzz;
}

VMATHDEF void mat3_eq_trans2d(mat3 *m, scalar tx, scalar ty)
{
	vmemset(m, 0, sizeof *m);
	m->a.x = m->b.y = m->c.z = VP(1);
	m->a.z = tx; m->b.z = ty;
}

VMATHDEF void mat3_eq_scale(mat3 *m, scalar sx, scalar sy, scalar sz)
{
	vmemset(m, 0, sizeof *m);
	m->a.x = sx; m->b.y = sy; m->c.z = sz;
}

VMATHDEF void mat3_eq_scale2d(mat3 *m, scalar sx, scalar sy)
{
	vmemset(m, 0, sizeof *m);
	m->a.x = sx; m->b.y = sy; m->c.z = VP(1);
}

VMATHDEF void mat3_eq_shear(mat3 *m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy)
{
	m->a.x = VP(1); m->a.y = xy;    m->a.z = xz;
	m->b.x = yx;    m->b.y = VP(1); m->b.z = yz;
	m->c.x = zx;    m->c.y = zy;    m->c.z = VP(1);
}

VMATHDEF void mat3_eq_shear2d(mat3 *m, scalar x, scalar y)
{
	m->a.x = VP(1); m->a.y = x;     m->a.z = 0;
	m->b.x = y;     m->b.y = VP(1); m->b.z = 0;
	m->c.x = 0; m->c.y = 0; m->c.z = VP(1);
}

VMATHDEF void mat3_setarr(mat3 *m, scalar *s)
{
	vmemcpy(m, s, sizeof *m);
}

VMATHDEF void mat3_copy(mat3 *r, mat3 *m)
{
	vmemcpy(r, m, sizeof *m);
}

VMATHDEF void mat3_from_mat4(mat3 *r, mat4 *m)
{
	vec3_from_vec4(&r->a, &m->a);
	vec3_from_vec4(&r->b, &m->b);
	vec3_from_vec4(&r->c, &m->c);
}

VMATHDEF void mat3_from_mat3x4(mat3 *r, mat3x4 *m)
{
	vec3_from_vec4(&r->a, &m->a);
	vec3_from_vec4(&r->b, &m->b);
	vec3_from_vec4(&r->c, &m->c);
}

VMATHDEF void mat3_setrows(mat3 *m, vec3 *ra, vec3 *rb, vec3 *rc)
{
	vec3_copy(&m->a, ra);
	vec3_copy(&m->b, rb);
	vec3_copy(&m->c, rc);
}

VMATHDEF void mat3_setcols(mat3 *m, vec3 *cx, vec3 *cy, vec3 *cz)
{
	vec3_setcomp(&m->a, cx->x, cy->x, cz->x);
	vec3_setcomp(&m->b, cx->y, cy->y, cz->y);
	vec3_setcomp(&m->c, cx->z, cy->z, cz->z);
}

VMATHDEF void mat3_setrowa(mat3 *m, vec3 *v) { vec3_copy(&m->a, v); }
VMATHDEF void mat3_setrowb(mat3 *m, vec3 *v) { vec3_copy(&m->b, v); }
VMATHDEF void mat3_setrowc(mat3 *m, vec3 *v) { vec3_copy(&m->c, v); }

VMATHDEF void mat3_setcolx(mat3 *m, vec3 *v)
{
	m->a.x = v->x; m->b.x = v->y; m->c.x = v->z;
}

VMATHDEF void mat3_setcoly(mat3 *m, vec3 *v)
{
	m->a.y = v->x; m->b.y = v->y; m->c.y = v->z;
}

VMATHDEF void mat3_setcolz(mat3 *m, vec3 *v)
{
	m->a.z = v->x; m->b.z = v->y; m->c.z = v->z;
}

VMATHDEF void mat3_rowa(vec3 *r, mat3 *m) { vec3_copy(r, &m->a); }
VMATHDEF void mat3_rowb(vec3 *r, mat3 *m) { vec3_copy(r, &m->b); }
VMATHDEF void mat3_rowc(vec3 *r, mat3 *m) { vec3_copy(r, &m->c); }

VMATHDEF void mat3_colx(vec3 *r, mat3 *m) { vec3_setcomp(r, m->a.x, m->b.x, m->c.x); }
VMATHDEF void mat3_coly(vec3 *r, mat3 *m) { vec3_setcomp(r, m->a.y, m->b.y, m->c.y); }
VMATHDEF void mat3_colz(vec3 *r, mat3 *m) { vec3_setcomp(r, m->a.z, m->b.z, m->c.z); }

VMATHDEF void mat3_smul(mat3 *r, scalar s, mat3 *m)
{
	vec3_smul(&r->a, s, &m->a);
	vec3_smul(&r->b, s, &m->b);
	vec3_smul(&r->c, s, &m->c);
}

VMATHDEF void mat3_mulrows(mat3 *r, mat3 *m, scalar a, scalar b, scalar c)
{
	vec3_smul(&r->a, a, &m->a);
	vec3_smul(&r->b, b, &m->b);
	vec3_smul(&r->c, c, &m->c);
}

VMATHDEF void mat3_mulrowv(mat3 *r, mat3 *m, vec3 *v)
{
	vec3_smul(&r->a, v->x, &m->a);
	vec3_smul(&r->b, v->y, &m->b);
	vec3_smul(&r->c, v->z, &m->c);
}

VMATHDEF void mat3_negate(mat3 *r, mat3 *m)
{
	vec3_negate(&r->a, &m->a);
	vec3_negate(&r->b, &m->b);
	vec3_negate(&r->c, &m->c);
}

VMATHDEF void mat3_add(mat3 *r, mat3 *f, mat3 *g)
{
	vec3_add(&r->a, &f->a, &g->a);
	vec3_add(&r->b, &f->b, &g->b);
	vec3_add(&r->c, &f->c, &g->c);
}

VMATHDEF void mat3_sub(mat3 *r, mat3 *f, mat3 *g)
{
	vec3_sub(&r->a, &f->a, &g->a);
	vec3_sub(&r->b, &f->b, &g->b);
	vec3_sub(&r->c, &f->c, &g->c);
}

VMATHDEF void mat3_tmul(mat3 *r, mat3 *f, mat3 *g)
{
	vec3_tmul(&r->a, &f->a, &g->a);
	vec3_tmul(&r->b, &f->b, &g->b);
	vec3_tmul(&r->c, &f->c, &g->c);
}

VMATHDEF void _vec3_mul_mat3(vec3 *r, vec3 *v, mat3 *m)
{
	r->x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z;
	r->y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z;
	r->z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z;
}

VMATHDEF void mat3_mul(mat3 *r, mat3 *f, mat3 *g)
{
	mat3 m;
	_vec3_mul_mat3(&m.a, &f->a, g);
	_vec3_mul_mat3(&m.b, &f->b, g);
	_vec3_mul_mat3(&m.c, &f->c, g);
	vmemcpy(r, &m, sizeof m);
}

VMATHDEF void mat3_ma(mat3 *r, mat3 *f, scalar t, mat3 *g)
{
	vec3_ma(&r->a, &f->a, t, &g->a);
	vec3_ma(&r->b, &f->b, t, &g->b);
	vec3_ma(&r->c, &f->c, t, &g->c);
}

VMATHDEF void mat3_combine(mat3 *r, scalar s, mat3 *f, scalar t, mat3 *g)
{
	vec3_combine(&r->a, s, &f->a, t, &g->a);
	vec3_combine(&r->b, s, &f->b, t, &g->b);
	vec3_combine(&r->c, s, &f->c, t, &g->c);
}

VMATHDEF void mat3_lerp(mat3 *r, mat3 *f, mat3 *g, scalar t)
{
	scalar s = VP(1) - t;
	vec3_combine(&r->a, s, &f->a, t, &g->a);
	vec3_combine(&r->b, s, &f->b, t, &g->b);
	vec3_combine(&r->c, s, &f->c, t, &g->c);
}

VMATHDEF void vec3_mul_mat3(vec3 *r, vec3 *v, mat3 *m)
{
	scalar x, y, z;
	x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z;
	y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z;
	z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z;
	r->x = x; r->y = y; r->z = z;
}

VMATHDEF void mat3_mul_vec3(vec3 *r, mat3 *m, vec3 *v)
{
	scalar x, y, z;
	x = m->a.x * v->x + m->a.y * v->y + m->a.z * v->z;
	y = m->b.x * v->x + m->b.y * v->y + m->b.z * v->z;
	z = m->c.x * v->x + m->c.y * v->y + m->c.z * v->z;
	r->x = x; r->y = y; r->z = z;
}

VMATHDEF void mat3_mul_vec2(vec2 *r, mat3 *m, vec2 *v)
{
	scalar x, y;
	x = m->a.x * v->x + m->a.y * v->y + m->a.z;
	y = m->b.x * v->x + m->b.y * v->y + m->b.z;
	r->x = x; r->y = y;
}

VMATHDEF void mat3_transpose(mat3 *r, mat3 *m)
{
	scalar t;
	t = m->a.y; r->a.y = m->b.x; r->b.x = t;
	t = m->a.z; r->a.z = m->c.x; r->c.x = t;
	
	t = m->b.z; r->b.z = m->c.y; r->c.y = t;
	
	r->a.x = m->a.x; r->b.y = m->b.y; r->c.z = m->c.z;
}

VMATHDEF void mat3_vtrace(vec3 *v, mat3 *m)
{
	v->x = m->a.x; v->y = m->b.y; v->z = m->c.z;
}

VMATHDEF scalar mat3_trace(mat3 *m)
{
	return m->a.x + m->b.y + m->c.z;
}

VMATHDEF scalar mat3_det(mat3 *m)
{
	return
	  m->a.x * (m->b.y * m->c.z - m->b.z * m->c.y)
	- m->a.y * (m->b.x * m->c.z - m->b.z * m->c.x)
	+ m->a.z * (m->b.x * m->c.y - m->b.y * m->c.x);
}

VMATHDEF void mat3_inverse(mat3 *r, mat3 *m)
{
	scalar az, bz, cz, invdet;
	az = m->b.x * m->c.y - m->b.y * m->c.x;
	bz = m->a.x * m->c.y - m->a.y * m->c.x;
	cz = m->a.x * m->b.y - m->a.y * m->b.x;
	
	invdet = VP(1) / (cz * m->c.z - bz * m->b.z + az * m->a.z);
	
	r->a.x =  (m->b.y * m->c.z - m->b.z * m->c.y) * invdet;
	r->a.y = -(m->a.y * m->c.z - m->a.z * m->c.y) * invdet;
	r->a.z =  (m->a.y * m->b.z - m->a.z * m->b.y) * invdet;
	
	r->b.x = -(m->b.x * m->c.z - m->b.z * m->c.x) * invdet;
	r->b.y =  (m->a.x * m->c.z - m->a.z * m->c.x) * invdet;
	r->b.z = -(m->a.x * m->b.z - m->a.z * m->b.x) * invdet;
	
	r->c.x =  az * invdet;
	r->c.y = -bz * invdet;
	r->c.z =  cz * invdet;
}


// ---------------------
// --- mat2x2 common ---
// ---------------------
VMATHDEF void mat2_zero(mat2 *m)
{
	m->a.x = m->a.y = m->b.x = m->b.y = 0;
}

VMATHDEF void mat2_id(mat2 *m)
{
	m->a.y = m->b.x = 0;
	m->a.x = m->b.y = VP(1);
}

VMATHDEF void mat2_t(mat2 *m)
{
	scalar t;
	t = m->a.y; m->a.y = m->b.x; m->b.x = t;
}

VMATHDEF void mat2_rot(mat2 *m, scalar a)
{
	scalar c = vcos(a), s = vsin(a);
	m->a.x =  c; m->a.y = -s;
	m->b.x =  s; m->b.y =  c;
}

VMATHDEF void mat2_eq_scale(mat2 *m, scalar sx, scalar sy)
{
	m->a.y = m->b.x = 0;
	m->a.x = sx; m->b.y = sy;
}

VMATHDEF void mat2_eq_shear(mat2 *m, scalar x, scalar y)
{
	m->a.x = VP(1); m->a.y = x;
	m->b.x = y;     m->b.y = VP(1);
}

VMATHDEF void mat2_setarr(mat2 *m, scalar *s)
{
	vmemcpy(m, s, sizeof *m);
}

VMATHDEF void mat2_copy(mat2 *r, mat2 *m)
{
	vmemcpy(r, m, sizeof *m);
}

VMATHDEF void mat2_from_vec4(mat2 *m, vec4 *v)
{
	vec2_setcomp(&m->a, v->x, v->y);
	vec2_setcomp(&m->b, v->z, v->w);
}

VMATHDEF void mat2_setrows(mat2 *m, vec2 *a, vec2 *b)
{
	m->a.x = a->x; m->a.y = a->y;
	m->b.x = b->x; m->b.y = b->y;
}

VMATHDEF void mat2_setcols(mat2 *m, vec2 *a, vec2 *b)
{
	vec2_setcomp(&m->a, a->x, b->y);
	vec2_setcomp(&m->b, a->y, b->y);
}

VMATHDEF void mat2_setrowa(mat2 *m, vec2 *v) { vec2_copy(&m->a, v); }
VMATHDEF void mat2_setrowb(mat2 *m, vec2 *v) { vec2_copy(&m->b, v); }

VMATHDEF void mat2_setcolx(mat2 *m, vec2 *v) { m->a.x = v->x; m->b.x = v->y; }
VMATHDEF void mat2_setcoly(mat2 *m, vec2 *v) { m->a.y = v->x; m->b.y = v->y; }

VMATHDEF void mat2_rowa(vec2 *r, mat2 *m) { vec2_copy(r, &m->a); }
VMATHDEF void mat2_rowb(vec2 *r, mat2 *m) { vec2_copy(r, &m->b); }

VMATHDEF void mat2_colx(vec2 *r, mat2 *m) { vec2_setcomp(r, m->a.x, m->b.x); }
VMATHDEF void mat2_coly(vec2 *r, mat2 *m) { vec2_setcomp(r, m->a.y, m->b.y); }

VMATHDEF void mat2_smul(mat2 *r, scalar s, mat2 *m)
{
	vec2_smul(&r->a, s, &m->a);
	vec2_smul(&r->b, s, &m->b);
}

VMATHDEF void mat2_mulrows(mat2 *r, mat2 *m, scalar a, scalar b)
{
	vec2_smul(&r->a, a, &m->a);
	vec2_smul(&r->b, b, &m->b);
}

VMATHDEF void mat2_mulrowv(mat2 *r, mat2 *m, vec2 *v)
{
	vec2_smul(&r->a, v->x, &m->a);
	vec2_smul(&r->b, v->y, &m->b);
}

VMATHDEF void mat2_negate(mat2 *r, mat2 *m)
{
	vec2_negate(&r->a, &m->a);
	vec2_negate(&r->b, &m->b);
}

VMATHDEF void mat2_add(mat2 *r, mat2 *f, mat2 *g)
{
	vec2_add(&r->a, &f->a, &g->a);
	vec2_add(&r->b, &f->b, &g->b);
}

VMATHDEF void mat2_sub(mat2 *r, mat2 *f, mat2 *g)
{
	vec2_sub(&r->a, &f->a, &g->a);
	vec2_sub(&r->b, &f->b, &g->b);
}

VMATHDEF void mat2_tmul(mat2 *r, mat2 *f, mat2 *g)
{
	vec2_tmul(&r->a, &f->a, &g->a);
	vec2_tmul(&r->b, &f->b, &g->b);
}

VMATHDEF void _vec2_mul_mat2(vec2 *r, vec2 *v, mat2 *m)
{
	r->x = m->a.x * v->x + m->b.x * v->y;
	r->y = m->a.y * v->x + m->b.y * v->y;
}

VMATHDEF void mat2_mul(mat2 *r, mat2 *f, mat2 *g)
{
	mat2 m;
	_vec2_mul_mat2(&m.a, &f->a, g);
	_vec2_mul_mat2(&m.b, &f->b, g);
	vmemcpy(r, &m, sizeof m);
}

VMATHDEF void mat2_ma(mat2 *r, mat2 *f, scalar t, mat2 *g)
{
	vec2_ma(&r->a, &f->a, t, &g->a);
	vec2_ma(&r->b, &f->b, t, &g->b);
}

VMATHDEF void mat2_combine(mat2 *r, scalar s, mat2 *f, scalar t, mat2 *g)
{
	vec2_combine(&r->a, s, &f->a, t, &g->a);
	vec2_combine(&r->b, s, &f->b, t, &g->b);
}

VMATHDEF void mat2_lerp(mat2 *r, mat2 *f, mat2 *g, scalar t)
{
	scalar s = VP(1) - t;
	vec2_combine(&r->a, s, &f->a, t, &g->a);
	vec2_combine(&r->b, s, &f->b, t, &g->b);
}

VMATHDEF void vec2_mul_mat2(vec2 *r, vec2 *v, mat2 *m)
{
	scalar x, y;
	x = m->a.x * v->x + m->b.x * v->y;
	y = m->a.y * v->x + m->b.y * v->y;
	r->x = x; r->y = y;
}

VMATHDEF void mat2_mul_vec2(vec2 *r, mat2 *m, vec2 *v)
{
	scalar x, y;
	x = m->a.x * v->x + m->a.y * v->y;
	y = m->b.x * v->x + m->b.y * v->y;
	r->x = x; r->y = y;
}

VMATHDEF void mat2_transpose(mat2 *r, mat2 *m)
{
	scalar t;
	t = m->a.y; r->a.y = m->b.x; r->b.x = t;
	
	r->a.x = m->a.x; r->b.y = m->b.y;
}

VMATHDEF void mat2_vtrace(vec2 *v, mat2 *m)
{
	v->x = m->a.x; v->y = m->b.y;;
}

VMATHDEF scalar mat2_trace(mat2 *m)
{
	return m->a.x + m->b.y;
}

VMATHDEF scalar mat2_det(mat2 *m)
{
	return m->a.x * m->b.y - m->a.y * m->b.x;
}

VMATHDEF void mat2_inverse(mat2 *r, mat2 *m)
{
	scalar invdet = VP(1) / (m->a.x * m->b.y - m->a.y * m->b.x);
	
	r->a.x =  m->b.y * invdet;
	r->a.y = -m->a.y * invdet;
	
	r->b.x = -m->b.x * invdet;
	r->b.y =  m->a.x * invdet;
}


// -------------------
// --- quat common ---
// -------------------
VMATHDEF void quat_id(quat *v)
{
	v->x = v->y = v->z = 0; v->w = VP(1);
}

VMATHDEF void quat_from_rot(quat *r, vec3 *v, scalar a)
{
	a = a / VP(2);
	scalar s = vsin(a) / vsqrt(v->x * v->x + v->y * v->y + v->z * v->z);
	r->x = v->x * s; r->y = v->y * s; r->z = v->z * s;
	r->w = vcos(a);
}

VMATHDEF void quat_inv(quat *r, quat *v)
{
	scalar s = VP(1) / vsqrt(v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w);
	r->x = -v->x * s; r->y = -v->y * s;
	r->z = -v->z * s; r->w =  v->w * s;
}

VMATHDEF void quat_conj(quat *r, quat *v)
{
	r->x = -v->x; r->y = -v->y;
	r->z = -v->z; r->w =  v->w;
}

VMATHDEF void quat_from_vec3(quat *r, vec3 *v)
{
	r->x = v->x; r->y = v->y;
	r->z = v->z; r->w = 0;
}

VMATHDEF void vec3_from_quat(vec3 *r, quat *v)
{
	r->x = v->x; r->y = v->y; r->z = v->z;
}


VMATHDEF void quat_mul(quat *r, quat *u, quat *v)
{
	scalar x, y, z, w;
	w = u->w * v->w - u->x * v->x - u->y * v->y - u->z * v->z;
	x = u->w * v->x + u->x * v->w + u->y * v->z - u->z * v->y;
	y = u->w * v->y - u->x * v->z + u->y * v->w + u->z * v->x;
	z = u->w * v->z + u->x * v->y - u->y * v->x + u->z * v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

VMATHDEF void quat_mul_vec3(quat *r, quat *u, vec3 *v)
{
	scalar x, y, z, w;
	w = - u->x * v->x - u->y * v->y - u->z * v->z;
	x = u->w * v->x + u->y * v->z - u->z * v->y;
	y = u->w * v->y - u->x * v->z + u->z * v->x;
	z = u->w * v->z + u->x * v->y - u->y * v->x;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

VMATHDEF void vec3_mul_quat(quat *r, vec3 *u, quat *v)
{
	scalar x, y, z, w;
	w = - u->x * v->x - u->y * v->y - u->z * v->z;
	x = + u->x * v->w + u->y * v->z - u->z * v->y;
	y = - u->x * v->z + u->y * v->w + u->z * v->x;
	z = + u->x * v->y - u->y * v->x + u->z * v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

VMATHDEF void vec3_rotate_quat(vec3 *r, quat *q, vec3 *v)
{
	quat p;
	quat_inv(&p, q);
	vec3_mul_quat(&p, v, &p); // p = v * inv(q)
	quat_mul(&p, q, &p); // p = q * v * inv(q)
	r->x = p.x; r->y = p.y; r->z = p.z;
}

VMATHDEF void mat3_from_quat(mat3 *m, quat *q)
{
	scalar x = q->x, y = q->y, z = q->z, w = q->w;
	
	m->a.x = VP(1) - VP(2) * (y * y + z * z);
	m->a.y = VP(2) * (x * y - z * w);
	m->a.z = VP(2) * (x * z + y * w);
	
	m->b.x = VP(2) * (x * y + z * w);
	m->b.y = VP(1) - VP(2) * (x * x + z * z);
	m->b.z = VP(2) * (y * z - x * w);
	
	m->c.x = VP(2) * (x * z - y * w);
	m->c.y = VP(2) * (y * z + x * w);
	m->c.z = VP(1) - VP(2) * (x * x + y * y);
}

VMATHDEF void mat3x4_from_srt(mat3x4 *m, vec3 *s, quat *r, vec3 *t)
{
	scalar x = r->x, y = r->y, z = r->z, w = r->w;
	
	m->a.x = s->x * (VP(1) - VP(2) * (y * y + z * z));
	m->a.y = s->y * VP(2) * (x * y - z * w);
	m->a.z = s->z * VP(2) * (x * z + y * w);
	m->a.w = t->x;
	
	m->b.x = s->x * VP(2) * (x * y + z * w);
	m->b.y = s->y * (VP(1) - VP(2) * (x * x + z * z));
	m->b.z = s->z * VP(2) * (y * z - x * w);
	m->b.w = t->y;
	
	m->c.x = s->x * VP(2) * (x * z - y * w);
	m->c.y = s->y * VP(2) * (y * z + x * w);
	m->c.z = s->z * (VP(1) - VP(2) * (x * x + y * y));
	m->c.w = t->z;
}

VMATHDEF void mat3x4_inverse_srt(mat3x4 *r, mat3x4 *m)
{
	scalar n, x, y, z;
	
	x = m->a.x; y = m->b.x; z = m->c.x;
	n = VP(1) / (x * x + y * y + z * z);
	r->a.x = x * n; r->a.y = y * n; r->a.z = z * n;
	
	x = m->a.y; y = m->b.y; z = m->c.y;
	n = VP(1) / (x * x + y * y + z * z);
	r->b.x = x * n; r->b.y = y * n; r->b.z = z * n;
	
	x = m->a.z; y = m->b.z; z = m->c.z;
	n = VP(1) / (x * x + y * y + z * z);
	r->c.x = x * n; r->c.y = y * n; r->c.z = z * n;
	
	x = m->a.w; y = m->b.w; z = m->c.w;
	
	r->a.w = -(r->a.x * x + r->a.y * y + r->a.z * z);
	r->b.w = -(r->b.x * x + r->b.y * y + r->b.z * z);
	r->c.w = -(r->c.x * x + r->c.y * y + r->c.z * z);
}


VMATHDEF void quat_from_mat3(quat *q, mat3 *m)
{
	scalar t, x,y,z,w;
	scalar tr0 = m->a.x, tr1 = m->b.y, tr2 = m->c.z;
	if (tr2 < 0)
	{
		if (tr0 > tr1)
		{
			t = VP(1) + tr0 - tr1 - tr2;
			x = t;               y = m->a.y + m->b.x;
			z = m->c.x + m->a.z; w = m->c.y - m->b.z;
		}
		else
		{
			t = VP(1) - tr0 + tr1 - tr2;
			x = m->a.y + m->b.x; y = t;
			z = m->b.z + m->c.y; w = m->a.z - m->c.x;
		}
	}
	else
	{
		if (tr0 < -tr1)
		{
			t = VP(1) - tr0 - tr1 + tr2;
			x = m->c.x + m->a.z;  y = m->b.z + m->c.y;
			z = t;                w = m->b.x - m->a.y;
		}
		else
		{
			t = VP(1) + tr0 + tr1 + tr2;
			x = m->c.y - m->b.z; y = m->a.z - m->c.x;
			y = m->b.x - m->a.y; w = t;
		}
	}
	
	t = VP(1) / (VP(2) * vsqrt(t));
	q->x = t * x; q->y = t * y; q->z = t * z; q->w = t * w;
}

VMATHDEF void mat3_from_dir(mat3 *m, vec3 *dir)
{
	scalar x = vabs(dir->x), y = vabs(dir->y), z = vabs(dir->z);
	vec3 u;
	vec3_zero(&u);
	if (x <= y && x <= z)
		u.x = VP(1);
	else if (y <= x && y <= z)
		u.y = VP(1);
	else
		u.z = VP(1);
	
	vec3_copy(&m->c, dir);
	vec3_makeunit(&m->c);
	
	vec3_project(&m->a, &u, &m->c);
	vec3_makeunit(&m->a);
	
	vec3_cross(&m->b, &m->a, &m->c);
	
	mat3_t(m);
}

#endif // VMATH_IMPLEMENTATION

