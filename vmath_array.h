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
typedef scalar vec4[4], quat[4], plane[4];

typedef scalar vec3[3];

typedef scalar vec2[2];


// a,b,c,d are rows of the matrix
// i.e. row major matrix
// However we use v' = Mv
// i.e. pre-multiplied
typedef vec4 mat4[4], mat4x4[4];

typedef vec4 mat3x4[3]; // last row is assumed to be {0,0,0,1}

typedef vec3 mat3[3], mat3x3[3];

typedef vec2 mat2[2], mat2x2[2];



// vec4 common
VMATHDEF scalar vec4_dot(vec4 u, vec4 v); // return u . v
VMATHDEF scalar vec4_lensqr(vec4 v); // return |v|^2
VMATHDEF scalar vec4_len(vec4 v); // return |v|
VMATHDEF scalar vec4_len1(vec4 v); // return d1(v, 0)
VMATHDEF scalar vec4_distsqr(vec4 u, vec4 v); // return |u - v|^2
VMATHDEF scalar vec4_dist(vec4 u, vec4 v); // return |u - v|
VMATHDEF scalar vec4_dist1(vec4 u, vec4 v); // return d1(u, v)
VMATHDEF void vec4_zero(vec4 v); // v = 0
VMATHDEF void vec4_makeunit(vec4 v); // v = v / |v|
VMATHDEF void vec4_setcomp(vec4 v, scalar x, scalar y, scalar z, scalar w); // v = {x, y, z, w} (set components)
VMATHDEF void vec4_setarr(vec4 v, scalar *u); // v = {u} (set array)
VMATHDEF void vec4_copy(vec4 r, vec4 v); // r = v
VMATHDEF void vec4_from_vec2(vec4 r, vec2 v); // r = {v, 0, 0}
VMATHDEF void vec4_from_vec3(vec4 r, vec3 v); // r = {v, 0}
VMATHDEF void vec4_from_mat2(vec4 v, mat2 m); // r = {m[0], m[1]}
VMATHDEF void vec4_smul(vec4 r, scalar s, vec4 v); // r = s * v (scalar multiplication)
VMATHDEF void vec4_negate(vec4 r, vec4 v); // r = -v
VMATHDEF void vec4_normalize(vec4 r, vec4 v); // r = v / |v| (possible div. by 0)
VMATHDEF void vec4_add(vec4 r, vec4 u, vec4 v); // r = u + v
VMATHDEF void vec4_sub(vec4 r, vec4 u, vec4 v); // r = u - v
VMATHDEF void vec4_tmul(vec4 r, vec4 u, vec4 v); // r = u * v (term-wise multiplication)
VMATHDEF void vec4_ma(vec4 r, vec4 u, scalar s, vec4 v); // r = u + t * v (multiply add)
VMATHDEF void vec4_combine(vec4 r, scalar s, vec4 u, scalar t, vec4 v); // r = s * u + t * v (linear combination)
VMATHDEF void vec4_lerp(vec4 r, vec4 u, vec4 v, scalar t); // r = (1 - t) * u + t * v
VMATHDEF void vec4_project(vec4 r, vec4 v, vec4 n); // r = projection of v wrt (unit) n
VMATHDEF void vec4_reflect(vec4 r, vec4 v, vec4 n); // r = reflection of v wrt (unit) n

// vec3 common
VMATHDEF scalar vec3_dot(vec3 u, vec3 v);
VMATHDEF scalar vec3_lensqr(vec3 v);
VMATHDEF scalar vec3_len(vec3 v);
VMATHDEF scalar vec3_len1(vec3 v);
VMATHDEF scalar vec3_distsqr(vec3 u, vec3 v);
VMATHDEF scalar vec3_dist(vec3 u, vec3 v);
VMATHDEF scalar vec3_dist1(vec3 u, vec3 v);
VMATHDEF void vec3_zero(vec3 v);
VMATHDEF void vec3_makeunit(vec3 v);
VMATHDEF void vec3_setcomp(vec3 v, scalar x, scalar y, scalar z);
VMATHDEF void vec3_setarr(vec3 v, scalar *u);
VMATHDEF void vec3_copy(vec3 r, vec3 v);
VMATHDEF void vec3_from_vec2(vec3 r, vec2 v);
VMATHDEF void vec3_from_vec4(vec3 r, vec4 v);
VMATHDEF void vec3_smul(vec3 r, scalar s, vec3 v);
VMATHDEF void vec3_negate(vec3 r, vec3 v);
VMATHDEF void vec3_normalize(vec3 r, vec3 v);
VMATHDEF void vec3_add(vec3 r, vec3 u, vec3 v);
VMATHDEF void vec3_sub(vec3 r, vec3 u, vec3 v);
VMATHDEF void vec3_tmul(vec3 r, vec3 u, vec3 v);
VMATHDEF void vec3_ma(vec3 r, vec3 u, scalar t, vec3 v);
VMATHDEF void vec3_combine(vec3 r, scalar s, vec3 u, scalar t, vec3 v);
VMATHDEF void vec3_lerp(vec3 r, vec3 u, vec3 v, scalar t);
VMATHDEF void vec3_project(vec3 r, vec3 v, vec3 n);
VMATHDEF void vec3_reflect(vec3 r, vec3 v, vec3 n);

// vec2 common
VMATHDEF scalar vec2_dot(vec2 u, vec2 v);
VMATHDEF scalar vec2_lensqr(vec2 v);
VMATHDEF scalar vec2_len(vec2 v);
VMATHDEF scalar vec2_len1(vec2 v);
VMATHDEF scalar vec2_distsqr(vec2 u, vec2 v);
VMATHDEF scalar vec2_dist(vec2 u, vec2 v);
VMATHDEF scalar vec2_dist1(vec2 u, vec2 v);
VMATHDEF void vec2_zero(vec2 v);
VMATHDEF void vec2_makeunit(vec2 v);
VMATHDEF void vec2_setcomp(vec2 v, scalar x, scalar y);
VMATHDEF void vec2_setarr(vec2 v, scalar *u);
VMATHDEF void vec2_copy(vec2 r, vec2 v);
VMATHDEF void vec2_from_vec3(vec2 r, vec3 v);
VMATHDEF void vec2_from_vec4(vec2 r, vec4 v);
VMATHDEF void vec2_smul(vec2 r, scalar s, vec2 v);
VMATHDEF void vec2_negate(vec2 r, vec2 v);
VMATHDEF void vec2_normalize(vec2 r, vec2 v);
VMATHDEF void vec2_add(vec2 r, vec2 u, vec2 v);
VMATHDEF void vec2_sub(vec2 r, vec2 u, vec2 v);
VMATHDEF void vec2_tmul(vec2 r, vec2 u, vec2 v);
VMATHDEF void vec2_ma(vec2 r, vec2 u, scalar t, vec2 v);
VMATHDEF void vec2_combine(vec2 r, scalar s, vec2 u, scalar t, vec2 v);
VMATHDEF void vec2_lerp(vec2 r, vec2 u, vec2 v, scalar t);
VMATHDEF void vec2_project(vec2 r, vec2 v, vec2 n);
VMATHDEF void vec2_reflect(vec2 r, vec2 v, vec2 n);

// vec other
VMATHDEF scalar vec2_cross(vec2 u, vec2 v); // scalar since first 2 components are always 0
VMATHDEF void vec3_cross(vec3 r, vec3 u, vec3 v); // r = u x v
VMATHDEF void vec4_cross(vec4 r, vec4 u, vec4 v); // r = {vec3(u) x vec3(v), 0}
VMATHDEF void plane_from_points(plane p, vec3 a, vec3 b, vec3 c);
VMATHDEF void vec3_plane_project(vec3 r, vec3 v, plane p); // same as vec3_project (p has unit normal)
VMATHDEF void vec3_plane_reflect(vec3 r, vec3 v, plane p); // same as vec3_reflect
VMATHDEF void plane_makeunit(plane p); // make normal of p unit
VMATHDEF void plane_normalize(plane r, plane p); // normalize the normal 

// mat4 common
VMATHDEF void mat4_zero(mat4 m); // m = 0
VMATHDEF void mat4_id(mat4 m); // m = I
VMATHDEF void mat4_t(mat4 m); // m = m^T
VMATHDEF void mat4_eq_rotx(mat4 m, scalar a); // m = R_x (rotation matrix around x with angle a)
VMATHDEF void mat4_eq_roty(mat4 m, scalar a); // m = R_y
VMATHDEF void mat4_eq_rotz(mat4 m, scalar a); // m = R_z
VMATHDEF void mat4_eq_rotaxis(mat4 m, vec3 u, scalar a); // m = R_u (rotation matrix around v with angle a)
VMATHDEF void mat4_eq_scale(mat4 m, scalar sx, scalar sy, scalar sz); // m = scale matrix
VMATHDEF void mat4_eq_shear(mat4 m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy); // xy: of x along y
VMATHDEF void mat4_eq_trans(mat4 m, scalar tx, scalar ty, scalar tz); // m = translation matrix
VMATHDEF void mat4_setarr(mat4 m, scalar *s); // m = [s]
VMATHDEF void mat4_copy(mat4 r, mat4 m); // r = m
VMATHDEF void mat4_from_mat3(mat4 r, mat3 m);
VMATHDEF void mat4_from_mat3x4(mat4 r, mat3x4 m);
VMATHDEF void mat4_setrows(mat4 m, vec4 ra, vec4 rb, vec4 rc, vec4 rd); // m = [ra,rb,rc,rd]
VMATHDEF void mat4_setcols(mat4 m, vec4 cx, vec4 cy, vec4 cz, vec4 cw); // m = [cx,cy,cz,cw]^T
VMATHDEF void mat4_setrowa(mat4 m, vec4 v); // row a of m = v
VMATHDEF void mat4_setrowb(mat4 m, vec4 v);
VMATHDEF void mat4_setrowc(mat4 m, vec4 v);
VMATHDEF void mat4_setrowd(mat4 m, vec4 v);
VMATHDEF void mat4_setcolx(mat4 m, vec4 v); // col x of m = v
VMATHDEF void mat4_setcoly(mat4 m, vec4 v);
VMATHDEF void mat4_setcolz(mat4 m, vec4 v);
VMATHDEF void mat4_setcolw(mat4 m, vec4 v);
VMATHDEF void mat4_rowa(vec4 r, mat4 m); // r = row a of m
VMATHDEF void mat4_rowb(vec4 r, mat4 m);
VMATHDEF void mat4_rowc(vec4 r, mat4 m);
VMATHDEF void mat4_rowd(vec4 r, mat4 m);
VMATHDEF void mat4_colx(vec4 r, mat4 m); // r = col x of m
VMATHDEF void mat4_coly(vec4 r, mat4 m);
VMATHDEF void mat4_colz(vec4 r, mat4 m);
VMATHDEF void mat4_colw(vec4 r, mat4 m);
VMATHDEF void mat4_smul(mat4 r, scalar s, mat4 m); // r = s * m  (scalar multiplication)
VMATHDEF void mat4_mulrows(mat4 r, mat4 m, scalar a, scalar b, scalar c, scalar d); // multiply rows with scalars
VMATHDEF void mat4_mulrowv(mat4 r, mat4 m, vec4 v); // multiply rows with components of v
VMATHDEF void mat4_negate(mat4 r, mat4 m); // r = -m
VMATHDEF void mat4_add(mat4 r, mat4 f, mat4 g); // r = f + g
VMATHDEF void mat4_sub(mat4 r, mat4 f, mat4 g); // r = f - g
VMATHDEF void mat4_tmul(mat4 r, mat4 f, mat4 g); // r = f * g (term-wise multiplication)
// VMATHDEF void _vec4_mul_mat4(vec4 r, vec4 v, mat4 m); // r = vm (r != v, used internally)
VMATHDEF void mat4_mul(mat4 r, mat4 f, mat4 g); // r = f * g
VMATHDEF void mat4_ma(mat4 r, mat4 f, scalar t, mat4 g); // r = f + t * g
VMATHDEF void mat4_combine(mat4 r, scalar s, mat4 f, scalar t, mat4 g); // r = s * f + t * g
VMATHDEF void mat4_lerp(mat4 r, mat4 f, mat4 g, scalar t); // r = (1 - t) * f + t * g
VMATHDEF void vec4_mul_mat4(vec4 r, vec4 v, mat4 m); // r = vm
VMATHDEF void mat4_mul_vec4(vec4 r, mat4 m, vec4 v); // r = mv
VMATHDEF void mat4_mul_vec3(vec3 r, mat4 m, vec3 v); // r = mv (affine)
VMATHDEF void mat4_transpose(mat4 r, mat4 m); // r = m^T
VMATHDEF void mat4_inverse(mat4 r, mat4 m); // r = inverse(m) (r != m)
VMATHDEF void mat4_vtrace(vec4 v, mat4 m); // return trace (as) vector
VMATHDEF scalar mat4_trace(mat4 m); // return trace
VMATHDEF scalar mat4_det3(mat4 m); // return determinant of inner 3x3
VMATHDEF scalar mat4_det(mat4 m); // return determinant of m

// mat3x4 common
VMATHDEF void mat3x4_zero(mat3x4 m);
VMATHDEF void mat3x4_id(mat3x4 m);
VMATHDEF void mat3x4_t(mat3x4 m);
VMATHDEF void mat3x4_eq_rotx(mat3x4 m, scalar a);
VMATHDEF void mat3x4_eq_roty(mat3x4 m, scalar a);
VMATHDEF void mat3x4_eq_rotz(mat3x4 m, scalar a);
VMATHDEF void mat3x4_eq_rotaxis(mat3x4 m, vec3 u, scalar a);
VMATHDEF void mat3x4_eq_scale(mat3x4 m, scalar sx, scalar sy, scalar sz);
VMATHDEF void mat3x4_eq_shear(mat3x4 m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy);
VMATHDEF void mat3x4_eq_trans(mat3x4 m, scalar tx, scalar ty, scalar tz);
VMATHDEF void mat3x4_setarr(mat3x4 m, scalar *s);
VMATHDEF void mat3x4_copy(mat3x4 r, mat3x4 m);
VMATHDEF void mat3x4_from_mat3(mat3x4 r, mat3 m);
VMATHDEF void mat3x4_from_mat4(mat3x4 r, mat4 m);
VMATHDEF void mat3x4_setrows(mat3x4 m, vec4 ra, vec4 rb, vec4 rc);
VMATHDEF void mat3x4_setcols(mat3x4 m, vec3 cx, vec3 cy, vec3 cz, vec3 cw);
VMATHDEF void mat3x4_setrowa(mat3x4 m, vec4 v);
VMATHDEF void mat3x4_setrowb(mat3x4 m, vec4 v);
VMATHDEF void mat3x4_setrowc(mat3x4 m, vec4 v);
VMATHDEF void mat3x4_setcolx(mat3x4 m, vec3 v);
VMATHDEF void mat3x4_setcoly(mat3x4 m, vec3 v);
VMATHDEF void mat3x4_setcolz(mat3x4 m, vec3 v);
VMATHDEF void mat3x4_setcolw(mat3x4 m, vec3 v);
VMATHDEF void mat3x4_rowa(vec4 r, mat3x4 m); 
VMATHDEF void mat3x4_rowb(vec4 r, mat3x4 m); 
VMATHDEF void mat3x4_rowc(vec4 r, mat3x4 m); 
VMATHDEF void mat3x4_colx(vec3 r, mat3x4 m);
VMATHDEF void mat3x4_coly(vec3 r, mat3x4 m);
VMATHDEF void mat3x4_colz(vec3 r, mat3x4 m);
VMATHDEF void mat3x4_colw(vec3 r, mat3x4 m);
VMATHDEF void mat3x4_smul(mat3x4 r, scalar s, mat3x4 m);
VMATHDEF void mat3x4_mulrows(mat3x4 r, mat3x4 m, scalar a, scalar b, scalar c);
VMATHDEF void mat3x4_mulrowv(mat3x4 r, mat3x4 m, vec3 v);
VMATHDEF void mat3x4_negate(mat3x4 r, mat3x4 m);
VMATHDEF void mat3x4_add(mat3x4 r, mat3x4 f, mat3x4 g);
VMATHDEF void mat3x4_sub(mat3x4 r, mat3x4 f, mat3x4 g);
VMATHDEF void mat3x4_tmul(mat3x4 r, mat3x4 f, mat3x4 g);
// VMATHDEF void _vec4_mul_mat3x4(vec4 r, vec4 v, mat3x4 m);
VMATHDEF void mat3x4_mul(mat3x4 r, mat3x4 f, mat3x4 g);
VMATHDEF void mat3x4_ma(mat3x4 r, mat3x4 f, scalar t, mat3x4 g);
VMATHDEF void mat3x4_combine(mat3x4 r, scalar s, mat3x4 f, scalar t, mat3x4 g);
VMATHDEF void mat3x4_lerp(mat3x4 r, mat3x4 f, mat3x4 g, scalar t);
VMATHDEF void vec4_mul_mat3x4(vec4 r, vec4 v, mat3x4 m);
VMATHDEF void mat3x4_mul_vec4(vec4 r, mat3x4 m, vec4 v);
VMATHDEF void mat3x4_mul_vec3(vec3 r, mat3x4 m, vec3 v);
VMATHDEF void mat3x4_transpose(mat3x4 r, mat3x4 m);
VMATHDEF void mat3x4_inverse(mat3x4 r, mat3x4 m);
VMATHDEF void mat3x4_vtrace(vec3 v, mat3x4 m);
VMATHDEF scalar mat3x4_trace(mat3x4 m);
VMATHDEF scalar mat3x4_det(mat3x4 m);

// mat3 common
VMATHDEF void mat3_zero(mat3 m);
VMATHDEF void mat3_id(mat3 m);
VMATHDEF void mat3_t(mat3 m);
VMATHDEF void mat3_eq_rotx(mat3 m, scalar a);
VMATHDEF void mat3_eq_roty(mat3 m, scalar a);
VMATHDEF void mat3_eq_rotz(mat3 m, scalar a);
VMATHDEF void mat3_eq_rot2d(mat3 m, scalar a);
VMATHDEF void mat3_eq_rotaxis(mat3 m, vec3 u, scalar a);
VMATHDEF void mat3_eq_trans2d(mat3 m, scalar tx, scalar ty);
VMATHDEF void mat3_eq_scale(mat3 m, scalar sx, scalar sy, scalar sz);
VMATHDEF void mat3_eq_scale2d(mat3 m, scalar sx, scalar sy);
VMATHDEF void mat3_eq_shear(mat3 m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy);
VMATHDEF void mat3_eq_shear2d(mat3 m, scalar x, scalar y);
VMATHDEF void mat3_setarr(mat3 m, scalar *s);
VMATHDEF void mat3_copy(mat3 r, mat3 m);
VMATHDEF void mat3_from_mat4(mat3 r, mat4 m);
VMATHDEF void mat3_from_mat3x4(mat3 r, mat3x4 m);
VMATHDEF void mat3_setrows(mat3 m, vec3 ra, vec3 rb, vec3 rc);
VMATHDEF void mat3_setcols(mat3 m, vec3 cx, vec3 cy, vec3 cz);
VMATHDEF void mat3_setrowa(mat3 m, vec3 v);
VMATHDEF void mat3_setrowb(mat3 m, vec3 v);
VMATHDEF void mat3_setrowc(mat3 m, vec3 v);
VMATHDEF void mat3_setcolx(mat3 m, vec3 v);
VMATHDEF void mat3_setcoly(mat3 m, vec3 v);
VMATHDEF void mat3_setcolz(mat3 m, vec3 v);
VMATHDEF void mat3_rowa(vec3 r, mat3 m);
VMATHDEF void mat3_rowb(vec3 r, mat3 m);
VMATHDEF void mat3_rowc(vec3 r, mat3 m);
VMATHDEF void mat3_colx(vec3 r, mat3 m);
VMATHDEF void mat3_coly(vec3 r, mat3 m);
VMATHDEF void mat3_colz(vec3 r, mat3 m);
VMATHDEF void mat3_smul(mat3 r, scalar s, mat3 m);
VMATHDEF void mat3_mulrows(mat3 r, mat3 m, scalar a, scalar b, scalar c);
VMATHDEF void mat3_mulrowv(mat3 r, mat3 m, vec3 v);
VMATHDEF void mat3_negate(mat3 r, mat3 m);
VMATHDEF void mat3_add(mat3 r, mat3 f, mat3 g);
VMATHDEF void mat3_sub(mat3 r, mat3 f, mat3 g);
VMATHDEF void mat3_tmul(mat3 r, mat3 f, mat3 g);
// VMATHDEF void _vec3_mul_mat3(vec3 r, vec3 v, mat3 m);
VMATHDEF void mat3_mul(mat3 r, mat3 f, mat3 g);
VMATHDEF void mat3_ma(mat3 r, mat3 f, scalar t, mat3 g);
VMATHDEF void mat3_combine(mat3 r, scalar s, mat3 f, scalar t, mat3 g);
VMATHDEF void mat3_lerp(mat3 r, mat3 f, mat3 g, scalar t);
VMATHDEF void vec3_mul_mat3(vec3 r, vec3 v, mat3 m);
VMATHDEF void mat3_mul_vec3(vec3 r, mat3 m, vec3 v);
VMATHDEF void mat3_mul_vec2(vec2 r, mat3 m, vec2 v);
VMATHDEF void mat3_transpose(mat3 r, mat3 m);
VMATHDEF void mat3_inverse(mat3 r, mat3 m);
VMATHDEF void mat3_vtrace(vec3 v, mat3 m);
VMATHDEF scalar mat3_trace(mat3 m);
VMATHDEF scalar mat3_det(mat3 m);

// mat2 common
VMATHDEF void mat2_zero(mat2 m);
VMATHDEF void mat2_id(mat2 m);
VMATHDEF void mat2_t(mat2 m);
VMATHDEF void mat2_eq_rot(mat2 m, scalar a); // around origin/z
VMATHDEF void mat2_eq_scale(mat2 m, scalar sx, scalar sy);
VMATHDEF void mat2_eq_shear(mat2 m, scalar x, scalar y);
VMATHDEF void mat2_setarr(mat2 m, scalar *s);
VMATHDEF void mat2_copy(mat2 r, mat2 m);
VMATHDEF void mat2_from_vec4(mat2 m, vec4 v);
VMATHDEF void mat2_setrows(mat2 m, vec2 a, vec2 b);
VMATHDEF void mat2_setcols(mat2 m, vec2 a, vec2 b);
VMATHDEF void mat2_setrowa(mat2 m, vec2 v);
VMATHDEF void mat2_setrowb(mat2 m, vec2 v);
VMATHDEF void mat2_setcolx(mat2 m, vec2 v);
VMATHDEF void mat2_setcoly(mat2 m, vec2 v);
VMATHDEF void mat2_rowa(vec2 r, mat2 m);
VMATHDEF void mat2_rowb(vec2 r, mat2 m);
VMATHDEF void mat2_colx(vec2 r, mat2 m);
VMATHDEF void mat2_coly(vec2 r, mat2 m);
VMATHDEF void mat2_smul(mat2 r, scalar s, mat2 m);
VMATHDEF void mat2_mulrows(mat2 r, mat2 m, scalar a, scalar b);
VMATHDEF void mat2_mulrowv(mat2 r, mat2 m, vec2 v);
VMATHDEF void mat2_negate(mat2 r, mat2 m);
VMATHDEF void mat2_add(mat2 r, mat2 f, mat2 g);
VMATHDEF void mat2_sub(mat2 r, mat2 f, mat2 g);
VMATHDEF void mat2_tmul(mat2 r, mat2 f, mat2 g);
// VMATHDEF void _vec2_mul_mat2(vec2 r, vec2 v, mat2 m);
VMATHDEF void mat2_mul(mat2 r, mat2 f, mat2 g);
VMATHDEF void mat2_ma(mat2 r, mat2 f, scalar t, mat2 g);
VMATHDEF void mat2_combine(mat2 r, scalar s, mat2 f, scalar t, mat2 g);
VMATHDEF void mat2_lerp(mat2 r, mat2 f, mat2 g, scalar t);
VMATHDEF void vec2_mul_mat2(vec2 r, vec2 v, mat2 m);
VMATHDEF void mat2_mul_vec2(vec2 r, mat2 m, vec2 v);
VMATHDEF void mat2_transpose(mat2 r, mat2 m);
VMATHDEF void mat2_inverse(mat2 r, mat2 m);
VMATHDEF void mat2_vtrace(vec2 v, mat2 m);
VMATHDEF scalar mat2_trace(mat2 m);
VMATHDEF scalar mat2_det(mat2 m);

// quat common (also vec4)
VMATHDEF void quat_id(quat v); // q = 1
VMATHDEF void quat_from_rot(quat r, vec3 v, scalar a); // q = rotation around axis:v with angle:a
VMATHDEF void quat_inv(quat r, quat v); // r = inverse of q
VMATHDEF void quat_conj(quat r, quat v); // r = conjugate of q
VMATHDEF void quat_from_vec3(quat q, vec3 v); // q = v ({x, y, z, 0})
VMATHDEF void vec3_from_quat(vec3 v, quat q); // v = q ({x, y, z})
VMATHDEF void quat_mul(quat r, quat u, quat v); // r = u * v (quat mult.)
VMATHDEF void quat_mul_vec3(quat r, quat q, vec3 v); // quat_mul with v = {x, y, z, 0}
VMATHDEF void vec3_mul_quat(quat r, vec3 v, quat q);
VMATHDEF void vec3_rotate_quat(vec3 r, quat q, vec3 v);
VMATHDEF void mat3_from_quat(mat3 m, quat q); // q is unit
VMATHDEF void quat_from_mat3(quat q, mat3 m);

// mat3x4 m from scale:s rotation:r translation:t
VMATHDEF void mat3x4_from_srt(mat3x4 m, vec3 s, quat r, vec3 t);
// z direction: dir
VMATHDEF void mat3_from_dir(mat3 m, vec3 dir);



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
VMATHDEF scalar vec4_dot(vec4 u, vec4 v)
{
	return u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3];
}

VMATHDEF scalar vec4_lensqr(vec4 v)
{
	return v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
}

VMATHDEF scalar vec4_len(vec4 v)
{
	return vsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
}

VMATHDEF scalar vec4_len1(vec4 v)
{
	return vabs(v[0]) + vabs(v[1]) + vabs(v[2]) + vabs(v[3]);
}

VMATHDEF scalar vec4_distsqr(vec4 u, vec4 v)
{
	scalar t, d;
	t = u[0] - v[0]; d  = t * t;
	t = u[1] - v[1]; d += t * t;
	t = u[2] - v[2]; d += t * t;
	t = u[3] - v[3]; d += t * t;
	return d;
}

VMATHDEF scalar vec4_dist(vec4 u, vec4 v)
{
	scalar t, d;
	t = u[0] - v[0]; d  = t * t;
	t = u[1] - v[1]; d += t * t;
	t = u[2] - v[2]; d += t * t;
	t = u[3] - v[3]; d += t * t;
	return vsqrt(d);
}

VMATHDEF scalar vec4_dist1(vec4 u, vec4 v)
{
	return vabs(u[0] - v[0]) + vabs(u[1] - v[1])
	     + vabs(u[2] - v[2]) + vabs(u[3] - v[3]);
}

VMATHDEF void vec4_zero(vec4 v)
{
	v[0] = v[1] = v[2] = v[3] = 0;
}

VMATHDEF void vec4_makeunit(vec4 v)
{
	scalar s = VP(1) / vsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
	v[0] *= s; v[1] *= s; v[2] *= s; v[3] *= s;
}

VMATHDEF void vec4_setcomp(vec4 v, scalar x, scalar y, scalar z, scalar w)
{
	v[0] = x; v[1] = y; v[2] = z; v[3] = w;
}

VMATHDEF void vec4_setarr(vec4 v, scalar *u)
{
	v[0] = u[0]; v[1] = u[1]; v[2] = u[2]; v[3] = u[3];
}

VMATHDEF void vec4_copy(vec4 r, vec4 v)
{
	r[0] = v[0]; r[1] = v[1];
	r[2] = v[2]; r[3] = v[3];
}

VMATHDEF void vec4_from_vec2(vec4 r, vec2 v)
{
	r[0] = v[0]; r[1] = v[1];
	r[2] = r[3] = 0; 
}

VMATHDEF void vec4_from_vec3(vec4 r, vec3 v)
{
	r[0] = v[0]; r[1] = v[1]; r[2] = v[2];
	r[3] = 0; 
}

VMATHDEF void vec4_from_mat2(vec4 v, mat2 m)
{
	v[0] = m[0][0]; v[1] = m[0][1];
	v[2] = m[1][0]; v[3] = m[1][1];
}

VMATHDEF void vec4_smul(vec4 r, scalar s, vec4 v)
{
	r[0] = s * v[0]; r[1] = s * v[1];
	r[2] = s * v[2]; r[3] = s * v[3];
}

VMATHDEF void vec4_negate(vec4 r, vec4 v)
{
	r[0] = -(v[0]); r[1] = -(v[1]);
	r[2] = -(v[2]); r[3] = -(v[3]);
}

VMATHDEF void vec4_normalize(vec4 r, vec4 v)
{
	scalar s = VP(1) / vsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
	r[0] = v[0] * s; r[1] = v[1] * s;
	r[2] = v[2] * s; r[3] = v[3] * s;
}

VMATHDEF void vec4_add(vec4 r, vec4 u, vec4 v)
{
	r[0] = u[0] + v[0]; r[1] = u[1] + v[1];
	r[2] = u[2] + v[2]; r[3] = u[3] + v[3];
}

VMATHDEF void vec4_sub(vec4 r, vec4 u, vec4 v)
{
	r[0] = u[0] - v[0]; r[1] = u[1] - v[1];
	r[2] = u[2] - v[2]; r[3] = u[3] - v[3];
}

VMATHDEF void vec4_tmul(vec4 r, vec4 u, vec4 v)
{
	r[0] = u[0] * v[0]; r[1] = u[1] * v[1];
	r[2] = u[2] * v[2]; r[3] = u[3] * v[3];
}

VMATHDEF void vec4_ma(vec4 r, vec4 u, scalar s, vec4 v)
{
	r[0] = u[0] + s * v[0]; r[1] = u[1] + s * v[1];
	r[2] = u[2] + s * v[2]; r[3] = u[3] + s * v[3];
}

VMATHDEF void vec4_combine(vec4 r, scalar s, vec4 u, scalar t, vec4 v)
{
	r[0] = s * u[0] + t * v[0]; r[1] = s * u[1] + t * v[1];
	r[2] = s * u[2] + t * v[2]; r[3] = s * u[3] + t * v[3];
}

VMATHDEF void vec4_lerp(vec4 r, vec4 u, vec4 v, scalar t)
{
	scalar s = VP(1) - t;
	r[0] = s * u[0] + t * v[0]; r[1] = s * u[1] + t * v[1];
	r[2] = s * u[2] + t * v[2]; r[3] = s * u[3] + t * v[3];
}

VMATHDEF void vec4_project(vec4 r, vec4 v, vec4 n)
{
	// scalar t = vec4_dot(n, v) / vec4_dot(n, n);
	vec4_ma(r, v, -vec4_dot(n, v), n);
}

VMATHDEF void vec4_reflect(vec4 r, vec4 v, vec4 n)
{
	// scalar t = vec4_dot(n, v) / vec4_dot(n, n);
	vec4_ma(r, v, -VP(2) * vec4_dot(n, v), n);
}


// -------------------
// --- common vec3 ---
// -------------------
VMATHDEF scalar vec3_dot(vec3 u, vec3 v)
{
	return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

VMATHDEF scalar vec3_lensqr(vec3 v)
{
	return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

VMATHDEF scalar vec3_len(vec3 v)
{
	return vsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

VMATHDEF scalar vec3_len1(vec3 v)
{
	return vabs(v[0]) + vabs(v[1]) + vabs(v[2]);
}

VMATHDEF scalar vec3_distsqr(vec3 u, vec3 v)
{
	scalar t, d;
	t = u[0] - v[0]; d  = t * t;
	t = u[1] - v[1]; d += t * t;
	t = u[2] - v[2]; d += t * t;
	return d;
}

VMATHDEF scalar vec3_dist(vec3 u, vec3 v)
{
	scalar t, d;
	t = u[0] - v[0]; d  = t * t;
	t = u[1] - v[1]; d += t * t;
	t = u[2] - v[2]; d += t * t;
	return vsqrt(d);
}

VMATHDEF scalar vec3_dist1(vec3 u, vec3 v)
{
	return vabs(u[0] - v[0]) + vabs(u[1] - v[1]) + vabs(u[2] - v[2]);
}

VMATHDEF void vec3_zero(vec3 v)
{
	v[0] = v[1] = v[2] = 0;
}

VMATHDEF void vec3_makeunit(vec3 v)
{
	scalar s = VP(1) / vsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	v[0] *= s; v[1] *= s; v[2] *= s;
}

VMATHDEF void vec3_setcomp(vec3 v, scalar x, scalar y, scalar z)
{
	v[0] = x; v[1] = y; v[2] = z;
}

VMATHDEF void vec3_setarr(vec3 v, scalar *u)
{
	v[0] = u[0]; v[1] = u[1]; v[2] = u[2];
}

VMATHDEF void vec3_copy(vec3 r, vec3 v)
{
	r[0] = v[0]; r[1] = v[1]; r[2] = v[2];
}

VMATHDEF void vec3_from_vec2(vec3 r, vec2 v)
{
	r[0] = v[0]; r[1] = v[1]; r[2] = 0; 
}

VMATHDEF void vec3_from_vec4(vec3 r, vec4 v)
{
	r[0] = v[0]; r[1] = v[1]; r[2] = v[2]; 
}

VMATHDEF void vec3_smul(vec3 r, scalar s, vec3 v)
{
	r[0] = s * v[0]; r[1] = s * v[1]; r[2] = s * v[2];
}

VMATHDEF void vec3_negate(vec3 r, vec3 v)
{
	r[0] = -(v[0]); r[1] = -(v[1]); r[2] = -(v[2]);
}

VMATHDEF void vec3_normalize(vec3 r, vec3 v)
{
	scalar s = VP(1) / vsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	r[0] = v[0] * s; r[1] = v[1] * s; r[2] = v[2] * s;
}

VMATHDEF void vec3_add(vec3 r, vec3 u, vec3 v)
{
	r[0] = u[0] + v[0]; r[1] = u[1] + v[1]; r[2] = u[2] + v[2];
}

VMATHDEF void vec3_sub(vec3 r, vec3 u, vec3 v)
{
	r[0] = u[0] - v[0]; r[1] = u[1] - v[1]; r[2] = u[2] - v[2];
}

VMATHDEF void vec3_tmul(vec3 r, vec3 u, vec3 v)
{
	r[0] = u[0] * v[0]; r[1] = u[1] * v[1]; r[2] = u[2] * v[2];
}

VMATHDEF void vec3_ma(vec3 r, vec3 u, scalar t, vec3 v)
{
	r[0] = u[0] + t * v[0]; r[1] = u[1] + t * v[1]; r[2] = u[2] + t * v[2];
}

VMATHDEF void vec3_combine(vec3 r, scalar s, vec3 u, scalar t, vec3 v)
{
	r[0] = s * u[0] + t * v[0]; r[1] = s * u[1] + t * v[1];
	r[2] = s * u[2] + t * v[2];
}

VMATHDEF void vec3_lerp(vec3 r, vec3 u, vec3 v, scalar t)
{
	scalar s = VP(1) - t;
	r[0] = s * u[0] + t * v[0]; r[1] = s * u[1] + t * v[1];
	r[2] = s * u[2] + t * v[2];
}

VMATHDEF void vec3_project(vec3 r, vec3 v, vec3 n)
{
	vec3_ma(r, v, -vec3_dot(n, v), n);
}

VMATHDEF void vec3_reflect(vec3 r, vec3 v, vec3 n)
{
	vec3_ma(r, v, -VP(2) * vec3_dot(n, v), n);
}


// -------------------
// --- common vec2 ---
// -------------------
VMATHDEF scalar vec2_dot(vec2 u, vec2 v)
{
	return u[0] * v[0] + u[1] * v[1];
}

VMATHDEF scalar vec2_lensqr(vec2 v)
{
	return v[0] * v[0] + v[1] * v[1];
}

VMATHDEF scalar vec2_len(vec2 v)
{
	return vsqrt(v[0] * v[0] + v[1] * v[1]);
}

VMATHDEF scalar vec2_len1(vec2 v)
{
	return vabs(v[0]) + vabs(v[1]);
}

VMATHDEF scalar vec2_distsqr(vec2 u, vec2 v)
{
	scalar t, d;
	t = u[0] - v[0]; d  = t * t;
	t = u[1] - v[1]; d += t * t;
	return d;
}

VMATHDEF scalar vec2_dist(vec2 u, vec2 v)
{
	scalar t, d;
	t = u[0] - v[0]; d  = t * t;
	t = u[1] - v[1]; d += t * t;
	return vsqrt(d);
}

VMATHDEF scalar vec2_dist1(vec2 u, vec2 v)
{
	return vabs(u[0] - v[0]) + vabs(u[1] - v[1]);
}

VMATHDEF void vec2_zero(vec2 v)
{
	v[0] = v[1] = 0;
}

VMATHDEF void vec2_makeunit(vec2 v)
{
	scalar s = VP(1) / vsqrt(v[0] * v[0] + v[1] * v[1]);
	v[0] *= s; v[1] *= s;
}

VMATHDEF void vec2_setcomp(vec2 v, scalar x, scalar y)
{
	v[0] = x; v[1] = y;
}

VMATHDEF void vec2_setarr(vec2 v, scalar *u)
{
	v[0] = u[0]; v[1] = u[1];
}

VMATHDEF void vec2_copy(vec2 r, vec2 v)
{
	r[0] = v[0]; r[1] = v[1];
}

VMATHDEF void vec2_from_vec3(vec2 r, vec3 v)
{
	r[0] = v[0]; r[1] = v[1];
}

VMATHDEF void vec2_from_vec4(vec2 r, vec4 v)
{
	r[0] = v[0]; r[1] = v[1]; 
}

VMATHDEF void vec2_smul(vec2 r, scalar s, vec2 v)
{
	r[0] = s * v[0]; r[1] = s * v[1];
}

VMATHDEF void vec2_negate(vec2 r, vec2 v)
{
	r[0] = -(v[0]); r[1] = -(v[1]);
}

VMATHDEF void vec2_normalize(vec2 r, vec2 v)
{
	scalar s = VP(1) / vsqrt(v[0] * v[0] + v[1] * v[1]);
	r[0] = v[0] * s; r[1] = v[1] * s;
}

VMATHDEF void vec2_add(vec2 r, vec2 u, vec2 v)
{
	r[0] = u[0] + v[0]; r[1] = u[1] + v[1];
}

VMATHDEF void vec2_sub(vec2 r, vec2 u, vec2 v)
{
	r[0] = u[0] - v[0]; r[1] = u[1] - v[1];
}

VMATHDEF void vec2_tmul(vec2 r, vec2 u, vec2 v)
{
	r[0] = u[0] * v[0]; r[1] = u[1] * v[1];
}

VMATHDEF void vec2_ma(vec2 r, vec2 u, scalar t, vec2 v)
{
	r[0] = u[0] + t * v[0]; r[1] = u[1] + t * v[1];
}

VMATHDEF void vec2_combine(vec2 r, scalar s, vec2 u, scalar t, vec2 v)
{
	r[0] = s * u[0] + t * v[0]; r[1] = s * u[1] + t * v[1];
}

VMATHDEF void vec2_lerp(vec2 r, vec2 u, vec2 v, scalar t)
{
	scalar s = VP(1) - t;
	r[0] = s * u[0] + t * v[0]; r[1] = s * u[1] + t * v[1];
}

VMATHDEF void vec2_project(vec2 r, vec2 v, vec2 n)
{
	vec2_ma(r, v, -vec2_dot(n, v), n);
}

VMATHDEF void vec2_reflect(vec2 r, vec2 v, vec2 n)
{
	vec2_ma(r, v, -VP(2) * vec2_dot(n, v), n);
}


// -------------------
// ---- vec other ----
// -------------------
VMATHDEF scalar vec2_cross(vec2 u, vec2 v)
{
	return u[0] * v[1] - u[1] * v[0];
}

VMATHDEF void vec3_cross(vec3 r, vec3 u, vec3 v)
{
	scalar x, y, z;
	x = u[1] * v[2] - u[2] * v[1];
	y = u[0] * v[2] - u[2] * v[0];
	z = u[0] * v[1] - u[1] * v[0];
	r[0] = x; r[1] = -y; r[2] = z;
}

VMATHDEF void vec4_cross(vec4 r, vec4 u, vec4 v)
{
	scalar x, y, z;
	x = u[1] * v[2] - u[2] * v[1];
	y = u[0] * v[2] - u[2] * v[0];
	z = u[0] * v[1] - u[1] * v[0];
	r[0] = x; r[1] = -y; r[2] = z; r[3] = 0;
}

VMATHDEF void plane_from_points(plane p, vec3 a, vec3 b, vec3 c)
{
	vec3 n, u, v;
	vec3_sub(u, b, a);
	vec3_sub(v, c, a);
	vec3_cross(n, u, v);
	vec3_makeunit(n);
	p[0] = n[0]; p[1] = n[1]; p[2] = n[2];
	p[3] = vec3_dot(a, n);
}

VMATHDEF void vec3_plane_project(vec3 r, vec3 v, plane p)
{
	vec3 n;
	vec3_from_vec4(n, p);
	vec3_project(r, n, v);
}

VMATHDEF void vec3_plane_reflect(vec3 r, vec3 v, plane p)
{
	vec3 n;
	vec3_from_vec4(n, p);
	vec3_reflect(r, n, v);
}

VMATHDEF void plane_makeunit(plane p)
{
	scalar s = vsqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
	p[3] *= s;
	s = VP(1) / s;
	p[0] *= s; p[1] *= s; p[2] *= s;
}

VMATHDEF void plane_normalize(plane r, plane p)
{
	scalar s = vsqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
	r[3] = s * p[3];
	s = VP(1) / s;
	r[0] = s * p[0]; r[1] = s * p[1]; r[2] = s * p[2];
}


// ---------------------
// --- mat4x4 common ---
// ---------------------
VMATHDEF void mat4_zero(mat4 m)
{
	vmemset(m, 0, sizeof(mat4));
}

VMATHDEF void mat4_id(mat4 m)
{
	vmemset(m, 0, sizeof(mat4));
	m[0][0] = m[1][1] = m[2][2] = m[3][3] = VP(1);
}

VMATHDEF void mat4_t(mat4 m)
{
	scalar t;
	t = m[0][1]; m[0][1] = m[1][0]; m[1][0] = t;
	t = m[0][2]; m[0][2] = m[2][0]; m[2][0] = t;
	t = m[0][3]; m[0][3] = m[3][0]; m[3][0] = t;
	
	t = m[1][2]; m[1][2] = m[2][1]; m[2][1] = t;
	t = m[1][3]; m[1][3] = m[3][1]; m[3][1] = t;
	
	t = m[2][3]; m[2][3] = m[3][2]; m[3][2] = t;
}

VMATHDEF void mat4_eq_rotx(mat4 m, scalar a)
{
	vmemset(m, 0, sizeof(mat4));
	scalar c = vcos(a), s = vsin(a);
	m[0][0] = m[3][3] = VP(1);
	m[1][1] = c; m[1][2] = -s;
	m[2][1] = s; m[2][2] =  c;
}

VMATHDEF void mat4_eq_roty(mat4 m, scalar a)
{
	vmemset(m, 0, sizeof(mat4));
	scalar c = vcos(a), s = vsin(a);
	m[1][1] = m[3][3] = VP(1);
	m[0][0] =  c; m[0][2] = s;
	m[2][0] = -s; m[2][2] = c;
}

VMATHDEF void mat4_eq_rotz(mat4 m, scalar a)
{
	vmemset(m, 0, sizeof(mat4));
	scalar c = vcos(a), s = vsin(a);
	m[2][2] = m[3][3] = VP(1);
	m[0][0] = c; m[0][1] = -s;
	m[1][0] = s; m[1][1] =  c;
}

VMATHDEF void mat4_eq_rotaxis(mat4 m, vec3 u, scalar a)
{
	scalar s = vsin(a), c = vcos(a);
	scalar d = 1 - c;

	scalar dxx = d * u[0] * u[0];
	scalar dyy = d * u[1] * u[1];
	scalar dzz = d * u[2] * u[2];
	scalar dxy = d * u[0] * u[1];
	scalar dxz = d * u[0] * u[2];
	scalar dyz = d * u[1] * u[2];
	
	scalar sx = s * u[0], sy = s * u[1], sz = s * u[2];

	m[0][0] = c + dxx;  m[0][1] = dxy - sz; m[0][2] = dxz + sy; m[0][3] = 0;
	m[1][0] = dxy + sz; m[1][1] = c + dyy;  m[1][2] = dyz - sx; m[1][3] = 0;
	m[2][0] = dxz - sy; m[2][1] = dyz + sx; m[2][2] = c + dzz;  m[2][3] = 0;
	m[3][0] = 0;        m[3][1] = 0;        m[3][2] = 0;        m[3][3] = VP(1);
}

VMATHDEF void mat4_eq_scale(mat4 m, scalar sx, scalar sy, scalar sz)
{
	vmemset(m, 0, sizeof(mat4));
	m[0][0] = sx; m[1][1] = sy; m[2][2] = sz; m[3][3] = VP(1);
}

VMATHDEF void mat4_eq_shear(mat4 m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy)
{
	m[0][0] = VP(1); m[0][1] = xy;    m[0][2] = xz;    m[0][3] = 0;
	m[1][0] = yx;    m[1][1] = VP(1); m[1][2] = yz;    m[1][3] = 0;
	m[2][0] = zx;    m[2][1] = zy;    m[2][2] = VP(1); m[2][3] = 0;
	m[3][0] = 0; m[3][1] = 0; m[3][2] = 0; m[3][3] = VP(1);
}

VMATHDEF void mat4_eq_trans(mat4 m, scalar tx, scalar ty, scalar tz)
{
	vmemset(m, 0, sizeof(mat4));
	m[0][0] = m[1][1] = m[2][2] = m[3][3] = VP(1);
	m[0][3] = tx; m[1][3] = ty; m[2][3] = tz;
}

VMATHDEF void mat4_setarr(mat4 m, scalar *s)
{
	vmemcpy(m, s, sizeof(mat4));
}

VMATHDEF void mat4_copy(mat4 r, mat4 m)
{
	vmemcpy(r, m, sizeof(mat4));
}

VMATHDEF void mat4_from_mat3(mat4 r, mat3 m)
{
	vec4_from_vec3(r[0], m[0]);
	vec4_from_vec3(r[1], m[1]);
	vec4_from_vec3(r[2], m[2]);
	vec4_setcomp(r[3], 0, 0, 0, VP(1));
}

VMATHDEF void mat4_from_mat3x4(mat4 r, mat3x4 m)
{
	vec4_copy(r[0], m[0]);
	vec4_copy(r[1], m[1]);
	vec4_copy(r[2], m[2]);
	vec4_setcomp(r[3], 0, 0, 0, VP(1));
}

VMATHDEF void mat4_setrows(mat4 m, vec4 ra, vec4 rb, vec4 rc, vec4 rd)
{
	vec4_copy(m[0], ra);
	vec4_copy(m[1], rb);
	vec4_copy(m[2], rc);
	vec4_copy(m[3], rd);
}

VMATHDEF void mat4_setcols(mat4 m, vec4 cx, vec4 cy, vec4 cz, vec4 cw)
{
	vec4_setcomp(m[0], cx[0], cy[0], cz[0], cw[0]);
	vec4_setcomp(m[1], cx[1], cy[1], cz[1], cw[1]);
	vec4_setcomp(m[2], cx[2], cy[2], cz[2], cw[2]);
	vec4_setcomp(m[3], cx[3], cy[3], cz[3], cw[3]);
}

VMATHDEF void mat4_setrowa(mat4 m, vec4 v) { vec4_copy(m[0], v); }
VMATHDEF void mat4_setrowb(mat4 m, vec4 v) { vec4_copy(m[1], v); }
VMATHDEF void mat4_setrowc(mat4 m, vec4 v) { vec4_copy(m[2], v); }
VMATHDEF void mat4_setrowd(mat4 m, vec4 v) { vec4_copy(m[3], v); }

VMATHDEF void mat4_setcolx(mat4 m, vec4 v)
{
	m[0][0] = v[0]; m[1][0] = v[1]; m[2][0] = v[2]; m[3][0] = v[3];
}

VMATHDEF void mat4_setcoly(mat4 m, vec4 v)
{
	m[0][1] = v[0]; m[1][1] = v[1]; m[2][1] = v[2]; m[3][1] = v[3];
}

VMATHDEF void mat4_setcolz(mat4 m, vec4 v)
{
	m[0][2] = v[0]; m[1][2] = v[1]; m[2][2] = v[2]; m[3][2] = v[3];
}

VMATHDEF void mat4_setcolw(mat4 m, vec4 v)
{
	m[0][3] = v[0]; m[1][3] = v[1]; m[2][3] = v[2]; m[3][3] = v[3];
}

VMATHDEF void mat4_rowa(vec4 r, mat4 m) { vec4_copy(r, m[0]); }
VMATHDEF void mat4_rowb(vec4 r, mat4 m) { vec4_copy(r, m[1]); }
VMATHDEF void mat4_rowc(vec4 r, mat4 m) { vec4_copy(r, m[2]); }
VMATHDEF void mat4_rowd(vec4 r, mat4 m) { vec4_copy(r, m[3]); }

VMATHDEF void mat4_colx(vec4 r, mat4 m) { vec4_setcomp(r, m[0][0], m[1][0], m[2][0], m[3][0]); }
VMATHDEF void mat4_coly(vec4 r, mat4 m) { vec4_setcomp(r, m[0][1], m[1][1], m[2][1], m[3][1]); }
VMATHDEF void mat4_colz(vec4 r, mat4 m) { vec4_setcomp(r, m[0][2], m[1][2], m[2][2], m[3][2]); }
VMATHDEF void mat4_colw(vec4 r, mat4 m) { vec4_setcomp(r, m[0][3], m[1][3], m[2][3], m[3][3]); }

VMATHDEF void mat4_smul(mat4 r, scalar s, mat4 m)
{
	vec4_smul(r[0], s, m[0]);
	vec4_smul(r[1], s, m[1]);
	vec4_smul(r[2], s, m[2]);
	vec4_smul(r[3], s, m[3]);
}

VMATHDEF void mat4_mulrows(mat4 r, mat4 m, scalar a, scalar b, scalar c, scalar d)
{
	vec4_smul(r[0], a, m[0]);
	vec4_smul(r[1], b, m[1]);
	vec4_smul(r[2], c, m[2]);
	vec4_smul(r[3], d, m[3]);
}

VMATHDEF void mat4_mulrowv(mat4 r, mat4 m, vec4 v)
{
	vec4_smul(r[0], v[0], m[0]);
	vec4_smul(r[1], v[1], m[1]);
	vec4_smul(r[2], v[2], m[2]);
	vec4_smul(r[3], v[3], m[3]);
}

VMATHDEF void mat4_negate(mat4 r, mat4 m)
{
	vec4_negate(r[0], m[0]);
	vec4_negate(r[1], m[1]);
	vec4_negate(r[2], m[2]);
	vec4_negate(r[3], m[3]);
}

VMATHDEF void mat4_add(mat4 r, mat4 f, mat4 g)
{
	vec4_add(r[0], f[0], g[0]);
	vec4_add(r[1], f[1], g[1]);
	vec4_add(r[2], f[2], g[2]);
	vec4_add(r[3], f[3], g[3]);
}

VMATHDEF void mat4_sub(mat4 r, mat4 f, mat4 g)
{
	vec4_sub(r[0], f[0], g[0]);
	vec4_sub(r[1], f[1], g[1]);
	vec4_sub(r[2], f[2], g[2]);
	vec4_sub(r[3], f[3], g[3]);
}

VMATHDEF void mat4_tmul(mat4 r, mat4 f, mat4 g)
{
	vec4_tmul(r[0], f[0], g[0]);
	vec4_tmul(r[1], f[1], g[1]);
	vec4_tmul(r[2], f[2], g[2]);
	vec4_tmul(r[3], f[3], g[3]);
}

VMATHDEF void _vec4_mul_mat4(vec4 r, vec4 v, mat4 m)
{
	r[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3];
	r[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3];
	r[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3];
	r[3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3];
}

VMATHDEF void mat4_mul(mat4 r, mat4 f, mat4 g)
{
	mat4 m;
	_vec4_mul_mat4(m[0], f[0], g);
	_vec4_mul_mat4(m[1], f[1], g);
	_vec4_mul_mat4(m[2], f[2], g);
	_vec4_mul_mat4(m[3], f[3], g);
	vmemcpy(r, m, sizeof m);
}

VMATHDEF void mat4_ma(mat4 r, mat4 f, scalar t, mat4 g)
{
	vec4_ma(r[0], f[0], t, g[0]);
	vec4_ma(r[1], f[1], t, g[1]);
	vec4_ma(r[2], f[2], t, g[2]);
}

VMATHDEF void mat4_combine(mat4 r, scalar s, mat4 f, scalar t, mat4 g)
{
	vec4_combine(r[0], s, f[0], t, g[0]);
	vec4_combine(r[1], s, f[1], t, g[1]);
	vec4_combine(r[2], s, f[2], t, g[2]);
}

VMATHDEF void mat4_lerp(mat4 r, mat4 f, mat4 g, scalar t)
{
	scalar s = 1 - t;
	vec4_combine(r[0], s, f[0], t, g[0]);
	vec4_combine(r[1], s, f[1], t, g[1]);
	vec4_combine(r[2], s, f[2], t, g[2]);
}

VMATHDEF void vec4_mul_mat4(vec4 r, vec4 v, mat4 m)
{
	scalar x, y, z, w;
	x = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3];
	y = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3];
	z = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3];
	w = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3];
	r[0] = x; r[1] = y; r[2] = z; r[3] = w;
}

VMATHDEF void mat4_mul_vec4(vec4 r, mat4 m, vec4 v)
{
	scalar x, y, z, w;
	x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3];
	y = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3];
	z = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3];
	w = m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3];
	r[0] = x; r[1] = y; r[2] = z; r[3] = w;
}

VMATHDEF void mat4_mul_vec3(vec3 r, mat4 m, vec3 v)
{
	scalar x, y, z;
	x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3];
	y = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3];
	z = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3];
	r[0] = x; r[1] = y; r[2] = z;
}

VMATHDEF void mat4_transpose(mat4 r, mat4 m)
{
	scalar t;
	t = m[0][1]; r[0][1] = m[1][0]; r[1][0] = t;
	t = m[0][2]; r[0][2] = m[2][0]; r[2][0] = t;
	t = m[0][3]; r[0][3] = m[3][0]; r[3][0] = t;
	
	t = m[1][2]; r[1][2] = m[2][1]; r[2][1] = t;
	t = m[1][3]; r[1][3] = m[3][1]; r[3][1] = t;
	
	t = m[2][3]; r[2][3] = m[3][2]; r[3][2] = t;
	
	r[0][0] = m[0][0]; r[1][1] = m[1][1]; r[2][2] = m[2][2]; r[3][3] = m[3][3];
}

VMATHDEF void mat4_inverse(mat4 r, mat4 m)
{
	scalar s0, s1, s2, s3, s4, s5;
	scalar c0, c1, c2, c3, c4, c5;
	scalar invdet;
	
	c0 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
	c1 = m[0][0] * m[1][2] - m[0][2] * m[1][0];
	c2 = m[0][0] * m[1][3] - m[0][3] * m[1][0];
	c3 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
	c4 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
	c5 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
	
	s0 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
	s1 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
	s2 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
	s3 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
	s4 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
	s5 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
	
	invdet = VP(1) / (c0 * s5 - c1 * s4 + c2 * s3 + c3 * s2 - c4 * s1 + c5 * s0);
	
	r[0][0] =  (m[1][1] * s5 - m[1][2] * s4 + m[1][3] * s3) * invdet;
	r[0][1] = -(m[0][1] * s5 - m[0][2] * s4 + m[0][3] * s3) * invdet;
	r[0][2] =  (m[3][1] * c5 - m[3][2] * c4 + m[3][3] * c3) * invdet;
	r[0][3] = -(m[2][1] * c5 - m[2][2] * c4 + m[2][3] * c3) * invdet;
	
	r[1][0] = -(m[1][0] * s5 - m[1][2] * s2 + m[1][3] * s1) * invdet;
	r[1][1] =  (m[0][0] * s5 - m[0][2] * s2 + m[0][3] * s1) * invdet;
	r[1][2] = -(m[3][0] * c5 - m[3][2] * c2 + m[3][3] * c1) * invdet;
	r[1][3] =  (m[2][0] * c5 - m[2][2] * c2 + m[2][3] * c1) * invdet;
	
	r[2][0] =  (m[1][0] * s4 - m[1][1] * s2 + m[1][3] * s0) * invdet;
	r[2][1] = -(m[0][0] * s4 - m[0][1] * s2 + m[0][3] * s0) * invdet;
	r[2][2] =  (m[3][0] * c4 - m[3][1] * c2 + m[3][3] * c0) * invdet;
	r[2][3] = -(m[2][0] * c4 - m[2][1] * c2 + m[2][3] * c0) * invdet;
	
	r[3][0] = -(m[1][0] * s3 - m[1][1] * s1 + m[1][2] * s0) * invdet;
	r[3][1] =  (m[0][0] * s3 - m[0][1] * s1 + m[0][2] * s0) * invdet;
	r[3][2] = -(m[3][0] * c3 - m[3][1] * c1 + m[3][2] * c0) * invdet;
	r[3][3] =  (m[2][0] * c3 - m[2][1] * c1 + m[2][2] * c0) * invdet;
}

VMATHDEF void mat4_vtrace(vec4 v, mat4 m)
{
	v[0] = m[0][0]; v[1] = m[1][1];
	v[2] = m[2][2]; v[3] = m[3][3];
}

VMATHDEF scalar mat4_trace(mat4 m)
{
	return m[0][0] + m[1][1] + m[2][2] + m[3][3];
}

VMATHDEF scalar mat4_det3(mat4 m)
{
	return
	  m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
	- m[2][1] * (m[0][0] * m[1][2] - m[0][2] * m[1][0])
	+ m[2][2] * (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
}

VMATHDEF scalar mat4_det(mat4 m)
{
	return // (c0 * s5 - c1 * s4 + c2 * s3 + c3 * s2 - c4 * s1 + c5 * s0) from mat4_inverse
	  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
	- (m[0][0] * m[1][2] - m[0][2] * m[1][0]) * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
	+ (m[0][0] * m[1][3] - m[0][3] * m[1][0]) * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
	+ (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
	- (m[0][1] * m[1][3] - m[0][3] * m[1][1]) * (m[2][0] * m[3][2] - m[2][2] * m[3][0])
	+ (m[0][2] * m[1][3] - m[0][3] * m[1][2]) * (m[2][0] * m[3][1] - m[2][1] * m[3][0]);
}


// ---------------------
// --- mat3x4 common ---
// ---------------------
VMATHDEF void mat3x4_zero(mat3x4 m)
{
	vmemset(m, 0, sizeof(mat3x4));
}

VMATHDEF void mat3x4_id(mat3x4 m)
{
	vmemset(m, 0, sizeof(mat3x4));
	m[0][0] = m[1][1] = m[2][2] = VP(1);
}

VMATHDEF void mat3x4_t(mat3x4 m)
{
	scalar t;
	t = m[0][1]; m[0][1] = m[1][0]; m[1][0] = t;
	t = m[0][2]; m[0][2] = m[2][0]; m[2][0] = t;
	
	t = m[1][2]; m[1][2] = m[2][1]; m[2][1] = t;
}

VMATHDEF void mat3x4_eq_rotx(mat3x4 m, scalar a)
{
	vmemset(m, 0, sizeof(mat3x4));
	scalar c = vcos(a), s = vsin(a);
	m[0][0] = VP(1);
	m[1][1] = c; m[1][2] = -s;
	m[2][1] = s; m[2][2] =  c;
}

VMATHDEF void mat3x4_eq_roty(mat3x4 m, scalar a)
{
	vmemset(m, 0, sizeof(mat3x4));
	scalar c = vcos(a), s = vsin(a);
	m[1][1] = VP(1);
	m[0][0] =  c; m[0][2] = s;
	m[2][0] = -s; m[2][2] = c;
}

VMATHDEF void mat3x4_eq_rotz(mat3x4 m, scalar a)
{
	vmemset(m, 0, sizeof(mat3x4));
	scalar c = vcos(a), s = vsin(a);
	m[2][2] = VP(1);
	m[0][0] = c; m[0][1] = -s;
	m[1][0] = s; m[1][1] =  c;
}

VMATHDEF void mat3x4_eq_rotaxis(mat3x4 m, vec3 u, scalar a)
{
	scalar s = vsin(a), c = vcos(a);
	scalar d = 1 - c;

	scalar dxx = d * u[0] * u[0];
	scalar dyy = d * u[1] * u[1];
	scalar dzz = d * u[2] * u[2];
	scalar dxy = d * u[0] * u[1];
	scalar dxz = d * u[0] * u[2];
	scalar dyz = d * u[1] * u[2];
	
	scalar sx = s * u[0], sy = s * u[1], sz = s * u[2];

	m[0][0] = c + dxx;  m[0][1] = dxy - sz; m[0][2] = dxz + sy; m[0][3] = 0;
	m[1][0] = dxy + sz; m[1][1] = c + dyy;  m[1][2] = dyz - sx; m[1][3] = 0;
	m[2][0] = dxz - sy; m[2][1] = dyz + sx; m[2][2] = c + dzz;  m[2][3] = 0;
}

VMATHDEF void mat3x4_eq_scale(mat3x4 m, scalar sx, scalar sy, scalar sz)
{
	vmemset(m, 0, sizeof(mat3x4));
	m[0][0] = sx; m[1][1] = sy; m[2][2] = sz;
}

VMATHDEF void mat3x4_eq_shear(mat3x4 m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy)
{
	m[0][0] = VP(1); m[0][1] = xy;    m[0][2] = xz;    m[0][3] = 0;
	m[1][0] = yx;    m[1][1] = VP(1); m[1][2] = yz;    m[1][3] = 0;
	m[2][0] = zx;    m[2][1] = zy;    m[2][2] = VP(1); m[2][3] = 0;
}

VMATHDEF void mat3x4_eq_trans(mat3x4 m, scalar tx, scalar ty, scalar tz)
{
	vmemset(m, 0, sizeof(mat3x4));
	m[0][0] = m[1][1] = m[2][2] = VP(1);
	m[0][3] = tx; m[1][3] = ty; m[2][3] = tz;
}

VMATHDEF void mat3x4_setarr(mat3x4 m, scalar *s)
{
	vmemcpy(m, s, sizeof(mat3x4));
}

VMATHDEF void mat3x4_copy(mat3x4 r, mat3x4 m)
{
	vmemcpy(r, m, sizeof(mat3x4));
}

VMATHDEF void mat3x4_from_mat3(mat3x4 r, mat3 m)
{
	vec4_from_vec3(r[0], m[0]);
	vec4_from_vec3(r[1], m[1]);
	vec4_from_vec3(r[2], m[2]);
}

VMATHDEF void mat3x4_from_mat4(mat3x4 r, mat4 m)
{
	vec4_copy(r[0], m[0]);
	vec4_copy(r[1], m[1]);
	vec4_copy(r[2], m[2]);
}

VMATHDEF void mat3x4_setrows(mat3x4 m, vec4 ra, vec4 rb, vec4 rc)
{
	vec4_copy(m[0], ra);
	vec4_copy(m[1], rb);
	vec4_copy(m[2], rc);
}

VMATHDEF void mat3x4_setcols(mat3x4 m, vec3 cx, vec3 cy, vec3 cz, vec3 cw)
{
	vec4_setcomp(m[0], cx[0], cy[0], cz[0], cw[0]);
	vec4_setcomp(m[1], cx[1], cy[1], cz[1], cw[1]);
	vec4_setcomp(m[2], cx[2], cy[2], cz[2], cw[2]);
}

VMATHDEF void mat3x4_setrowa(mat3x4 m, vec4 v) { vec4_copy(m[0], v); }
VMATHDEF void mat3x4_setrowb(mat3x4 m, vec4 v) { vec4_copy(m[1], v); }
VMATHDEF void mat3x4_setrowc(mat3x4 m, vec4 v) { vec4_copy(m[2], v); }

VMATHDEF void mat3x4_setcolx(mat3x4 m, vec3 v)
{
	m[0][0] = v[0]; m[1][0] = v[1]; m[2][0] = v[2];
}

VMATHDEF void mat3x4_setcoly(mat3x4 m, vec3 v)
{
	m[0][1] = v[0]; m[1][1] = v[1]; m[2][1] = v[2];
}

VMATHDEF void mat3x4_setcolz(mat3x4 m, vec3 v)
{
	m[0][2] = v[0]; m[1][2] = v[1]; m[2][2] = v[2];
}

VMATHDEF void mat3x4_setcolw(mat3x4 m, vec3 v)
{
	m[0][3] = v[0]; m[1][3] = v[1]; m[2][3] = v[2];
}

VMATHDEF void mat3x4_rowa(vec4 r, mat3x4 m) { vec4_copy(r, m[0]); }
VMATHDEF void mat3x4_rowb(vec4 r, mat3x4 m) { vec4_copy(r, m[1]); }
VMATHDEF void mat3x4_rowc(vec4 r, mat3x4 m) { vec4_copy(r, m[2]); }

VMATHDEF void mat3x4_colx(vec3 r, mat3x4 m) { vec3_setcomp(r, m[0][0], m[1][0], m[2][0]); } //d[0] = 0
VMATHDEF void mat3x4_coly(vec3 r, mat3x4 m) { vec3_setcomp(r, m[0][1], m[1][1], m[2][1]); } //d[1] = 0
VMATHDEF void mat3x4_colz(vec3 r, mat3x4 m) { vec3_setcomp(r, m[0][2], m[1][2], m[2][2]); } //d[2] = 0
VMATHDEF void mat3x4_colw(vec3 r, mat3x4 m) { vec3_setcomp(r, m[0][3], m[1][3], m[2][3]); } //d[3] = 1

VMATHDEF void mat3x4_smul(mat3x4 r, scalar s, mat3x4 m)
{
	vec4_smul(r[0], s, m[0]);
	vec4_smul(r[1], s, m[1]);
	vec4_smul(r[2], s, m[2]);
}

VMATHDEF void mat3x4_mulrows(mat3x4 r, mat3x4 m, scalar a, scalar b, scalar c)
{
	vec4_smul(r[0], a, m[0]);
	vec4_smul(r[1], b, m[1]);
	vec4_smul(r[2], c, m[2]);
}

VMATHDEF void mat3x4_mulrowv(mat3x4 r, mat3x4 m, vec3 v)
{
	vec4_smul(r[0], v[0], m[0]);
	vec4_smul(r[1], v[1], m[1]);
	vec4_smul(r[2], v[2], m[2]);
}

VMATHDEF void mat3x4_negate(mat3x4 r, mat3x4 m)
{
	vec4_negate(r[0], m[0]);
	vec4_negate(r[1], m[1]);
	vec4_negate(r[2], m[2]);
}

VMATHDEF void mat3x4_add(mat3x4 r, mat3x4 f, mat3x4 g)
{
	vec4_add(r[0], f[0], g[0]);
	vec4_add(r[1], f[1], g[1]);
	vec4_add(r[2], f[2], g[2]);
}

VMATHDEF void mat3x4_sub(mat3x4 r, mat3x4 f, mat3x4 g)
{
	vec4_sub(r[0], f[0], g[0]);
	vec4_sub(r[1], f[1], g[1]);
	vec4_sub(r[2], f[2], g[2]);
}

VMATHDEF void mat3x4_tmul(mat3x4 r, mat3x4 f, mat3x4 g)
{
	vec4_tmul(r[0], f[0], g[0]);
	vec4_tmul(r[1], f[1], g[1]);
	vec4_tmul(r[2], f[2], g[2]);
}

VMATHDEF void _vec4_mul_mat3x4(vec4 r, vec4 v, mat3x4 m)
{
	r[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2];
	r[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2];
	r[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2];
	r[3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + v[3];
}

VMATHDEF void mat3x4_mul(mat3x4 r, mat3x4 f, mat3x4 g)
{
	mat3x4 m;
	_vec4_mul_mat3x4(m[0], f[0], g);
	_vec4_mul_mat3x4(m[1], f[1], g);
	_vec4_mul_mat3x4(m[2], f[2], g);
	vmemcpy(r, m, sizeof m);
}

VMATHDEF void mat3x4_ma(mat3x4 r, mat3x4 f, scalar t, mat3x4 g)
{
	vec4_ma(r[0], f[0], t, g[0]);
	vec4_ma(r[1], f[1], t, g[1]);
	vec4_ma(r[2], f[2], t, g[2]);
}

VMATHDEF void mat3x4_combine(mat3x4 r, scalar s, mat3x4 f, scalar t, mat3x4 g)
{
	vec4_combine(r[0], s, f[0], t, g[0]);
	vec4_combine(r[1], s, f[1], t, g[1]);
	vec4_combine(r[2], s, f[2], t, g[2]);
}

VMATHDEF void mat3x4_lerp(mat3x4 r, mat3x4 f, mat3x4 g, scalar t)
{
	scalar s = 1 - t;
	vec4_combine(r[0], s, f[0], t, g[0]);
	vec4_combine(r[1], s, f[1], t, g[1]);
	vec4_combine(r[2], s, f[2], t, g[2]);
}

VMATHDEF void vec4_mul_mat3x4(vec4 r, vec4 v, mat3x4 m)
{
	scalar x, y, z, w;
	x = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2];
	y = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2];
	z = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2];
	w = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + v[3];
	r[0] = x; r[1] = y; r[2] = z; r[3] = w;
}

VMATHDEF void mat3x4_mul_vec4(vec4 r, mat3x4 m, vec4 v)
{
	scalar x, y, z, w;
	x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3];
	y = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3];
	z = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3];
	w = v[3];
	r[0] = x; r[1] = y; r[2] = z; r[3] = w;
}

VMATHDEF void mat3x4_mul_vec3(vec3 r, mat3x4 m, vec3 v)
{
	scalar x, y, z;
	x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3];
	y = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3];
	z = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3];
	r[0] = x; r[1] = y; r[2] = z;
}

VMATHDEF void mat3x4_transpose(mat3x4 r, mat3x4 m)
{
	scalar t;
	t = m[0][1]; r[0][1] = m[1][0]; r[1][0] = t;
	t = m[0][2]; r[0][2] = m[2][0]; r[2][0] = t;
	
	t = m[1][2]; r[1][2] = m[2][1]; r[2][1] = t;
	
	r[0][0] = m[0][0]; r[1][1] = m[1][1]; r[2][2] = m[2][2];
	r[0][3] = m[0][3]; r[1][3] = m[1][3]; r[2][3] = m[2][3];
}

VMATHDEF void mat3x4_inverse(mat3x4 r, mat3x4 m)
{
	scalar c0, c1, c2, c3, c4, c5;
	scalar invdet;
	
	c0 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
	c1 = m[0][0] * m[1][2] - m[0][2] * m[1][0];
	c2 = m[0][0] * m[1][3] - m[0][3] * m[1][0];
	c3 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
	c4 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
	c5 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
	
	invdet = VP(1) / (m[2][0] * c3 - m[2][1] * c1 + m[2][2] * c0);
	
	r[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invdet;
	r[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * invdet;
	r[0][2] =  (c3) * invdet;
	r[0][3] = -(m[2][1] * c5 - m[2][2] * c4 + m[2][3] * c3) * invdet;
	
	r[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * invdet;
	r[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
	r[1][2] = -(c1) * invdet;
	r[1][3] =  (m[2][0] * c5 - m[2][2] * c2 + m[2][3] * c1) * invdet;
	
	r[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invdet;
	r[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * invdet;
	r[2][2] =  (c0) * invdet;
	r[2][3] = -(m[2][0] * c4 - m[2][1] * c2 + m[2][3] * c0) * invdet;
}

VMATHDEF void mat3x4_vtrace(vec3 v, mat3x4 m)
{
	v[0] = m[0][0]; v[1] = m[1][1]; v[2] = m[2][2];
}

VMATHDEF scalar mat3x4_trace(mat3x4 m)
{
	return m[0][0] + m[1][1] + m[2][2];
}

VMATHDEF scalar mat3x4_det(mat3x4 m)
{
	return
	  m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
	- m[2][1] * (m[0][0] * m[1][2] - m[0][2] * m[1][0])
	+ m[2][2] * (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
}


// ---------------------
// --- mat3x3 common ---
// ---------------------
VMATHDEF void mat3_zero(mat3 m)
{
	vmemset(m, 0, sizeof(mat3));
}

VMATHDEF void mat3_id(mat3 m)
{
	vmemset(m, 0, sizeof(mat3));
	m[0][0] = m[1][1] = m[2][2] = VP(1);
}

VMATHDEF void mat3_t(mat3 m)
{
	scalar t;
	t = m[0][1]; m[0][1] = m[1][0]; m[1][0] = t;
	t = m[0][2]; m[0][2] = m[2][0]; m[2][0] = t;
	
	t = m[1][2]; m[1][2] = m[2][1]; m[2][1] = t;
}

VMATHDEF void mat3_eq_rotx(mat3 m, scalar a)
{
	vmemset(m, 0, sizeof(mat3));
	scalar c = vcos(a), s = vsin(a);
	m[0][0] = VP(1);
	m[1][1] = c; m[1][2] = -s;
	m[2][1] = s; m[2][2] =  c;
}

VMATHDEF void mat3_eq_roty(mat3 m, scalar a)
{
	vmemset(m, 0, sizeof(mat3));
	scalar c = vcos(a), s = vsin(a);
	m[1][1] = VP(1);
	m[0][0] =  c; m[0][2] = s;
	m[2][0] = -s; m[2][2] = c;
}

VMATHDEF void mat3_eq_rotz(mat3 m, scalar a)
{
	vmemset(m, 0, sizeof(mat3));
	scalar c = vcos(a), s = vsin(a);
	m[2][2] = VP(1);
	m[0][0] = c; m[0][1] = -s;
	m[1][0] = s; m[1][1] =  c;
}

VMATHDEF void mat3_eq_rot2d(mat3 m, scalar a)
{
	mat3_eq_rotz(m, a);
}

VMATHDEF void mat3_eq_rotaxis(mat3 m, vec3 u, scalar a)
{
	scalar s = vsin(a), c = vcos(a);
	scalar d = 1 - c;

	scalar dxx = d * u[0] * u[0];
	scalar dyy = d * u[1] * u[1];
	scalar dzz = d * u[2] * u[2];
	scalar dxy = d * u[0] * u[1];
	scalar dxz = d * u[0] * u[2];
	scalar dyz = d * u[1] * u[2];
	
	scalar sx = s * u[0], sy = s * u[1], sz = s * u[2];

	m[0][0] = c + dxx;  m[0][1] = dxy - sz; m[0][2] = dxz + sy;
	m[1][0] = dxy + sz; m[1][1] = c + dyy;  m[1][2] = dyz - sx;
	m[2][0] = dxz - sy; m[2][1] = dyz + sx; m[2][2] = c + dzz;
}

VMATHDEF void mat3_eq_trans2d(mat3 m, scalar tx, scalar ty)
{
	vmemset(m, 0, sizeof(mat3));
	m[0][0] = m[1][1] = m[2][2] = VP(1);
	m[0][2] = tx; m[1][2] = ty;
}

VMATHDEF void mat3_eq_scale(mat3 m, scalar sx, scalar sy, scalar sz)
{
	vmemset(m, 0, sizeof(mat3));
	m[0][0] = sx; m[1][1] = sy; m[2][2] = sz;
}

VMATHDEF void mat3_eq_scale2d(mat3 m, scalar sx, scalar sy)
{
	vmemset(m, 0, sizeof(mat3));
	m[0][0] = sx; m[1][1] = sy; m[2][2] = VP(1);
}

VMATHDEF void mat3_eq_shear(mat3 m, scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy)
{
	m[0][0] = VP(1); m[0][1] = xy;    m[0][2] = xz;
	m[1][0] = yx;    m[1][1] = VP(1); m[1][2] = yz;
	m[2][0] = zx;    m[2][1] = zy;    m[2][2] = VP(1);
}

VMATHDEF void mat3_eq_shear2d(mat3 m, scalar x, scalar y)
{
	m[0][0] = VP(1); m[0][1] = x;     m[0][2] = 0;
	m[1][0] = y;     m[1][1] = VP(1); m[1][2] = 0;
	m[2][0] = 0; m[2][1] = 0; m[2][2] = VP(1);
}

VMATHDEF void mat3_setarr(mat3 m, scalar *s)
{
	vmemcpy(m, s, sizeof(mat3));
}

VMATHDEF void mat3_copy(mat3 r, mat3 m)
{
	vmemcpy(r, m, sizeof(mat3));
}

VMATHDEF void mat3_from_mat4(mat3 r, mat4 m)
{
	vec3_from_vec4(r[0], m[0]);
	vec3_from_vec4(r[1], m[1]);
	vec3_from_vec4(r[2], m[2]);
}

VMATHDEF void mat3_from_mat3x4(mat3 r, mat3x4 m)
{
	vec3_from_vec4(r[0], m[0]);
	vec3_from_vec4(r[1], m[1]);
	vec3_from_vec4(r[2], m[2]);
}

VMATHDEF void mat3_setrows(mat3 m, vec3 ra, vec3 rb, vec3 rc)
{
	vec3_copy(m[0], ra);
	vec3_copy(m[1], rb);
	vec3_copy(m[2], rc);
}

VMATHDEF void mat3_setcols(mat3 m, vec3 cx, vec3 cy, vec3 cz)
{
	vec3_setcomp(m[0], cx[0], cy[0], cz[0]);
	vec3_setcomp(m[1], cx[1], cy[1], cz[1]);
	vec3_setcomp(m[2], cx[2], cy[2], cz[2]);
}

VMATHDEF void mat3_setrowa(mat3 m, vec3 v) { vec3_copy(m[0], v); }
VMATHDEF void mat3_setrowb(mat3 m, vec3 v) { vec3_copy(m[1], v); }
VMATHDEF void mat3_setrowc(mat3 m, vec3 v) { vec3_copy(m[2], v); }

VMATHDEF void mat3_setcolx(mat3 m, vec3 v)
{
	m[0][0] = v[0]; m[1][0] = v[1]; m[2][0] = v[2];
}

VMATHDEF void mat3_setcoly(mat3 m, vec3 v)
{
	m[0][1] = v[0]; m[1][1] = v[1]; m[2][1] = v[2];
}

VMATHDEF void mat3_setcolz(mat3 m, vec3 v)
{
	m[0][2] = v[0]; m[1][2] = v[1]; m[2][2] = v[2];
}

VMATHDEF void mat3_rowa(vec3 r, mat3 m) { vec3_copy(r, m[0]); }
VMATHDEF void mat3_rowb(vec3 r, mat3 m) { vec3_copy(r, m[1]); }
VMATHDEF void mat3_rowc(vec3 r, mat3 m) { vec3_copy(r, m[2]); }

VMATHDEF void mat3_colx(vec3 r, mat3 m) { vec3_setcomp(r, m[0][0], m[1][0], m[2][0]); }
VMATHDEF void mat3_coly(vec3 r, mat3 m) { vec3_setcomp(r, m[0][1], m[1][1], m[2][1]); }
VMATHDEF void mat3_colz(vec3 r, mat3 m) { vec3_setcomp(r, m[0][2], m[1][2], m[2][2]); }

VMATHDEF void mat3_smul(mat3 r, scalar s, mat3 m)
{
	vec3_smul(r[0], s, m[0]);
	vec3_smul(r[1], s, m[1]);
	vec3_smul(r[2], s, m[2]);
}

VMATHDEF void mat3_mulrows(mat3 r, mat3 m, scalar a, scalar b, scalar c)
{
	vec3_smul(r[0], a, m[0]);
	vec3_smul(r[1], b, m[1]);
	vec3_smul(r[2], c, m[2]);
}

VMATHDEF void mat3_mulrowv(mat3 r, mat3 m, vec3 v)
{
	vec3_smul(r[0], v[0], m[0]);
	vec3_smul(r[1], v[1], m[1]);
	vec3_smul(r[2], v[2], m[2]);
}

VMATHDEF void mat3_negate(mat3 r, mat3 m)
{
	vec3_negate(r[0], m[0]);
	vec3_negate(r[1], m[1]);
	vec3_negate(r[2], m[2]);
}

VMATHDEF void mat3_add(mat3 r, mat3 f, mat3 g)
{
	vec3_add(r[0], f[0], g[0]);
	vec3_add(r[1], f[1], g[1]);
	vec3_add(r[2], f[2], g[2]);
}

VMATHDEF void mat3_sub(mat3 r, mat3 f, mat3 g)
{
	vec3_sub(r[0], f[0], g[0]);
	vec3_sub(r[1], f[1], g[1]);
	vec3_sub(r[2], f[2], g[2]);
}

VMATHDEF void mat3_tmul(mat3 r, mat3 f, mat3 g)
{
	vec3_tmul(r[0], f[0], g[0]);
	vec3_tmul(r[1], f[1], g[1]);
	vec3_tmul(r[2], f[2], g[2]);
}

VMATHDEF void _vec3_mul_mat3(vec3 r, vec3 v, mat3 m)
{
	r[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2];
	r[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2];
	r[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2];
}

VMATHDEF void mat3_mul(mat3 r, mat3 f, mat3 g)
{
	mat3 m;
	_vec3_mul_mat3(m[0], f[0], g);
	_vec3_mul_mat3(m[1], f[1], g);
	_vec3_mul_mat3(m[2], f[2], g);
	vmemcpy(r, m, sizeof m);
}

VMATHDEF void mat3_ma(mat3 r, mat3 f, scalar t, mat3 g)
{
	vec3_ma(r[0], f[0], t, g[0]);
	vec3_ma(r[1], f[1], t, g[1]);
	vec3_ma(r[2], f[2], t, g[2]);
}

VMATHDEF void mat3_combine(mat3 r, scalar s, mat3 f, scalar t, mat3 g)
{
	vec3_combine(r[0], s, f[0], t, g[0]);
	vec3_combine(r[1], s, f[1], t, g[1]);
	vec3_combine(r[2], s, f[2], t, g[2]);
}

VMATHDEF void mat3_lerp(mat3 r, mat3 f, mat3 g, scalar t)
{
	scalar s = VP(1) - t;
	vec3_combine(r[0], s, f[0], t, g[0]);
	vec3_combine(r[1], s, f[1], t, g[1]);
	vec3_combine(r[2], s, f[2], t, g[2]);
}

VMATHDEF void vec3_mul_mat3(vec3 r, vec3 v, mat3 m)
{
	scalar x, y, z;
	x = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2];
	y = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2];
	z = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2];
	r[0] = x; r[1] = y; r[2] = z;
}

VMATHDEF void mat3_mul_vec3(vec3 r, mat3 m, vec3 v)
{
	scalar x, y, z;
	x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2];
	y = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2];
	z = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2];
	r[0] = x; r[1] = y; r[2] = z;
}

VMATHDEF void mat3_mul_vec2(vec2 r, mat3 m, vec2 v)
{
	scalar x, y;
	x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2];
	y = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2];
	r[0] = x; r[1] = y;
}

VMATHDEF void mat3_transpose(mat3 r, mat3 m)
{
	scalar t;
	t = m[0][1]; r[0][1] = m[1][0]; r[1][0] = t;
	t = m[0][2]; r[0][2] = m[2][0]; r[2][0] = t;
	
	t = m[1][2]; r[1][2] = m[2][1]; r[2][1] = t;
	
	r[0][0] = m[0][0]; r[1][1] = m[1][1]; r[2][2] = m[2][2];
}

VMATHDEF void mat3_inverse(mat3 r, mat3 m)
{
	scalar c0, c1, c3;
	scalar invdet;
	
	c0 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
	c1 = m[0][0] * m[1][2] - m[0][2] * m[1][0];
	c3 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
	
	invdet = VP(1) / (m[2][0] * c3 - m[2][1] * c1 + m[2][2] * c0);
	
	r[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invdet;
	r[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * invdet;
	r[0][2] =  (c3) * invdet;
	
	r[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * invdet;
	r[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
	r[1][2] = -(c1) * invdet;
	
	r[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invdet;
	r[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * invdet;
	r[2][2] =  (c0) * invdet;
}

VMATHDEF void mat3_vtrace(vec3 v, mat3 m)
{
	v[0] = m[0][0]; v[1] = m[1][1]; v[2] = m[2][2];
}

VMATHDEF scalar mat3_trace(mat3 m)
{
	return m[0][0] + m[1][1] + m[2][2];
}

VMATHDEF scalar mat3_det(mat3 m)
{
	return
	  m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
	- m[2][1] * (m[0][0] * m[1][2] - m[0][2] * m[1][0])
	+ m[2][2] * (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
}


// ---------------------
// --- mat2x2 common ---
// ---------------------
VMATHDEF void mat2_zero(mat2 m)
{
	m[0][0] = m[0][1] = m[1][0] = m[1][1] = 0;
}

VMATHDEF void mat2_id(mat2 m)
{
	m[0][1] = m[1][0] = 0;
	m[0][0] = m[1][1] = VP(1);
}

VMATHDEF void mat2_t(mat2 m)
{
	scalar t;
	t = m[0][1]; m[0][1] = m[1][0]; m[1][0] = t;
}

VMATHDEF void mat2_rot(mat2 m, scalar a)
{
	scalar c = vcos(a), s = vsin(a);
	m[0][0] = c; m[0][1] = -s;
	m[1][0] = s; m[1][1] =  c;
}

VMATHDEF void mat2_eq_scale(mat2 m, scalar sx, scalar sy)
{
	m[0][1] = m[1][0] = 0;
	m[0][0] = sx; m[1][1] = sy;
}

VMATHDEF void mat2_eq_shear(mat2 m, scalar x, scalar y)
{
	m[0][0] = VP(1); m[0][1] = x;
	m[1][0] = y;     m[1][1] = VP(1);
}

VMATHDEF void mat2_setarr(mat2 m, scalar *s)
{
	vmemcpy(m, s, sizeof(mat2));
}

VMATHDEF void mat2_copy(mat2 r, mat2 m)
{
	vmemcpy(r, m, sizeof(mat2));
}

VMATHDEF void mat2_from_vec4(mat2 m, vec4 v)
{
	vec2_setcomp(m[0], v[0], v[1]);
	vec2_setcomp(m[1], v[2], v[3]);
}

VMATHDEF void mat2_setrows(mat2 m, vec2 a, vec2 b)
{
	m[0][0] = a[0]; m[0][1] = a[1];
	m[1][0] = b[0]; m[1][1] = b[1];
}

VMATHDEF void mat2_setcols(mat2 m, vec2 a, vec2 b)
{
	vec2_setcomp(m[0], a[0], b[1]);
	vec2_setcomp(m[1], a[1], b[1]);
}

VMATHDEF void mat2_setrowa(mat2 m, vec2 v) { vec2_copy(m[0], v); }
VMATHDEF void mat2_setrowb(mat2 m, vec2 v) { vec2_copy(m[1], v); }

VMATHDEF void mat2_setcolx(mat2 m, vec2 v) { m[0][0] = v[0]; m[1][0] = v[1]; }
VMATHDEF void mat2_setcoly(mat2 m, vec2 v) { m[0][1] = v[0]; m[1][1] = v[1]; }

VMATHDEF void mat2_rowa(vec2 r, mat2 m) { vec2_copy(r, m[0]); }
VMATHDEF void mat2_rowb(vec2 r, mat2 m) { vec2_copy(r, m[1]); }

VMATHDEF void mat2_colx(vec2 r, mat2 m) { vec2_setcomp(r, m[0][0], m[1][0]); }
VMATHDEF void mat2_coly(vec2 r, mat2 m) { vec2_setcomp(r, m[0][1], m[1][1]); }

VMATHDEF void mat2_smul(mat2 r, scalar s, mat2 m)
{
	vec2_smul(r[0], s, m[0]);
	vec2_smul(r[1], s, m[1]);
}

VMATHDEF void mat2_mulrows(mat2 r, mat2 m, scalar a, scalar b)
{
	vec2_smul(r[0], a, m[0]);
	vec2_smul(r[1], b, m[1]);
}

VMATHDEF void mat2_mulrowv(mat2 r, mat2 m, vec2 v)
{
	vec2_smul(r[0], v[0], m[0]);
	vec2_smul(r[1], v[1], m[1]);
}

VMATHDEF void mat2_negate(mat2 r, mat2 m)
{
	vec2_negate(r[0], m[0]);
	vec2_negate(r[1], m[1]);
}

VMATHDEF void mat2_add(mat2 r, mat2 f, mat2 g)
{
	vec2_add(r[0], f[0], g[0]);
	vec2_add(r[1], f[1], g[1]);
}

VMATHDEF void mat2_sub(mat2 r, mat2 f, mat2 g)
{
	vec2_sub(r[0], f[0], g[0]);
	vec2_sub(r[1], f[1], g[1]);
}

VMATHDEF void mat2_tmul(mat2 r, mat2 f, mat2 g)
{
	vec2_tmul(r[0], f[0], g[0]);
	vec2_tmul(r[1], f[1], g[1]);
}

VMATHDEF void _vec2_mul_mat2(vec2 r, vec2 v, mat2 m)
{
	r[0] = m[0][0] * v[0] + m[1][0] * v[1];
	r[1] = m[0][1] * v[0] + m[1][1] * v[1];
}

VMATHDEF void mat2_mul(mat2 r, mat2 f, mat2 g)
{
	mat2 m;
	_vec2_mul_mat2(m[0], f[0], g);
	_vec2_mul_mat2(m[1], f[1], g);
	vmemcpy(r, m, sizeof m);
}

VMATHDEF void mat2_ma(mat2 r, mat2 f, scalar t, mat2 g)
{
	vec2_ma(r[0], f[0], t, g[0]);
	vec2_ma(r[1], f[1], t, g[1]);
}

VMATHDEF void mat2_combine(mat2 r, scalar s, mat2 f, scalar t, mat2 g)
{
	vec2_combine(r[0], s, f[0], t, g[0]);
	vec2_combine(r[1], s, f[1], t, g[1]);
}

VMATHDEF void mat2_lerp(mat2 r, mat2 f, mat2 g, scalar t)
{
	scalar s = VP(1) - t;
	vec2_combine(r[0], s, f[0], t, g[0]);
	vec2_combine(r[1], s, f[1], t, g[1]);
}

VMATHDEF void vec2_mul_mat2(vec2 r, vec2 v, mat2 m)
{
	scalar x, y;
	x = m[0][0] * v[0] + m[1][0] * v[1];
	y = m[0][1] * v[0] + m[1][1] * v[1];
	r[0] = x; r[1] = y;
}

VMATHDEF void mat2_mul_vec2(vec2 r, mat2 m, vec2 v)
{
	scalar x, y;
	x = m[0][0] * v[0] + m[0][1] * v[1];
	y = m[1][0] * v[0] + m[1][1] * v[1];
	r[0] = x; r[1] = y;
}

VMATHDEF void mat2_transpose(mat2 r, mat2 m)
{
	scalar t;
	t = m[0][1]; r[0][1] = m[1][0]; r[1][0] = t;
	
	r[0][0] = m[0][0]; r[1][1] = m[1][1];
}

VMATHDEF void mat2_inverse(mat2 r, mat2 m)
{
	scalar invdet = VP(1) / (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
	
	r[0][0] =  m[1][1] * invdet;
	r[0][1] = -m[0][1] * invdet;
	
	r[1][0] = -m[1][0] * invdet;
	r[1][1] =  m[0][0] * invdet;
}

VMATHDEF void mat2_vtrace(vec2 v, mat2 m)
{
	v[0] = m[0][0]; v[1] = m[1][1];;
}

VMATHDEF scalar mat2_trace(mat2 m)
{
	return m[0][0] + m[1][1];
}

VMATHDEF scalar mat2_det(mat2 m)
{
	return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

// -------------------
// --- quat common ---
// -------------------
VMATHDEF void quat_id(quat v)
{
	v[0] = v[1] = v[2] = 0; v[3] = VP(1);
}

VMATHDEF void quat_from_rot(quat r, vec3 v, scalar a)
{
	a = a / VP(2);
	scalar s = vsin(a) / vsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	r[0] = v[0] * s; r[1] = v[1] * s; r[2] = v[2] * s;
	r[3] = vcos(a);
}

VMATHDEF void quat_inv(quat r, quat v)
{
	scalar s = VP(1) / vsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
	r[0] = -v[0] * s; r[1] = -v[1] * s;
	r[2] = -v[2] * s; r[3] =  v[3] * s;
}

VMATHDEF void quat_conj(quat r, quat v)
{
	r[0] = -v[0]; r[1] = -v[1];
	r[2] = -v[2]; r[3] =  v[3];
}

VMATHDEF void quat_from_vec3(quat r, vec3 v)
{
	r[0] = v[0]; r[1] = v[1];
	r[2] = v[2]; r[3] = 0;
}

VMATHDEF void vec3_from_quat(vec3 r, quat v)
{
	r[0] = v[0]; r[1] = v[1]; r[2] = v[2];
}


VMATHDEF void quat_mul(quat r, quat u, quat v)
{
	scalar x, y, z, w;
	w = u[3] * v[3] - u[0] * v[0] - u[1] * v[1] - u[2] * v[2];
	x = u[3] * v[0] + u[0] * v[3] + u[1] * v[2] - u[2] * v[1];
	y = u[3] * v[1] - u[0] * v[2] + u[1] * v[3] + u[2] * v[0];
	z = u[3] * v[2] + u[0] * v[1] - u[1] * v[0] + u[2] * v[3];
	r[0] = x; r[1] = y; r[2] = z; r[3] = w;
}

VMATHDEF void quat_mul_vec3(quat r, quat u, vec3 v)
{
	scalar x, y, z, w;
	w = - u[0] * v[0] - u[1] * v[1] - u[2] * v[2];
	x = u[3] * v[0] + u[1] * v[2] - u[2] * v[1];
	y = u[3] * v[1] - u[0] * v[2] + u[2] * v[0];
	z = u[3] * v[2] + u[0] * v[1] - u[1] * v[0];
	r[0] = x; r[1] = y; r[2] = z; r[3] = w;
}

VMATHDEF void vec3_mul_quat(quat r, vec3 u, quat v)
{
	scalar x, y, z, w;
	w = - u[0] * v[0] - u[1] * v[1] - u[2] * v[2];
	x = + u[0] * v[3] + u[1] * v[2] - u[2] * v[1];
	y = - u[0] * v[2] + u[1] * v[3] + u[2] * v[0];
	z = + u[0] * v[1] - u[1] * v[0] + u[2] * v[3];
	r[0] = x; r[1] = y; r[2] = z; r[3] = w;
}

VMATHDEF void vec3_rotate_quat(vec3 r, quat q, vec3 v)
{
	quat p;
	quat_inv(p, q);
	vec3_mul_quat(p, v, p); // p = v * inv(q)
	quat_mul(p, q, p); // p = q * v * inv(q)
	r[0] = p[0]; r[1] = p[1]; r[2] = p[2];
}

VMATHDEF void mat3_from_quat(mat3 m, quat q)
{
	scalar x = q[0], y = q[1], z = q[2], w = q[3];
	
	m[0][0] = VP(1) - VP(2) * (y * y + z * z);
	m[0][1] = VP(2) * (x * y - z * w);
	m[0][2] = VP(2) * (x * z + y * w);
	
	m[1][0] = VP(2) * (x * y + z * w);
	m[1][1] = VP(1) - VP(2) * (x * x + z * z);
	m[1][2] = VP(2) * (y * z - x * w);
	
	m[2][0] = VP(2) * (x * z - y * w);
	m[2][1] = VP(2) * (y * z + x * w);
	m[2][2] = VP(1) - VP(2) * (x * x + y * y);
}

VMATHDEF void mat3x4_from_srt(mat3x4 m, vec3 s, quat r, vec3 t)
{
	scalar x = r[0], y = r[1], z = r[2], w = r[3];
	
	m[0][0] = s[0] * (VP(1) - VP(2) * (y * y + z * z));
	m[0][1] = s[1] * VP(2) * (x * y - z * w);
	m[0][2] = s[2] * VP(2) * (x * z + y * w);
	m[0][3] = t[0];
	
	m[1][0] = s[0] * VP(2) * (x * y + z * w);
	m[1][1] = s[1] * (VP(1) - VP(2) * (x * x + z * z));
	m[1][2] = s[2] * VP(2) * (y * z - x * w);
	m[1][3] = t[1];
	
	m[2][0] = s[0] * VP(2) * (x * z - y * w);
	m[2][1] = s[1] * VP(2) * (y * z + x * w);
	m[2][2] = s[2] * (VP(1) - VP(2) * (x * x + y * y));
	m[2][3] = t[2];
}

VMATHDEF void mat3x4_inverse_srt(mat3x4 r, mat3x4 m)
{
	scalar n, x, y, z;
	
	x = m[0][0]; y = m[1][0]; z = m[2][0];
	n = VP(1) / (x * x + y * y + z * z);
	r[0][0] = x * n; r[0][1] = y * n; r[0][2] = z * n;
	
	x = m[0][1]; y = m[1][1]; z = m[2][1];
	n = VP(1) / (x * x + y * y + z * z);
	r[1][0] = x * n; r[1][1] = y * n; r[1][2] = z * n;
	
	x = m[0][2]; y = m[1][2]; z = m[2][2];
	n = VP(1) / (x * x + y * y + z * z);
	r[2][0] = x * n; r[2][1] = y * n; r[2][2] = z * n;
	
	x = m[0][3]; y = m[1][3]; z = m[2][3];
	
	r[0][3] = -(r[0][0] * x + r[0][1] * y + r[0][2] * z);
	r[1][3] = -(r[1][0] * x + r[1][1] * y + r[1][2] * z);
	r[2][3] = -(r[2][0] * x + r[2][1] * y + r[2][2] * z);
}


VMATHDEF void quat_from_mat3(quat q, mat3 m)
{
	scalar t, x,y,z,w;
	scalar tr0 = m[0][0], tr1 = m[1][1], tr2 = m[2][2];
	if (tr2 < 0)
	{
		if (tr0 > tr1)
		{
			t = VP(1) + tr0 - tr1 - tr2;
			x = t;               y = m[0][1] + m[1][0];
			z = m[2][0] + m[0][2]; w = m[2][1] - m[1][2];
		}
		else
		{
			t = VP(1) - tr0 + tr1 - tr2;
			x = m[0][1] + m[1][0]; y = t;
			z = m[1][2] + m[2][1]; w = m[0][2] - m[2][0];
		}
	}
	else
	{
		if (tr0 < -tr1)
		{
			t = VP(1) - tr0 - tr1 + tr2;
			x = m[2][0] + m[0][2];  y = m[1][2] + m[2][1];
			z = t;                w = m[1][0] - m[0][1];
		}
		else
		{
			t = VP(1) + tr0 + tr1 + tr2;
			x = m[2][1] - m[1][2]; y = m[0][2] - m[2][0];
			y = m[1][0] - m[0][1]; w = t;
		}
	}
	
	t = VP(1) / (VP(2) * vsqrt(t));
	q[0] = t * x; q[1] = t * y; q[2] = t * z; q[3] = t * w;
}

VMATHDEF void mat3_from_dir(mat3 m, vec3 dir)
{
	scalar x = vabs(dir[0]), y = vabs(dir[1]), z = vabs(dir[2]);
	vec3 u;
	vec3_zero(u);
	if (x <= y && x <= z)
		u[0] = VP(1);
	else if (y <= x && y <= z)
		u[1] = VP(1);
	else
		u[2] = VP(1);
	
	vec3_copy(m[2], dir);
	vec3_makeunit(m[2]);
	
	vec3_project(m[0], u, m[2]);
	vec3_makeunit(m[0]);
	
	vec3_cross(m[1], m[0], m[2]);
	
	mat3_t(m);
}

#endif // VMATH_IMPLEMENTATION


