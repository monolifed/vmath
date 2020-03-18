#include <math.h>

#ifndef VEC_DOUBLEPREC
typedef float scalar;
#else
typedef double scalar;
#endif

#ifndef VEC_NOSTDMEM
#include <string.h>
#define vmemset memset
#define vmemcpy memcpy
#else
#error "not implemented"
#endif


typedef struct vec2_s
{
	scalar x, y;
} vec2;

typedef struct vec3_s
{
	scalar x, y, z;
} vec3;

// plane: x,y,z is unit normal
//        w is (signed) distance in normal direction
// quat: x,y,z is i,j,k components, w is real comp.

typedef struct vec4_s
{
	scalar x, y, z, w;
} vec4, quat, plane;

// a,b,c,d are rows of the matrix
// i.e. row major matrix

typedef struct mat3x3_s
{
	vec3 a, b, c;
} mat3, mat3x3;

typedef struct mat3x4_s
{
	vec4 a, b, c; // d = {0, 0, 0, 1}
} mat3x4;

typedef struct mat4x4_s
{
	vec4 a, b, c, d;
} mat4, mat4x4;






#ifndef VEC_DOUBLEPREC
#define vsqrt sqrtf
#define vabs fabsf
#define vsin sinf
#define vcos cosf
#else
#define vsqrt sqrt
#define vabs fabs
#define vsin sin
#define vcos cos
#endif


// -------------------
// --- common vec2 ---
// -------------------

// = v · v
scalar vec2_lensqr(vec2 *v)
{
	return v->x * v->x + v->y * v->y;
}

// = |v|
scalar vec2_len(vec2 *v)
{
	return vsqrt(v->x * v->x + v->y * v->y);
}

// = d1(v, 0)
scalar vec2_len1(vec2 *v)
{
	return vabs(v->x) + vabs(v->y);
}

// v = 0
void vec2_zero(vec2 *v)
{
	v->x = v->y = 0;
}

// v = {x, y}
void vec2_set(vec2 *v, scalar x, scalar y)
{
	v->x = x; v->y = y;
}

// v = {...}
void vec2_seta(vec2 *v, scalar *u)
{
	v->x = u[0]; v->y = u[1];
}

// r = v
void vec2_copy(vec2 *r, vec2 *v)
{
	r->x = v->x; r->y = v->y;
}

// r = s * v
void vec2_scale(vec2 *r, scalar s, vec2 *v)
{
	r->x = s * v->x; r->y = s * v->y;
}

// r = -v
void vec2_neg(vec2 *r, vec2 *v)
{
	r->x = -(v->x); r->y = -(v->y);
}

// r = v / |v| (possible division by 0)
void vec2_normalize(vec2 *r, vec2 *v)
{
	scalar s = vsqrt(v->x * v->x + v->y * v->y);
	r->x = v->x / s; r->y = v->y / s;
}

// r = u + v
void vec2_add(vec2 *r, vec2 *u, vec2 *v)
{
	r->x = u->x + v->x; r->y = u->y + v->y;
}

// r = u - v
void vec2_sub(vec2 *r, vec2 *u, vec2 *v)
{
	r->x = u->x - v->x; r->y = u->y - v->y;
}

// r = u * v (term-wise)
void vec2_tmul(vec2 *r, vec2 *u, vec2 *v)
{
	r->x = u->x * v->x; r->y = u->y * v->y;
}

// = u · v
scalar vec2_dot(vec2 *u, vec2 *v)
{
	return u->x * v->x + u->y * v->y;
}

// r = u + s * v
void vec2_ma(vec2 *r, vec2 *u, scalar s, vec2 *v)
{
	r->x = u->x + s * v->x; r->y = u->y + s * v->y;
}

// = |u - v|^2
scalar vec2_distsqr(vec2 *u, vec2 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	return d;
}

// |u - v|
scalar vec2_dist(vec2 *u, vec2 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	return vsqrt(d);
}

// = d1(u, v)
scalar vec2_dist1(vec2 *u, vec2 *v)
{
	return vabs(u->x - v->x) + vabs(u->y - v->y);
}

// r = projection of v wrt n
void vec2_project(vec2 *r, vec2 *n, vec2 *v)
{
	scalar t = -(vec2_dot(n, v) / vec2_dot(n, n));
	vec2_ma(r, v, t, n);
}

// r = reflection of v wrt n
void vec2_reflect(vec2 *r, vec2 *n, vec2 *v)
{
	scalar t = -2 * (vec2_dot(n, v) / vec2_dot(n, n));
	vec2_ma(r, v, t, n);
}


// -------------------
// --- common vec3 ---
// -------------------

// = v · v
scalar vec3_lensqr(vec3 *v)
{
	return v->x * v->x + v->y * v->y + v->z * v->z;
}

// = |v|
scalar vec3_len(vec3 *v)
{
	return vsqrt(v->x * v->x + v->y * v->y + v->z * v->z);
}

// = d1(v, 0)
scalar vec3_len1(vec3 *v)
{
	return vabs(v->x) + vabs(v->y) + vabs(v->z);
}

// v = 0
void vec3_zero(vec3 *v)
{
	v->x = v->y = v->z = 0;
}

// r = {x, y, z}
void vec3_set(vec3 *v, scalar x, scalar y, scalar z)
{
	v->x = x; v->y = y; v->z = z;
}

// v = {...}
void vec3_seta(vec3 *v, scalar *u)
{
	v->x = u[0]; v->y = u[1]; v->z = u[2];
}

// r = v
void vec3_copy(vec3 *r, vec3 *v)
{
	r->x = v->x; r->y = v->y; r->z = v->z;
}

// r = s * v
void vec3_scale(vec3 *r, scalar s, vec3 *v)
{
	r->x = s * v->x; r->y = s * v->y; r->z = s * v->z;
}

// r = -v
void vec3_neg(vec3 *r, vec3 *v)
{
	r->x = -(v->x); r->y = -(v->y); r->z = -(v->z);
}

// r = v / |v|  (possible division by 0)
void vec3_normalize(vec3 *r, vec3 *v)
{
	scalar s = vsqrt(v->x * v->x + v->y * v->y + v->z * v->z);
	r->x = v->x / s; r->y = v->y / s; r->z = v->z / s;
}

// r = u + v
void vec3_add(vec3 *r, vec3 *u, vec3 *v)
{
	r->x = u->x + v->x; r->y = u->y + v->y; r->z = u->z + v->z;
}

// r = u - v
void vec3_sub(vec3 *r, vec3 *u, vec3 *v)
{
	r->x = u->x - v->x; r->y = u->y - v->y; r->z = u->z - v->z;
}

// r = u * v (term-wise)
void vec3_tmul(vec3 *r, vec3 *u, vec3 *v)
{
	r->x = u->x * v->x; r->y = u->y * v->y; r->z = u->z * v->z;
}

// = u · v
scalar vec3_dot(vec3 *u, vec3 *v)
{
	return u->x * v->x + u->y * v->y + u->z * v->z;
}

// r = u + s * v
void vec3_ma(vec3 *r, vec3 *u, scalar s, vec3 *v)
{
	r->x = u->x + s * v->x; r->y = u->y + s * v->y; r->z = u->z + s * v->z;
}

// = |u - v|^2
scalar vec3_distsqr(vec3 *u, vec3 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	t = u->z - v->z; d += t * t;
	return d;
}

// |u - v|
scalar vec3_dist(vec3 *u, vec3 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	t = u->z - v->z; d += t * t;
	return vsqrt(d);
}

// = d1(u, v)
scalar vec3_dist1(vec3 *u, vec3 *v)
{
	return vabs(u->x - v->x) + vabs(u->y - v->y) + vabs(u->z - v->z);
}

// r = projection of v wrt n
void vec3_project(vec3 *r, vec3 *n, vec3 *v)
{
	scalar t = -(vec3_dot(n, v) / vec3_dot(n, n));
	vec3_ma(r, v, t, n);
}

// r = reflection of v wrt n
void vec3_reflect(vec3 *r, vec3 *n, vec3 *v)
{
	scalar t = -2 * (vec3_dot(n, v) / vec3_dot(n, n));
	vec3_ma(r, v, t, n);
}


// -------------------
// --- common vec4 ---
// -------------------

// = v · v
scalar vec4_lensqr(vec4 *v)
{
	return v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w;
}

// = |v|
scalar vec4_len(vec4 *v)
{
	return vsqrt(v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w);
}

// = d1(v, 0)
scalar vec4_len1(vec4 *v)
{
	return vabs(v->x) + vabs(v->y) + vabs(v->z) + vabs(v->w);
}

// v = 0
void vec4_zero(vec4 *v)
{
	v->x = v->y = v->z = v->w = 0;
}

// v = {x, y, z, w}
void vec4_set(vec4 *v, scalar x, scalar y, scalar z, scalar w)
{
	v->x = x; v->y = y; v->z = z; v->w = w;
}

// v = {...}
void vec4_seta(vec4 *v, scalar *u)
{
	v->x = u[0]; v->y = u[1]; v->z = u[2]; v->w = u[3];
}

// r = v
void vec4_copy(vec4 *r, vec4 *v)
{
	r->x = v->x; r->y = v->y;
	r->z = v->z; r->w = v->w;
}

// r = s * v
void vec4_scale(vec4 *r, scalar s, vec4 *v)
{
	r->x = s * v->x; r->y = s * v->y;
	r->z = s * v->z; r->w = s * v->w;
}

// r = -v
void vec4_neg(vec4 *r, vec4 *v)
{
	r->x = -(v->x); r->y = -(v->y);
	r->z = -(v->z); r->w = -(v->w);
}

// r = v / |v| (possible division by 0)
void vec4_normalize(vec4 *r, vec4 *v)
{
	scalar s = vsqrt(v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w);
	r->x = v->x / s; r->y = v->y / s;
	r->z = v->z / s; r->w = v->w / s;
}

// r = u + v
void vec4_add(vec4 *r, vec4 *u, vec4 *v)
{
	r->x = u->x + v->x; r->y = u->y + v->y;
	r->z = u->z + v->z; r->w = u->w + v->w;
}

// r = u - v
void vec4_sub(vec4 *r, vec4 *u, vec4 *v)
{
	r->x = u->x - v->x; r->y = u->y - v->y;
	r->z = u->z - v->z; r->w = u->w - v->w;
}

// r = u * v (term-wise)
void vec4_tmul(vec4 *r, vec4 *u, vec4 *v)
{
	r->x = u->x * v->x; r->y = u->y * v->y;
	r->z = u->z * v->z; r->w = u->w * v->w;
}

// = u · v
scalar vec4_dot(vec4 *u, vec4 *v)
{
	return u->x * v->x + u->y * v->y
	     + u->z * v->z + u->w * v->w;
}

// r = u + s * v
void vec4_ma(vec4 *r, vec4 *u, scalar s, vec4 *v)
{
	r->x = u->x + s * v->x; r->y = u->y + s * v->y;
	r->z = u->z + s * v->z; r->w = u->w + s * v->w;
}

// = |u - v|^2
scalar vec4_distsqr(vec4 *u, vec4 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	t = u->z - v->z; d += t * t;
	t = u->w - v->w; d += t * t;
	return d;
}

// |u - v|
scalar vec4_dist(vec4 *u, vec4 *v)
{
	scalar t, d;
	t = u->x - v->x; d  = t * t;
	t = u->y - v->y; d += t * t;
	t = u->z - v->z; d += t * t;
	t = u->w - v->w; d += t * t;
	return vsqrt(d);
}

// = d1(u, v)
scalar vec4_dist1(vec4 *u, vec4 *v)
{
	return vabs(u->x - v->x) + vabs(u->y - v->y)
	     + vabs(u->z - v->z) + vabs(u->w - v->w);
}

// r = projection of v wrt n
void vec4_project(vec4 *r, vec4 *n, vec4 *v)
{
	scalar t = -(vec4_dot(n, v) / vec4_dot(n, n));
	vec4_ma(r, v, t, n);
}

// r = reflection of v wrt n
void vec4_reflect(vec4 *r, vec4 *n, vec4 *v)
{
	scalar t = -2 * (vec4_dot(n, v) / vec4_dot(n, n));
	vec4_ma(r, v, t, n);
}


// -------------------
// --- vec convert ---
// -------------------

// vec2 = vec3
void vec2_eq_vec3(vec2 *r, vec3 *v)
{
	r->x = v->x; r->y = v->y;
}

// vec2 = vec4
void vec2_eq_vec4(vec2 *r, vec4 *v)
{
	r->x = v->x; r->y = v->y; 
}

// vec3 = vec2
void vec3_eq_vec2(vec3 *r, vec2 *v)
{
	r->x = v->x; r->y = v->y; r->z = 0; 
}

// vec3 = vec4
void vec3_eq_vec4(vec3 *r, vec4 *v)
{
	r->x = v->x; r->y = v->y; r->z = v->z; 
}

// vec4 = vec2
void vec4_eq_vec2(vec4 *r, vec2 *v)
{
	r->x = v->x; r->y = v->y; r->z = r->w = 0; 
}

// vec4 = vec3
void vec4_eq_vec3(vec4 *r, vec3 *v)
{
	r->x = v->x; r->y = v->y; r->z = v->z; r->w = 0; 
}


// -------------------
// ---- vec other ----
// -------------------

// = u x v (scalar since first 2 components are always 0)
scalar vec2_cross(vec2 *u, vec2 *v)
{
	return u->x * v->y - u->y * v->x;
}

// = u x v
void vec3_cross(vec3 *r, vec3 *u, vec3 *v)
{
	scalar x, y, z;
	x = u->y * v->z - u->z * v->y;
	y = u->z * v->x - u->x * v->z;
	z = u->x * v->y - u->y * v->x;
	r->x = x; r->y = y; r->z = z;
}

// p = plane of non collinear points a,b,c
void plane_from_points(plane *p, vec3 *a, vec3 *b, vec3 *c)
{
	vec3 n, u, v;
	vec3_sub(&u, b, a);
	vec3_sub(&v, c, a);
	vec3_cross(&n, &u, &v);
	vec3_normalize(&n, &n);
	p->x = n.x; p->y = n.y; p->z = n.z;
	p->w = vec3_dot(a, &n);
}

// r = projection of v wrt normal of p
void vec3_plane_project(vec3 *r, plane *p, vec3 *v)
{
	vec3 n;
	vec3_eq_vec4(&n, p);
	vec3_project(r, &n, v);
}

// r = reflection of v wrt normal of p
void vec3_plane_reflect(vec3 *r, plane *p, vec3 *v)
{
	vec3 n;
	vec3_eq_vec4(&n, p);
	vec3_reflect(r, &n, v);
}

// r = plane v with unit normal
void plane_normalize(plane *r, plane *v)
{
	scalar s = vsqrt(v->x * v->x + v->y * v->y + v->z * v->z);
	r->x = v->x / s; r->y = v->y / s; r->z = v->z / s;
	r->w = s * v->w;
}


// ---------------------
// --- common mat3x3 ---
// ---------------------

// m = 0
void mat3_zero(mat3 *m)
{
	memset(m, 0, sizeof *m);
}

// m = I
void mat3_id(mat3 *m)
{
	memset(m, 0, sizeof *m);
	m->a.x = m->b.y = m->c.z = 1;
}

// m = [...]
void mat3_seta(mat3 *m, scalar *s)
{
	memcpy(m, s, sizeof *m);
}

// m = [a, b, c]^T
void mat3_setv(mat3 *m, vec3 *a, vec3 *b, vec3 *c)
{
	vec3_copy(&m->a, a);
	vec3_copy(&m->b, b);
	vec3_copy(&m->c, c);
}

// r = m
void mat3_copy(mat3 *r, mat3 *m)
{
	memcpy(r, m, sizeof *m);
}

// r = column(m, ?)
void mat3_colx(vec3 *r, mat3 *m) { vec3_set(r, m->a.x, m->b.x, m->c.x); }
void mat3_coly(vec3 *r, mat3 *m) { vec3_set(r, m->a.y, m->b.y, m->c.y); }
void mat3_colz(vec3 *r, mat3 *m) { vec3_set(r, m->a.z, m->b.z, m->c.z); }
// r = row(m, ?)
void mat3_rowa(vec3 *r, mat3 *m) { vec3_copy(r, &m->a); }
void mat3_rowb(vec3 *r, mat3 *m) { vec3_copy(r, &m->b); }
void mat3_rowc(vec3 *r, mat3 *m) { vec3_copy(r, &m->c); }

// r = s * m
void mat3_scale(mat3 *r, scalar s, mat3 *m)
{
	vec3_scale(&r->a, s, &m->a);
	vec3_scale(&r->b, s, &m->b);
	vec3_scale(&r->c, s, &m->c);
}

// r = -m
void mat3_neg(mat3 *r, mat3 *m)
{
	vec3_neg(&r->a, &m->a);
	vec3_neg(&r->b, &m->b);
	vec3_neg(&r->c, &m->c);
}

// r = f + g
void mat3_add(mat3 *r, mat3 *f, mat3 *g)
{
	vec3_add(&r->a, &f->a, &g->a);
	vec3_add(&r->b, &f->b, &g->b);
	vec3_add(&r->c, &f->c, &g->c);
}

// r = f - g
void mat3_sub(mat3 *r, mat3 *f, mat3 *g)
{
	vec3_sub(&r->a, &f->a, &g->a);
	vec3_sub(&r->b, &f->b, &g->b);
	vec3_sub(&r->c, &f->c, &g->c);
}

// r = f * g (term-wise)
void mat3_tmul(mat3 *r, mat3 *f, mat3 *g)
{
	vec3_tmul(&r->a, &f->a, &g->a);
	vec3_tmul(&r->b, &f->b, &g->b);
	vec3_tmul(&r->c, &f->c, &g->c);
}

// r = v x m (r != v)
void _vec3_mul_mat3(vec3 *r, vec3 *v, mat3 *m)
{
	r->x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z;
	r->y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z;
	r->z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z;
}

// r = f x g
void mat3_mul(mat3 *r, mat3 *f, mat3 *g)
{
	mat3 m;
	_vec3_mul_mat3(&m.a, &f->a, g);
	_vec3_mul_mat3(&m.b, &f->b, g);
	_vec3_mul_mat3(&m.c, &f->c, g);
	memcpy(r, &m, sizeof m);
}

// r = f + s * g
void mat3_ma(mat3 *r, mat3 *f, scalar s, mat3 *g)
{
	vec3_ma(&r->a, &f->a, s, &g->a);
	vec3_ma(&r->b, &f->b, s, &g->b);
	vec3_ma(&r->c, &f->c, s, &g->c);
}

// = det(m)
scalar mat3_det(mat3 *m)
{
	return m->a.x * (m->b.y * m->c.z - m->b.z * m->c.y)
	     - m->a.y * (m->b.x * m->c.z - m->b.z * m->c.x)
	     + m->a.z * (m->b.x * m->c.y - m->b.y * m->c.x);
}

// r = v x m
void vec3_mul_mat3(vec3 *r, vec3 *v, mat3 *m)
{
	scalar x, y, z;
	x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z;
	y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z;
	z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z;
	r->x = x; r->y = y; r->z = z;
}

// r = m x v
void mat3_mul_vec3(vec3 *r, mat3 *m, vec3 *v)
{
	scalar x, y, z;
	x = m->a.x * v->x + m->a.y * v->y + m->a.z * v->z;
	y = m->b.x * v->x + m->b.y * v->y + m->b.z * v->z;
	z = m->c.x * v->x + m->c.y * v->y + m->c.z * v->z;
	r->x = x; r->y = y; r->z = z;
}

// r = transpose(m)
void mat3_trans(mat3 *r, mat3 *m)
{
	scalar t;
	t = m->a.y; r->a.y = m->b.x; r->b.x = t;
	t = m->a.z; r->a.z = m->c.x; r->c.x = t;
	
	t = m->b.z; r->b.z = m->c.y; r->c.y = t;
	
	r->a.x = m->a.x; r->b.y = m->b.y; r->c.z = m->c.z;
}


// ---------------------
// --- common mat3x4 ---
// ---------------------

// m = 0
void mat3x4_zero(mat3x4 *m)
{
	memset(m, 0, sizeof *m);
}

// m = I
void mat3x4_id(mat3x4 *m)
{
	memset(m, 0, sizeof *m);
	m->a.x = m->b.y = m->c.z = 1;
}

// m = [...]
void mat3x4_seta(mat3x4 *m, scalar *s)
{
	memcpy(m, s, sizeof *m);
}

// m = [a, b, c]^T
void mat3x4_setv(mat3x4 *m, vec4 *a, vec4 *b, vec4 *c)
{
	vec4_copy(&m->a, a);
	vec4_copy(&m->b, b);
	vec4_copy(&m->c, c);
}

// r = m
void mat3x4_copy(mat3x4 *r, mat3x4 *m)
{
	memcpy(r, m, sizeof *m);
}

// r = column(m, ?)
void mat3x4_colx(vec3 *r, mat3x4 *m) { vec3_set(r, m->a.x, m->b.x, m->c.x); } //d.x = 0
void mat3x4_coly(vec3 *r, mat3x4 *m) { vec3_set(r, m->a.y, m->b.y, m->c.y); } //d.y = 0
void mat3x4_colz(vec3 *r, mat3x4 *m) { vec3_set(r, m->a.z, m->b.z, m->c.z); } //d.z = 0
void mat3x4_colw(vec3 *r, mat3x4 *m) { vec3_set(r, m->a.w, m->b.w, m->c.w); } //d.w = 1
// r = row(m, ?)
void mat3x4_rowa(vec4 *r, mat3x4 *m) { vec4_copy(r, &m->a); }
void mat3x4_rowb(vec4 *r, mat3x4 *m) { vec4_copy(r, &m->b); }
void mat3x4_rowc(vec4 *r, mat3x4 *m) { vec4_copy(r, &m->c); }

// r = s * m
void mat3x4_scale(mat3x4 *r, scalar s, mat3x4 *m)
{
	vec4_scale(&r->a, s, &m->a);
	vec4_scale(&r->b, s, &m->b);
	vec4_scale(&r->c, s, &m->c);
}

// r = -m
void mat3x4_neg(mat3x4 *r, mat3x4 *m)
{
	vec4_neg(&r->a, &m->a);
	vec4_neg(&r->b, &m->b);
	vec4_neg(&r->c, &m->c);
}

// r = f + g
void mat3x4_add(mat3x4 *r, mat3x4 *f, mat3x4 *g)
{
	vec4_add(&r->a, &f->a, &g->a);
	vec4_add(&r->b, &f->b, &g->b);
	vec4_add(&r->c, &f->c, &g->c);
}

// r = f - g
void mat3x4_sub(mat3x4 *r, mat3x4 *f, mat3x4 *g)
{
	vec4_sub(&r->a, &f->a, &g->a);
	vec4_sub(&r->b, &f->b, &g->b);
	vec4_sub(&r->c, &f->c, &g->c);
}

// r = f * g (term-wise)
void mat3x4_tmul(mat3x4 *r, mat3x4 *f, mat3x4 *g)
{
	vec4_tmul(&r->a, &f->a, &g->a);
	vec4_tmul(&r->b, &f->b, &g->b);
	vec4_tmul(&r->c, &f->c, &g->c);
}

// r = v x m (r != v)
void _vec4_mul_mat3x4(vec4 *r, vec4 *v, mat3x4 *m)
{
	r->x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z;
	r->y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z;
	r->z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z;
	r->w = m->a.w * v->x + m->b.w * v->y + m->c.w * v->z + v->w;
}

// r = f x g
void mat3x4_mul(mat3x4 *r, mat3x4 *f, mat3x4 *g)
{
	mat3x4 m;
	_vec4_mul_mat3x4(&m.a, &f->a, g);
	_vec4_mul_mat3x4(&m.b, &f->b, g);
	_vec4_mul_mat3x4(&m.c, &f->c, g);
	memcpy(r, &m, sizeof m);
}

// r = f + s * g
void mat3x4_ma(mat3x4 *r, mat3x4 *f, scalar s, mat3x4 *g)
{
	vec4_ma(&r->a, &f->a, s, &g->a);
	vec4_ma(&r->b, &f->b, s, &g->b);
	vec4_ma(&r->c, &f->c, s, &g->c);
}

// = det(m)
scalar mat3x4_det(mat3x4 *m)
{
	return m->a.x * (m->b.y * m->c.z - m->b.z * m->c.y)
	     - m->a.y * (m->b.x * m->c.z - m->b.z * m->c.x)
	     + m->a.z * (m->b.x * m->c.y - m->b.y * m->c.x);
}

// r = v x m
void vec4_mul_mat3x4(vec4 *r, vec4 *v, mat3x4 *m)
{
	scalar x, y, z, w;
	x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z;
	y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z;
	z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z;
	w = m->a.w * v->x + m->b.w * v->y + m->c.w * v->z + v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

// r = m x v
void mat3x4_mul_vec4(vec4 *r, mat3x4 *m, vec4 *v)
{
	scalar x, y, z, w;
	x = m->a.x * v->x + m->a.y * v->y + m->a.z * v->z + m->a.w * v->w;
	y = m->b.x * v->x + m->b.y * v->y + m->b.z * v->z + m->b.w * v->w;
	z = m->c.x * v->x + m->c.y * v->y + m->c.z * v->z + m->c.w * v->w;
	w = v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

// r = transpose(m) (inner 3x3 only)
void mat3x4_trans(mat3x4 *r, mat3x4 *m)
{
	scalar t;
	t = m->a.y; r->a.y = m->b.x; r->b.x = t;
	t = m->a.z; r->a.z = m->c.x; r->c.x = t;
	
	t = m->b.z; r->b.z = m->c.y; r->c.y = t;
	
	r->a.x = m->a.x; r->b.y = m->b.y; r->c.z = m->c.z;
	r->a.w = m->a.w; r->b.w = m->b.w; r->c.w = m->c.w;
}


// ---------------------
// --- common mat4x4 ---
// ---------------------

// m = 0
void mat4_zero(mat4 *m)
{
	memset(m, 0, sizeof *m);
}

// m = I
void mat4_id(mat4 *m)
{
	memset(m, 0, sizeof *m);
	m->a.x = m->b.y = m->c.z = m->d.w = 1;
}

// m = [...]
void mat4_seta(mat4 *m, scalar *s)
{
	memcpy(m, s, sizeof *m);
}

// m = [a, b, c]^T
void mat4_setv(mat4 *m, vec4 *a, vec4 *b, vec4 *c, vec4 *d)
{
	vec4_copy(&m->a, a);
	vec4_copy(&m->b, b);
	vec4_copy(&m->c, c);
	vec4_copy(&m->d, d);
}

// r = m
void mat4_copy(mat4 *r, mat4 *m)
{
	memcpy(r, m, sizeof *m);
}

// r = column(m, ?)
void mat4_colx(vec4 *r, mat4 *m) { vec4_set(r, m->a.x, m->b.x, m->c.x, m->d.x); }
void mat4_coly(vec4 *r, mat4 *m) { vec4_set(r, m->a.y, m->b.y, m->c.y, m->d.y); }
void mat4_colz(vec4 *r, mat4 *m) { vec4_set(r, m->a.z, m->b.z, m->c.z, m->d.z); }
void mat4_colw(vec4 *r, mat4 *m) { vec4_set(r, m->a.w, m->b.w, m->c.w, m->d.w); }
// r = row(m, ?)
void mat4_rowa(vec4 *r, mat4 *m) { vec4_copy(r, &m->a); }
void mat4_rowb(vec4 *r, mat4 *m) { vec4_copy(r, &m->b); }
void mat4_rowc(vec4 *r, mat4 *m) { vec4_copy(r, &m->c); }
void mat4_rowd(vec4 *r, mat4 *m) { vec4_copy(r, &m->d); }

// r = s * m
void mat4_scale(mat4 *r, scalar s, mat4 *m)
{
	vec4_scale(&r->a, s, &m->a);
	vec4_scale(&r->b, s, &m->b);
	vec4_scale(&r->c, s, &m->c);
	vec4_scale(&r->d, s, &m->d);
}

// r = -m
void mat4_neg(mat4 *r, mat4 *m)
{
	vec4_neg(&r->a, &m->a);
	vec4_neg(&r->b, &m->b);
	vec4_neg(&r->c, &m->c);
	vec4_neg(&r->d, &m->d);
}

// r = f + g
void mat4_add(mat4 *r, mat4 *f, mat4 *g)
{
	vec4_add(&r->a, &f->a, &g->a);
	vec4_add(&r->b, &f->b, &g->b);
	vec4_add(&r->c, &f->c, &g->c);
	vec4_add(&r->d, &f->d, &g->d);
}

// r = f - g
void mat4_sub(mat4 *r, mat4 *f, mat4 *g)
{
	vec4_sub(&r->a, &f->a, &g->a);
	vec4_sub(&r->b, &f->b, &g->b);
	vec4_sub(&r->c, &f->c, &g->c);
	vec4_sub(&r->d, &f->d, &g->d);
}

// r = f * g (term-wise)
void mat4_tmul(mat4 *r, mat4 *f, mat4 *g)
{
	vec4_tmul(&r->a, &f->a, &g->a);
	vec4_tmul(&r->b, &f->b, &g->b);
	vec4_tmul(&r->c, &f->c, &g->c);
	vec4_tmul(&r->d, &f->d, &g->d);
}

// r = v x m (r != v)
void _vec4_mul_mat4(vec4 *r, vec4 *v, mat4 *m)
{
	r->x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z + m->d.x * v->w;
	r->y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z + m->d.y * v->w;
	r->z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z + m->d.z * v->w;
	r->w = m->a.w * v->x + m->b.w * v->y + m->c.w * v->z + m->d.w * v->w;
}

// r = f x g
void mat4_mul(mat4 *r, mat4 *f, mat4 *g)
{
	mat4 m;
	_vec4_mul_mat4(&m.a, &f->a, g);
	_vec4_mul_mat4(&m.b, &f->b, g);
	_vec4_mul_mat4(&m.c, &f->c, g);
	_vec4_mul_mat4(&m.d, &f->d, g);
	memcpy(r, &m, sizeof m);
}

// r = f + s * g
void mat4_ma(mat4 *r, mat4 *f, scalar s, mat4 *g)
{
	vec4_ma(&r->a, &f->a, s, &g->a);
	vec4_ma(&r->b, &f->b, s, &g->b);
	vec4_ma(&r->c, &f->c, s, &g->c);
}

#define VDET2(a,b, c,d) ((a)*(d) - (b)*(c))
#define VDET3(a,b,c, d,e,f, g,h,i) \
	((a)*VDET2(e,f,h,i) - (b)*VDET2(d,f,g,i) + (c)*VDET2(d,e,g,h))
// = det(m)
scalar mat4_det(mat4 *m)
{
	scalar s = 0;
	s += m->a.x * VDET3(m->b.y, m->b.z, m->b.w, m->c.y, m->c.z, m->c.w, m->d.y, m->d.z, m->d.w);
	s -= m->a.y * VDET3(m->b.x, m->b.z, m->b.w, m->c.x, m->c.z, m->c.w, m->d.x, m->d.z, m->d.w);
	s += m->a.z * VDET3(m->b.x, m->b.y, m->b.w, m->c.x, m->c.y, m->c.w, m->d.x, m->d.y, m->d.w);
	s -= m->a.w * VDET3(m->b.x, m->b.y, m->b.z, m->c.x, m->c.y, m->c.z, m->d.x, m->d.y, m->d.z);
	return s;
}

scalar mat4_det3(mat4 *m)
{
	return m->a.x * (m->b.y * m->c.z - m->b.z * m->c.y)
	     - m->a.y * (m->b.x * m->c.z - m->b.z * m->c.x)
	     + m->a.z * (m->b.x * m->c.y - m->b.y * m->c.x);
}

// r = v x m
void vec4_mul_mat4(vec4 *r, vec4 *v, mat4 *m)
{
	scalar x, y, z, w;
	x = m->a.x * v->x + m->b.x * v->y + m->c.x * v->z + m->d.x * v->w;
	y = m->a.y * v->x + m->b.y * v->y + m->c.y * v->z + m->d.y * v->w;
	z = m->a.z * v->x + m->b.z * v->y + m->c.z * v->z + m->d.z * v->w;
	w = m->a.w * v->x + m->b.w * v->y + m->c.w * v->z + m->d.w * v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

// r = m x v
void mat4_mul_vec4(vec4 *r, mat4 *m, vec4 *v)
{
	scalar x, y, z, w;
	x = m->a.x * v->x + m->a.y * v->y + m->a.z * v->z + m->a.w * v->w;
	y = m->b.x * v->x + m->b.y * v->y + m->b.z * v->z + m->b.w * v->w;
	z = m->c.x * v->x + m->c.y * v->y + m->c.z * v->z + m->c.w * v->w;
	w = m->d.x * v->x + m->d.y * v->y + m->d.z * v->z + m->d.w * v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

// r = transpose(m)
void mat4_trans(mat4 *r, mat4 *m)
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


// -------------------
// ---- mat other ----
// -------------------

void mat3_eq_mat4(mat3 *r, mat4 *m)
{
	vec3_eq_vec4(&r->a, &m->a);
	vec3_eq_vec4(&r->b, &m->b);
	vec3_eq_vec4(&r->c, &m->c);
}

void mat3_eq_mat3x4(mat3 *r, mat3x4 *m)
{
	vec3_eq_vec4(&r->a, &m->a);
	vec3_eq_vec4(&r->b, &m->b);
	vec3_eq_vec4(&r->c, &m->c);
}


void mat3x4_eq_mat3(mat3x4 *r, mat3 *m)
{
	vec4_eq_vec3(&r->a, &m->a);
	vec4_eq_vec3(&r->b, &m->b);
	vec4_eq_vec3(&r->c, &m->c);
}

void mat3x4_eq_mat4(mat3x4 *r, mat4 *m)
{
	vec4_copy(&r->a, &m->a);
	vec4_copy(&r->b, &m->b);
	vec4_copy(&r->c, &m->c);
}

void mat4_eq_mat3(mat4 *r, mat3 *m)
{
	vec4_eq_vec3(&r->a, &m->a);
	vec4_eq_vec3(&r->b, &m->b);
	vec4_eq_vec3(&r->c, &m->c);
	vec4_set(&r->d, 0, 0, 0, 1);
}

void mat4_eq_mat3x4(mat4 *r, mat3x4 *m)
{
	vec4_copy(&r->a, &m->a);
	vec4_copy(&r->b, &m->b);
	vec4_copy(&r->c, &m->c);
	vec4_set(&r->d, 0, 0, 0, 1);
}



// -------------------
// ----    quat   ----
// -------------------

scalar quat_norm(quat *v)
{
	return v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w;
}

// r = 1
void quat_id(quat *v)
{
	v->x = v->y = v->z = 0; v->w = 1;
}

// r = v^-1 (possible divby0)
void quat_inv(quat *r, quat *v)
{
	scalar s = vsqrt(v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w);
	r->x = -(v->x / s); r->y = -(v->y / s);
	r->z = -(v->z / s); r->w =  (v->w / s);
}

// r = v'
void quat_conj(quat *r, quat *v)
{
	r->x = -v->x; r->y = -v->y;
	r->z = -v->z; r->w =  v->w;
}

void quat_eq_vec3(quat *r, vec3 *v)
{
	r->x = v->x; r->y = v->y;
	r->z = v->z; r->w = 0;
}

void vec3_eq_quat(vec3 *r, quat *v)
{
	r->x = v->x; r->y = v->y; r->z = v->z;
}


// = u x v
void quat_mul(quat *r, quat *u, quat *v)
{
	scalar x, y, z, w;
	w = u->w * v->w - u->x * v->x - u->y * v->y - u->z * v->z;
	x = u->w * v->x + u->x * v->w + u->y * v->z - u->z * v->y;
	y = u->w * v->y - u->x * v->z + u->y * v->w + u->z * v->x;
	z = u->w * v->z + u->x * v->y - u->y * v->x + u->z * v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

// = u x v
void quat_mul_vec3(quat *r, quat *u, vec3 *v)
{
	scalar x, y, z, w;
	w = - u->x * v->x - u->y * v->y - u->z * v->z;
	x = u->w * v->x + u->y * v->z - u->z * v->y;
	y = u->w * v->y - u->x * v->z + u->z * v->x;
	z = u->w * v->z + u->x * v->y - u->y * v->x;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

// = u x v
void vec3_mul_quat(quat *r, vec3 *u, quat *v)
{
	scalar x, y, z, w;
	w = - u->x * v->x - u->y * v->y - u->z * v->z;
	x = + u->x * v->w + u->y * v->z - u->z * v->y;
	y = - u->x * v->z + u->y * v->w + u->z * v->x;
	z = + u->x * v->y - u->y * v->x + u->z * v->w;
	r->x = x; r->y = y; r->z = z; r->w = w;
}

// r = qvq'
void vec3_rotate_quat(vec3 *r, quat *q, vec3 *v)
{
	quat p;
	quat_inv(&p, q);
	vec3_mul_quat(&p, v, &p);
	quat_mul(&p, q, &p);
	r->x = p.x; r->y = p.y; r->z = p.z;
}








void mat4_from_quat(mat4 *m, quat *q)
{
	scalar i = q->x, j = q->y, k = q->z, r = q->w;
	
	m->a.x = 1 - 2 * (j * j + k * k);
	m->a.y = 2 * (i * j - k * r);
	m->a.z = 2 * (i * k + j * r);
	
	m->b.x = 2 * (j * i + k * r);
	m->b.y = 1 - 2 * (i * i + k * k);
	m->b.z = 2 * (j * k - i * r);
	
	m->c.x = 2 * (k * i - j * r);
	m->c.y = 2 * (k * j + i * r);
	m->c.z = 1 - 2 * (i * i + j * j);
	
	m->a.w = m->b.w = m->c.w = m->d.x = m->d.y = m->d.z = 0;
	m->d.w = 1;
}







#ifdef VEC_TEST
int main(void) {return 0;}
#endif
