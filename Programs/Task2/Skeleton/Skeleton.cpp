//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Piller Trisztan
// Neptun : WHKJZX
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char* vertexSource = R"(
	#version 330
	precision highp float;

	uniform vec3 wLookAt, wRight, wUp;

	layout(location = 0) in vec2 cCamWindowVertex;
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

const char* fragmentSource = R"(
	#version 330
	precision highp float;

	struct Material {
		vec3 ka, kd, ks, F0;
		float shininess;
		int rough, reflective;
	};

	struct Pentagon {
		vec3 a, b, c, d, e;
	};

	struct Sphere {
		vec3 center;
		float radius;
	};

	struct Implicit {
		float a, b, c;
		mat4 Q;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;
		vec3 nearestPentagonCenter;
		int hitPentagon;
	};

	struct Ray {
		vec3 start, dir;
	};

	struct Light {
		vec3 La, location, power;
	};

	uniform vec3 wEye;
	uniform Light light;
	uniform Material materials[3]; // 0: rough, 1: fully ref, 2: golden
	uniform Pentagon pentagons[12];
	uniform Sphere sphere;
	uniform Implicit implicit;

	in vec3 p;
	out vec4 fragmentColor;

	const float epsilon = 0.0001;
	const int maxdepth = 5;
	const float M_PI = 3.1415926538;
	const float rotation = 72. / 360. * 2 * M_PI;
	const float sin54plusone = sin(54. / 360. * 2 * M_PI) + 1;
	const float threshold = 0.9;

	float distanceOf(vec3 point) {
		return length(light.location - point);
	}

	vec3 directionOf(vec3 point) {
		return normalize(light.location - point);
	}

	vec3 radianceAt(vec3 point) {
		float distance2 = dot(light.location - point, light.location - point);
		if (distance2 < epsilon) 
			distance2 = epsilon;
		return light.power / distance2 / 4. / M_PI;
	}

	vec3 Rodrigues(vec3 r, vec3 d, float phi) {
		return r * cos(phi) +  d * dot(r, d) * (1 - cos(phi)) + cross(d, r) * sin(phi);
	}

	Hit intersectSphere(const Sphere sphere, const Ray ray) {
		Hit hit;
		hit.t = -1;

		vec3 dist = ray.start - sphere.center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2;
		float c = dot(dist, dist) - sphere.radius * sphere.radius;
		float discr = b * b - 4 * a * c;

		if (discr < 0) return hit;

		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2. / a; // t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2. / a;

		if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - sphere.center) / sphere.radius;
        return hit;
	}

    vec3 normalOfImplicit(Implicit imp, vec3 p) {
		vec4 x = vec4(p, 1);
		vec4 g = x * imp.Q * 2;
        return normalize(vec3(g.x, g.y, g.z));
    }

	Hit intersectImplicit(const Implicit implicit, const Ray ray) {
		Hit hit;
		hit.t = -1;

        vec4 start = vec4(ray.start, 1); // point
        vec4 dir = vec4(ray.dir, 0); // vector

        float a = dot(dir * implicit.Q, dir);
        float b = dot(start * implicit.Q, dir) + dot(dir * implicit.Q, start);
        float c = dot(start * implicit.Q, start);
        float discr = b * b - 4 * a * c;

        if (discr < 0) return hit;

        float sqrt_discr = sqrt(discr);
        float t1 = (-b + sqrt_discr) / 2. / a; // t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2. / a;

		if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalOfImplicit(implicit, hit.position);
        return hit;
	}

	Hit intersect(const Sphere sphere, const Implicit implicit, const Ray ray) {
		Hit hit, sHit, iHit;
		hit.t = -1;
		sHit = intersectSphere(sphere, ray);
		iHit = intersectImplicit(implicit, ray);

		if (sHit.t <= 0)
			return hit;
		
		if (iHit.t <= 0)
			return hit;

		if (dot(vec4(ray.start, 1) * implicit.Q, vec4(ray.start, 1)) < 0)
			hit = sHit;
		else if (iHit.t < sHit.t)
			hit = sHit;
		else
			hit = iHit;

		hit.hitPentagon = 0;
		hit.mat = 2;
		return hit;
	}

	void intersectTriangle(const vec3 r1, const vec3 r2, const vec3 r3, const Ray ray, inout Hit hit) {
		if (hit.t != -1)
			return;

		vec3 n = cross((r2 - r1), (r3 - r1));
		float t = dot((r1 - ray.start), n) / dot(ray.dir, n);
		vec3 p = ray.start + t * ray.dir;

		if (t > 0) {
			float lim1 = dot(cross((r2 - r1), (p - r1)), n);
			float lim2 = dot(cross((r3 - r2), (p - r2)), n);
			float lim3 = dot(cross((r1 - r3), (p - r3)), n);
			
			if (lim1 > 0 && lim2 > 0 && lim3 > 0) {
				vec3 r2in = r1 + (r2 - r1) * threshold;
				vec3 r3in = r1 + (r3 - r1) * threshold;
				float lim = dot(cross((r3in - r2in), (p - r2in)), n);

				if (lim > 0) // inside
					hit.mat = 1; // fully ref
				else
					hit.mat = 0; // rough

				hit.t = t;
				hit.normal = normalize(n);
				hit.position = p;
			}
		}
	}

	Hit intersect(const Pentagon pentagon, const Ray ray) {
		Hit hit;
		hit.t = -1;
		vec3 center = pentagon.a + ((pentagon.c + pentagon.d) / 2 - pentagon.a) / sin54plusone;
		intersectTriangle(center, pentagon.a, pentagon.b, ray, hit);
		intersectTriangle(center, pentagon.b, pentagon.c, ray, hit);
		intersectTriangle(center, pentagon.c, pentagon.d, ray, hit);
		intersectTriangle(center, pentagon.d, pentagon.e, ray, hit);
		intersectTriangle(center, pentagon.e, pentagon.a, ray, hit);
		hit.nearestPentagonCenter = center;
		hit.hitPentagon = 1;
		return hit;
	}

	void goThroughPortal(Hit hit, inout Ray ray) {
		vec3 hitPoint = hit.position - hit.nearestPentagonCenter;
		vec3 n = hit.normal;
		vec3 extra = hitPoint + ray.dir;

		hitPoint = Rodrigues(hitPoint, n, rotation);
		extra = Rodrigues(extra, n, rotation);

		hit.position = hitPoint + hit.nearestPentagonCenter;
		ray.dir = extra - hitPoint;

		ray.start = hit.position + hit.normal * epsilon;
		ray.dir = reflect(ray.dir, hit.normal);
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;

		Hit hit = intersect(sphere, implicit, ray);
		if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
			bestHit = hit;

		for (int o = 0; o < 12; o++) {
			Hit hit = intersect(pentagons[o], ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
				bestHit = hit;
		}

		if (dot(ray.dir, bestHit.normal) > 0) 
			bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = light.La;

		for (int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			
			if (hit.t < 0)
				return outRadiance;

			if (materials[hit.mat].rough == 1) {
				outRadiance += weight * materials[hit.mat].ka * light.La;

				vec3 lightDir = directionOf(hit.position);
				vec3 lightRad = radianceAt(hit.position);
				float lightDist = distanceOf(hit.position);

				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = directionOf(hit.position);

				float cosTheta = dot(hit.normal, lightDir);
				Hit shadowHit = firstIntersect(Ray(hit.position + hit.normal * epsilon, lightDir));

				if (cosTheta > 0 && (shadowHit.t < epsilon || shadowHit.t > lightDist)) {
					outRadiance += weight * lightRad * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + lightDir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) 
						outRadiance += weight * lightRad * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}

			if (materials[hit.mat].reflective == 1) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				if (hit.hitPentagon == 1)
					goThroughPortal(hit, ray);
				else {
					ray.start = hit.position + hit.normal * epsilon;
					ray.dir = reflect(ray.dir, hit.normal);
				}
			}
			else 
				return outRadiance;
		}
		return outRadiance;
	}

	void main() {
		Ray ray;
		ray.start = wEye;
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1);
	}
)";

vec3 Rodrigues(vec3 r, vec3 d, float phi) {
	return r * cosf(phi) + d * dot(r, d) * (1 - cosf(phi)) + cross(d, r) * sinf(phi);
}

vec3 operator/(vec3 a, vec3 b) {
	return { a.x / b.x, a.y / b.y, a.z / b.z };
}

struct Material {
	vec3 ka, kd, ks, F0;
	float shininess;
	int rough, reflective;
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

struct SmoothMaterial : Material {
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

struct Pentagon {
	vec3 a, b, c, d, e;

	Pentagon(const vec3& _a, const vec3& _b, const vec3& _c, const vec3& _d, const vec3& _e)
		:a(_a), b(_b), c(_c), d(_d), e(_e) { }
};

struct Dodecahedron {
	const vec3 vertices[20] = {
		vec3(0.0f, 0.618f, 1.618f),
		vec3(0.0f, -0.618f, 1.618f),
		vec3(0.0f, -0.618f, -1.618f),
		vec3(0.0f, 0.618f, -1.618f),
		vec3(1.618f, 0.0f, 0.618f),
		vec3(-1.618f, 0.0f, 0.618f),
		vec3(-1.618f, 0.0f, -0.618f),
		vec3(1.618f, 0.0f, -0.618f),
		vec3(0.618f, 1.618f, 0.0f),
		vec3(-0.618f, 1.618f, 0.0f),
		vec3(-0.618f, -1.618f, 0.0f),
		vec3(0.618f, -1.618f, 0.0f),
		vec3(1.0f, 1.0f, 1.0f),
		vec3(-1.0f, 1.0f, 1.0f),
		vec3(-1.0f, -1.0f, 1.0f),
		vec3(1.0f, -1.0f, 1.0f),
		vec3(1.0f, -1.0f, -1.0f),
		vec3(1.0f, 1.0f, -1.0f),
		vec3(-1.0f, 1.0f, -1.0f),
		vec3(-1.0f, -1.0f, -1.0f)
	};

	const int faces[5 * 12] = {
		1, 2, 16, 5, 13,
		1, 13, 9, 10, 14,
		1, 14, 6, 15, 2,
		2, 15, 11, 12, 16,
		3, 4, 18, 8, 17,
		3, 17, 12, 11, 20,
		3, 20, 7, 19, 4,
		19, 10, 9, 18, 4,
		16, 12, 17, 8, 5,
		5, 8, 18, 9, 13,
		14, 10, 19, 7, 6,
		6, 7, 20, 11, 15
	};

	std::vector<Pentagon*> sides;

	Dodecahedron() {
		for (int i = 0; i < 12; ++i) {
			vec3 points[5];
			for (int j = 0; j < 5; ++j) {
				points[j] = vertices[faces[i * 5 + j] - 1];
			}
			sides.push_back(new Pentagon(points[0], points[1], points[2], points[3], points[4]));
		}
	}

	~Dodecahedron() {
		for (int i = 0; i < 12; ++i) {
			delete sides[i];
		}
	}
};

struct Sphere {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius)
		:center(_center), radius(_radius) { }
};

struct Implicit {
	float a, b, c;
	mat4 Q;

	Implicit(float _a, float _b, float _c):a(_a), b(_b), c(_c) {
		float d = -c / 2.0f;
		Q = mat4(
			vec4(a, 0, 0, 0),
			vec4(0, b, 0, 0),
			vec4(0, 0, 0, d),
			vec4(0, 0, d, 0)
		);
	}
};

struct Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}

	void Animate(float dt) {
		eye = Rodrigues(eye, vec3(0, 0, 1), dt);
		set(eye, lookat, vec3(0, 0, 1), fov);
	}
};

struct Light {
	vec3 La, location, power;

	Light(const vec3& _location, const vec3& _power, const vec3& _La)
		:location(_location), power(_power), La(_La) { }
};

class Shader : public GPUProgram {
public:
	void setUniformMaterials(const std::vector<Material*>& materials) {
		char name[256];
		for (size_t i = 0; i < materials.size(); i++) {
			sprintf(name, "materials[%d].ka", i); setUniform(materials[i]->ka, name);
			sprintf(name, "materials[%d].kd", i); setUniform(materials[i]->kd, name);
			sprintf(name, "materials[%d].ks", i); setUniform(materials[i]->ks, name);
			sprintf(name, "materials[%d].shininess", i); setUniform(materials[i]->shininess, name);
			sprintf(name, "materials[%d].F0", i); setUniform(materials[i]->F0, name);
			sprintf(name, "materials[%d].rough", i); setUniform(materials[i]->rough, name);
			sprintf(name, "materials[%d].reflective", i); setUniform(materials[i]->reflective, name);
		}
	}

	void setUniformLight(Light* light) {
		setUniform(light->La, "light.La");
		setUniform(light->power, "light.power");
		setUniform(light->location, "light.location");
	}

	void setUniformCamera(Camera* camera) {
		setUniform(camera->eye, "wEye");
		setUniform(camera->lookat, "wLookAt");
		setUniform(camera->right, "wRight");
		setUniform(camera->up, "wUp");
	}

	void setUniformSphere(Sphere* sphere) {
		setUniform(sphere->center, "sphere.center");
		setUniform(sphere->radius, "sphere.radius");
	}

	void setUniformImplicit(Implicit* implicit) {
		setUniform(implicit->a, "implicit.a");
		setUniform(implicit->b, "implicit.b");
		setUniform(implicit->c, "implicit.c");
		setUniform(implicit->Q, "implicit.Q");
	}

	void setUniformPentagons(const std::vector<Pentagon*>& pentagons) {
		char name[256];
		for (size_t o = 0; o < pentagons.size(); o++) {
			sprintf(name, "pentagons[%d].a", o);  setUniform(pentagons[o]->a, name);
			sprintf(name, "pentagons[%d].b", o);  setUniform(pentagons[o]->b, name);
			sprintf(name, "pentagons[%d].c", o);  setUniform(pentagons[o]->c, name);
			sprintf(name, "pentagons[%d].d", o);  setUniform(pentagons[o]->d, name);
			sprintf(name, "pentagons[%d].e", o);  setUniform(pentagons[o]->e, name);
		}
	}
};

class Scene {
	Sphere* sphere;
	Implicit* implicit;
	Dodecahedron* dodecahedron;
	Light* light;
	std::vector<Material*> materials;
public:
	Camera* camera;

	Scene() {
		vec3 kd, ks, n, kappa, F0, rgb, one(1, 1, 1);

		camera = new Camera();
		vec3 eye = vec3(0, 1.2f, 0.4f);
		vec3 vup = vec3(0, 0, 1);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * (float)M_PI / 180;
		camera->set(eye, lookat, vup, fov);

		rgb = { 30, 30, 30 };
		light = new Light(vec3(0.7f, 0, 0.5f), vec3(30, 30, 30), vec3(rgb.x / 255, rgb.y / 255, rgb.z / 255));

		rgb = { 160, 100, 100 };
		kd = { rgb.x / 255, rgb.y / 255, rgb.z / 255 };
		ks = { 160, 60, 60 };
		materials.push_back(new RoughMaterial(kd, ks, 70));
		
		F0 = { 1, 1, 1 };
		materials.push_back(new SmoothMaterial(F0));
		dodecahedron = new Dodecahedron();

		n = { 0.17f, 0.35f, 1.5f };
		kappa = { 3.1f, 2.7f, 1.9f };
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
		materials.push_back(new SmoothMaterial(F0));

		sphere = new Sphere(vec3(0.0f, 0.0f, 0.0f), 0.3f);
		implicit = new Implicit(2.1f, 1.5f, 0.5f);
	}

	void setUniform(Shader* shader) {
		shader->setUniformSphere(sphere);
		shader->setUniformImplicit(implicit);
		shader->setUniformPentagons(dodecahedron->sides);
		shader->setUniformMaterials(materials);
		shader->setUniformLight(light);
		shader->setUniformCamera(camera);
	}

	~Scene() {
		for (size_t i = 0; i < materials.size(); ++i) {
			delete materials[i];
		}
		delete implicit;
		delete sphere;
		delete light;
		delete dodecahedron;
		delete camera;
	}
};

class FullScreenTexturedQuad {
	unsigned int vao = 0;

public:
	FullScreenTexturedQuad() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

Shader* shader;
Scene* scene;
FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene = new Scene();
	fullScreenTexturedQuad = new FullScreenTexturedQuad();
	shader = new Shader();
	shader->create(vertexSource, fragmentSource, "fragmentColor");
	shader->Use();
	scene->setUniform(shader);
}

void onDisplay() {
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	shader->setUniformCamera(scene->camera);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {}
void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}

void onIdle() {
	scene->camera->Animate(0.005f);
	glutPostRedisplay();
}
