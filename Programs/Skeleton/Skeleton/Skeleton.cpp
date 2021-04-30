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

template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) { return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }
typedef Dnum<vec2> Dnum2;

const int TESSELATION_LEVEL = 100;
const vec3 GRAVITY_VEC = { 0, 0, -9.81f };
const float R_NULL = 0.005f;
const float AMPLIFIER = 15.0f;
const float UPPER_BOUND = 10.0f;
const float FOV_DEGREES = 45.0f;
const float SPHERE_RADIUS = 0.5f;
const float STARTING_MASS = 0.05f;
const float SPHERE_MASS = 1.0f;
const float MAX_DEPTH = -12.0f;

struct Weight {
	Dnum2 X, Y;
	float mass;

	Weight(float _mass, Dnum2 _X, Dnum2 _Y) : mass(_mass), X(_X), Y(_Y) { }
};
std::vector<Weight> weights;

float rnd() { return (float)rand() / RAND_MAX; }
bool operator==(vec3 a, vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
vec3 operator/(vec3 a, vec3 b) { return { a.x / b.x, a.y / b.y, a.z / b.z }; }

Dnum2 h(Dnum2 X, Dnum2 Y) {
	Dnum2 sum;
	for (auto& weight : weights) {
		Dnum2 sqrtRes = Pow(Pow(X - weight.X, 2) + Pow(Y - weight.Y, 2), 0.5f);
		sum = sum + Dnum2(weight.mass) / (sqrtRes + Dnum2(R_NULL));
	}
	return sum * (-1);
}

vec4 qmul(vec4 q1, vec4 q2) {
	vec3 d1 = { q1.x, q1.y, q1.z };
	vec3 d2 = { q2.x, q2.y, q2.z };
	vec3 res = d2 * q1.w + d1 * q2.w + cross(d1, d2);
	return { res.x, res.y, res.z, q1.w * q2.w - dot(d1, d2) };
}

vec4 qinv(vec4 q) {
	return vec4(-q.x, -q.y, -q.z, q.w) / dot(q, q);
}

vec4 quat(float t) {
	return { sinf(t / 4) * cosf(t) / 2, sinf(t / 4) * sinf(t) / 2, sinf(t / 4) * sqrtf(3 / 4), cosf(t / 4) };
}

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = FOV_DEGREES * (float)M_PI / 180.0f;
		fp = 1;
		bp = UPPER_BOUND * 10;
	} 
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			                                       u.y, v.y, w.y, 0,
			                                       u.z, v.z, w.z, 0,
			                                       0,   0,   0,   1);
	}

	mat4 P() {
		return mat4(1 / (tanf(fov / 2)*asp), 0,                 0,                       0,
			        0,                       1 / tanf(fov / 2), 0,                       0,
			        0,                       0,                 -(fp + bp) / (bp - fp), -1,
			        0,                       0,                 -2 * fp*bp / (bp - fp),  0);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shine;

	Material(vec3 _kd, vec3 _ks, vec3 _ka, float s): kd(_kd), ks(_ks), ka(_ka), shine(s) { }
};

class Light {
public:
	vec3 La, Le;
	vec4 pos, posCurr;
	vec4 posPair;

	Light(vec3 _La, vec3 _Le, vec4 pos1, vec4 pos2): La(_La), Le(_Le), pos(pos1), posPair(pos2) { }

	void Animate(float tstart, float tend) {
		// move pair to origo
		vec4 origo = vec4(0, 0, 0, 1);
		vec4 shift = origo - posPair;
		vec4 posShifted = pos + shift;

		// turn
		vec4 q = quat(tend);
		vec4 res = qmul(qmul(q, vec4(posShifted.x, posShifted.y, posShifted.z, 0)), qinv(q));
		posShifted = vec4(res.x, res.y, res.z, pos.w);

		// get back
		posCurr = posShifted - shift;
	}
};

struct RenderState {
	mat4 MVP, M, Minv, V, P;
	Material* material = nullptr;
	std::vector<Light*> lights;
	vec3 wEye;
};

class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shine, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.posCurr, name + ".wLightPos");
	}
};

class PhongShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4 MVP, M, Minv;
		uniform Light[2] lights;
		uniform int nLights;
		uniform vec3 wEye;

		layout(location = 0) in vec3 vtxPos;
		layout(location = 1) in vec3 vtxNorm;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[2];
		out vec3 vtxPosi;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
			vtxPosi = vtxPos;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[2] lights;
		uniform int nLights;

		in vec3 wNormal;       // interpolated world sp normal
		in vec3 wView;         // interpolated world sp view
		in vec3 wLight[2];     // interpolated world sp illum dir
		in vec3 vtxPosi;
		
        out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			float darken = (vtxPosi.z < 0) ? pow(0.5, floor(-vtxPosi.z)) : 1;
			vec3 ka = material.ka * darken;
			vec3 kd = material.kd * darken;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");
		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(*state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

class Geometry {
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
	struct VertexData {
		vec3 position, normal;
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = normalize(cross(drdU, drdV));
		return vtxData;
	}

	void create(int N = TESSELATION_LEVEL, int M = TESSELATION_LEVEL) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

class Sphere : public ParamSurface {
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};

class Sheet : public ParamSurface {
public:
	Sheet() { create(); }

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		X = U;
		Y = V;
		Z = h(U, V);
	}
};

Sphere* sphere;
Sheet* sheet;
PhongShader* shader;

struct Object {
	Material* material;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Material* _material, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { }
};

struct SheetObject : Object {
	SheetObject(Material* _material) : Object(_material, sheet) {
		scale = { UPPER_BOUND * 2, UPPER_BOUND * 2, 1 };
		translation = { -UPPER_BOUND, -UPPER_BOUND, 0 };
	}
};

struct SphereObject : Object {
	vec3 v, normal;
	float E;

	SphereObject(Material* _material) : Object(_material, sphere), E(0), v({ 0, 0, 0 }), normal({ 0, 0, 1 }) { }

	float CalcNormal() {
		// get normal on sheet
		vec3 p = (translation / UPPER_BOUND + vec3(1, 1, 1)) / 2.0f;
		Dnum2 X = { p.x, vec2(1, 0) };
		Dnum2 Y = { p.y, vec2(0, 1) };
		Dnum2 Z = h(X, Y);
		normal = normalize({ -Z.d.x, -Z.d.y, 1 });
		return Z.f;
	}

	float CalcEnergy() {
		return SPHERE_MASS * length(GRAVITY_VEC) * fabsf(MAX_DEPTH - translation.z) + 0.5f * SPHERE_MASS * dot(v, v);
	}

	void SetStartVec(vec3 _v) {
		v = _v * AMPLIFIER;
		E = CalcEnergy();
	}

	void Animate(float tstart, float tend) {
		if (translation.z <= MAX_DEPTH)
			return;

		float dt = tend - tstart;

		// Euler's
		CalcNormal();
		vec3 a = GRAVITY_VEC - dot(GRAVITY_VEC, normal) * normal;
		v = v + a * dt;
		translation = translation + v * dt;

		// flat torus topology
		if (translation.x >= UPPER_BOUND)
			translation.x = translation.x - UPPER_BOUND * 2;

		if (translation.y >= UPPER_BOUND)
			translation.y = translation.y - UPPER_BOUND * 2;

		// bring back to sheet	
		float z = CalcNormal();
		translation = vec3(translation.x, translation.y, z); //+ normal * SPHERE_MASS;

		// energy correction
		float expectedVelocity = sqrtf(2 * (E - length(GRAVITY_VEC) * fabsf(MAX_DEPTH - translation.z)));
		float rate = expectedVelocity / length(v);
		v = v * rate;
		printf("E: %g\tv: %g (%g %g %g)\ta: %g (%g %g %g)\r", CalcEnergy(), length(v), v.x, v.y, v.z, length(a), a.x, a.y, a.z);
	}
};

class Scene {	
	std::vector<SphereObject*> balls;
	SheetObject* sheetObject = nullptr;
	Camera camera;
	std::vector<Light*> lights;
public:
	SphereObject* BuildOneSphere() {
		Material* mat = new Material({ rnd(), rnd(), rnd() }, { 1, 1, 1 }, { 0.1f, 0.1f, 0.1f }, 30);
		SphereObject* sphereObject = new SphereObject(mat);
		sphereObject->scale = { SPHERE_RADIUS, SPHERE_RADIUS, SPHERE_RADIUS };
		sphereObject->translation = { -UPPER_BOUND + 1.5f * SPHERE_RADIUS, -UPPER_BOUND + 1.5f * SPHERE_RADIUS, SPHERE_RADIUS };
		balls.push_back(sphereObject);
		return sphereObject;
	}

	void StartSphere(SphereObject* sphereObject, vec3 startVec) {
		sphereObject->SetStartVec(startVec);
	}

	void Build() {
		Material* mat = new Material({ 0.2f, 0.5f, 0.8f }, { 0.6f, 0.6f, 0.6f }, { 0.1f, 0.1f, 0.1f }, 30);
		sheetObject = new SheetObject(mat);
		
		float z = UPPER_BOUND / tanf(M_PI * 22.5f / 180.0f);
		camera.wEye = { 0, 0, z };
		camera.wLookat = { 0, 0, 0 };
		camera.wVup = { 0, 1, 0 };

		vec4 pos1 = { UPPER_BOUND * 1.5f, -UPPER_BOUND * 1.5f, z, 1 };
		vec4 pos2 = { -UPPER_BOUND * 1.5f, UPPER_BOUND * 1.5f, z, 1 };
		lights.push_back(new Light({ 0, 0, 0 }, { 1, 1, 1 }, pos1, pos2));
		lights.push_back(new Light({ 0, 0, 0 }, { 1, 1, 1 }, pos2, pos1));
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;

		for (auto& obj : balls) {
			obj->Draw(state);
		}
		sheetObject->Draw(state);
	}

	void Render(SphereObject* sphereObject) {
		RenderState state;
		Camera camera;
		camera.wEye = sphereObject->translation + vec3(0, 0, SPHERE_RADIUS);
		camera.wLookat = sphereObject->v == vec3(0, 0, 0)?
			vec3(10, 10, SPHERE_RADIUS) : 
			sphereObject->translation + normalize(sphereObject->v) + vec3(0, 0, SPHERE_RADIUS);
		camera.wVup = sphereObject->normal;

		// printf("wEye: (%g %g %g)\twLookat: (%g %g %g)\twVup: (%g %g %g)\r", camera.wEye.x, camera.wEye.y, camera.wEye.z, camera.wLookat.x, camera.wLookat.y, camera.wLookat.z, camera.wVup.x, camera.wVup.y, camera.wVup.z);

		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;

		for (auto& obj : balls) {
			if (obj != sphereObject)
				obj->Draw(state);
		}
		sheetObject->Draw(state);
	}

	void Animate(float tstart, float tend) {
		for (auto& obj : balls) {
			if (!(obj->v == vec3(0, 0, 0)))
				obj->Animate(tstart, tend);
		}

		for (auto& light : lights) {
			light->Animate(tstart, tend);
		}
	}
};

bool isCameraAttached = false;
Scene scene;
SphereObject* nextSphere;
SphereObject* renderedSphere;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	shader = new PhongShader();
	sphere = new Sphere();
	sheet = new Sheet();
	scene.Build();
	nextSphere = scene.BuildOneSphere();
}

void onDisplay() {
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (isCameraAttached)
		scene.Render(renderedSphere);
	else
		scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		renderedSphere = nextSphere;
		isCameraAttached = !isCameraAttached;
		glutPostRedisplay();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

float mass = STARTING_MASS;

void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1.0f;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	float x = (cX + 1) / 2.0f;
	float y = (cY + 1) / 2.0f;

	if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
		scene.StartSphere(nextSphere, { x, y, 0 });
		nextSphere = scene.BuildOneSphere();
	}

	if (state == GLUT_DOWN && button == GLUT_RIGHT_BUTTON) {
		weights.push_back(Weight(mass, Dnum2(x), Dnum2(y)));
		mass *= 1.25f;
		sheet->create();
		glutPostRedisplay();
	}
}

void onMouseMotion(int pX, int pY) { }

void onIdle() {
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fminf(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}
