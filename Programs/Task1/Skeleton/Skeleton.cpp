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

GPUProgram gpuProgram;

const int TEXTURE_RADIUS = 30;
const int TEXTURE_DIAMETER = TEXTURE_RADIUS * 2;

const int MAX_NODES = 50;
const float SATURATION = 0.05f;
const int MAX_PLANE = 100; // DO NOT TUNE OVER 144
const float CIRCLE_RADIUS = 0.06f;

const int MAX_EXPERIMENT = 100;
const int MAX_SIMULATION = 300;

const float D_STAR = 0.6f;
const float RESISTANCE = 0.05f;

// ------------------------------------------- HYPERBOLIC CODE ---------------------------------------------
vec3 pushToHyperbolic(vec3 v) {
	float z = sqrtf(v.x * v.x + v.y * v.y + 1.0f);
	return { v.x, v.y, z };
}

vec3 projToHyperbolic(vec3 v) {
	float denom = sqrtf(1 - v.x * v.x - v.y * v.y);
	return { v.x / denom, v.y / denom, 1.0f / denom };
}

vec3 projToHyperbolic(vec2 v) {
	return projToHyperbolic(vec3(v.x, v.y, 1.0f));
}

float getHyperbDistance(vec3 p, vec3 q) {
	return acoshf(-(p.x * q.x + p.y * q.y - p.z * q.z));
}

vec3 getVectorOfHyperbLine(vec3 p, vec3 result) {
	float dist = getHyperbDistance(p, result);

	vec3 vector = (result - p * coshf(dist)) / sinhf(dist);
	return vector;
}

vec3 getResultOfHyperbLine(vec3 p, vec3 v, float dist) {
	vec3 r = p * coshf(dist) + v * sinhf(dist);
	return r;
}

vec3 mirrorTwice(vec3 p, vec3 m1, vec3 m2) {
	vec3 v, mirrored;

	float dist1 = getHyperbDistance(p, m1);
	v = getVectorOfHyperbLine(p, m1);
	mirrored = getResultOfHyperbLine(p, v, getHyperbDistance(p, m1) * 2);

	float dist2 = getHyperbDistance(mirrored, m2);
	v = getVectorOfHyperbLine(mirrored, m2);
	mirrored = getResultOfHyperbLine(mirrored, v, getHyperbDistance(mirrored, m2) * 2);
	
	return mirrored;
}

vec3 calcFe(vec3 v, float d) {
	return (d > D_STAR) ? v / d : -v / d;
}

vec3 calcFn(vec3 v, float d) {
	return (d > D_STAR) ? -v / (100.0f * d) : -v / d;
}

bool operator!=(vec3 p, vec3 q) {
	return p.x != p.x || p.y != p.y || p.z != p.z;
}

vec3 normalizeHyperb(vec3 v) {
	return v / sqrtf(v.x * v.x + v.y * v.y - v.z * v.z);
}
// --------------------------------------- END OF HYPERBOLIC CODE ------------------------------------------

// -------------------------------------------- CIRCLE CODE ------------------------------------------------
class Circle {
	unsigned int vao;
	unsigned int vbo[2];

	float radius;
	vec3 originalCenter;
	vec3 velocity = { 0.0f, 0.0f, 0.0f };
	Texture* texture;
public:
	int id;
	vec3 hCenter;
	std::vector<Circle*> adjacents;

	bool adjacentTo(Circle* circle) {
		for (const auto& item : adjacents) {
			if (circle == item)
				return true;
		}
		return false;
	}

	Circle(int id, vec2 center, float radius) : id(id), radius(radius) {
		hCenter = projToHyperbolic(center);
		originalCenter = hCenter;
		glGenVertexArrays(1, &vao);
		glGenBuffers(2, vbo);
		initTexture();		
		refreshBuffer();
	}

	void loadOldCenter() {
		hCenter = originalCenter;
		velocity = { 0.0f, 0.0f, 0.0f };
	}

	void initTexture() {
		int width = TEXTURE_DIAMETER, height = TEXTURE_DIAMETER;
		std::vector<vec4> image(width * height);

		vec4 color1 = { rand() % 256 / 256.0f, rand() % 256 / 256.0f, rand() % 256 / 256.0f, 1.0f };
		vec4 color2 = { rand() % 256 / 256.0f, rand() % 256 / 256.0f, rand() % 256 / 256.0f, 1.0f };
		vec4 color3 = { rand() % 256 / 256.0f, rand() % 256 / 256.0f, rand() % 256 / 256.0f, 1.0f };

		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				if ((i - TEXTURE_RADIUS) * (i - TEXTURE_RADIUS) 
					+ (j - TEXTURE_RADIUS) * (j - TEXTURE_RADIUS) <= TEXTURE_RADIUS * TEXTURE_RADIUS) {
					if (i >= TEXTURE_DIAMETER * 2 / 3)
						image[i * height + j] = color1;
					else if (i >= TEXTURE_DIAMETER / 3)
						image[i * height + j] = color2;
					else
						image[i * height + j] = color3;
				}
				else
					image[i * height + j] = { 0.0f, 0.0f, 0.0f, 0.0f };
			}
		}

		texture = new Texture(width, height, image);

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		float verticesTex[] = { 0, 0, 1, 0, 1, 1, 0, 1 };

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(verticesTex),
			verticesTex,
			GL_STATIC_DRAW);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1,
			2, GL_FLOAT, GL_FALSE,
			0, NULL);
	}

	std::vector<vec3> calcPoints() {
		std::vector<vec3> hPoints;
		vec3 origo = { 0.0f, 0.0f, 1.0f };

		vec3 v = getVectorOfHyperbLine(origo, hCenter);
		float dist = getHyperbDistance(origo, hCenter);
		vec3 m1 = getResultOfHyperbLine(origo, v, dist / 4.0f);
		vec3 m2 = getResultOfHyperbLine(origo, v, (3.0f / 4.0f) * dist);

		vec3 hPointOnOrigo = projToHyperbolic({ -radius, -radius, 1.0f });
		vec3 hPoint = mirrorTwice(hPointOnOrigo, m1, m2);
		hPoints.push_back(hPoint);

		hPointOnOrigo = projToHyperbolic({ radius, -radius, 1.0f });
		hPoint = mirrorTwice(hPointOnOrigo, m1, m2);
		hPoints.push_back(hPoint);

		hPointOnOrigo = projToHyperbolic({ radius, radius, 1.0f });
		hPoint = mirrorTwice(hPointOnOrigo, m1, m2);
		hPoints.push_back(hPoint);

		hPointOnOrigo = projToHyperbolic({ -radius, radius, 1.0f });
		hPoint = mirrorTwice(hPointOnOrigo, m1, m2);
		hPoints.push_back(hPoint);

		return hPoints;
	}

	void refreshBuffer() {
		std::vector<vec3> hPoints = calcPoints();

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		float vertices[4 * 3];

		for (int i = 0; i < 4; ++i) {
			vertices[i * 3] = hPoints[i].x;
			vertices[i * 3 + 1] = hPoints[i].y;
			vertices[i * 3 + 2] = hPoints[i].z;
		}

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vertices),
			vertices,
			GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			3, GL_FLOAT, GL_FALSE,
			0, NULL);
	}

	void draw() {
		gpuProgram.setUniform(*texture, "textureUnit");
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}

	vec3 getForced(std::vector<Circle*> circles, long time) {
		float delta = time * 0.001f;
		vec3 origo = { 0.0f, 0.0f, 1.0f };		
		vec3 sumForce = { 0.0f, 0.0f, 0.0f };

		for (const auto& item : circles) {
			if (item->hCenter != item->hCenter)
				continue;

			vec3 v = getVectorOfHyperbLine(hCenter, item->hCenter);
			float d = getHyperbDistance(hCenter, item->hCenter);
			
			if (adjacentTo(item)) {
				sumForce = sumForce + calcFe(v, d);
			}
			else if (item != this) {
				sumForce = sumForce + calcFn(v, d);
			}
		}
		
		vec3 v = getVectorOfHyperbLine(hCenter, origo);
		float d = getHyperbDistance(hCenter, origo);
		vec3 homeForce = calcFe(v, d) * 0.9f;

		velocity = velocity + sumForce + homeForce - RESISTANCE * velocity;
		velocity = normalizeHyperb(velocity);
		vec3 newCenter = getResultOfHyperbLine(hCenter, velocity, delta);
		if (newCenter != newCenter)
			return hCenter;

		return pushToHyperbolic(newCenter);
	}

	~Circle() {
		delete texture;
	}
};
// ----------------------------------------- END OF CIRCLE CODE --------------------------------------------

// --------------------------------------------- LINE CODE -------------------------------------------------
class Line {
	unsigned int vao;
	unsigned int vbo;
	Circle* nStart;
	Circle* nEnd;
public:
	Line(Circle* nStart, Circle* nEnd): nStart(nStart), nEnd(nEnd) {
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		refreshBuffer();
	}

	void refreshBuffer() {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		vec3 start = nStart->hCenter;
		vec3 end = nEnd->hCenter;

		float vertices[] = { start.x, start.y, start.z, end.x, end.y, end.z };

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vertices),
			vertices,
			GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			3, GL_FLOAT, GL_FALSE,
			0, NULL);
	}

	void draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_LINES, 0, 4);
	}
};

class Experiment {
public:
	std::vector<vec3> hCenters;

	void buildCenters() {
		for (int i = 0; i < MAX_NODES; ++i) {
			vec2 center = { (rand() % MAX_PLANE - MAX_PLANE / 2) * 0.01f, (rand() % MAX_PLANE - MAX_PLANE / 2) * 0.01f };
			hCenters.push_back(projToHyperbolic(center));
		}
	}

	float calcEnergy(std::vector<Circle*> circles) {
		float accu = 0.0f;
		for (size_t i = 0; i < circles.size(); ++i) {
			Circle* currentCircle = circles[i];

			for (size_t j = 0; j < circles.size(); ++j) {
				Circle* comparedCircle = circles[j];
				float dist = getHyperbDistance(hCenters[j], hCenters[i]);

				if (currentCircle->adjacentTo(comparedCircle)) {
					accu -= (1.0f / dist);
				}
				else if (currentCircle != comparedCircle) {
					accu += (1.0f / dist);
				}
			}
		}
		return accu;
	}
};
// ------------------------------------------ END OF LINE CODE ---------------------------------------------

// ------------------------------------------ COLLECTION CODE ----------------------------------------------
class Collection {
	std::vector<Circle*> circles;
	std::vector<Line*> lines;

	std::vector<Experiment*> experiments;
	vec2 wCenter = { 0.0f, 0.0f };
public:
	void initWorld() {
		gpuProgram.setUniform(vec3(wCenter.x, wCenter.y, 1.0f), "hC");
		buildNodes();
		buildEdges();
	}

	void buildNodes() {
		for (int i = 0; i < MAX_NODES; ++i) {
			vec2 center = { (rand() % MAX_PLANE - MAX_PLANE / 2) * 0.01f, (rand() % MAX_PLANE - MAX_PLANE / 2) * 0.01f };
			circles.push_back(new Circle(i, center, CIRCLE_RADIUS));
		}
	}

	void buildEdges() {
		int numOfEdgesNeeded = (int)((MAX_NODES * (MAX_NODES - 1) / 2) * SATURATION);

		int edgesSoFar = 0;
		while (edgesSoFar < numOfEdgesNeeded) {
			int random1 = (rand() % MAX_NODES);
			int random2 = (rand() % MAX_NODES);
			Circle* node1 = circles[random1];
			Circle* node2 = circles[random2];
			if (node1 != node2 && !node1->adjacentTo(node2)) {
				node1->adjacents.push_back(node2);
				node2->adjacents.push_back(node1);
				lines.push_back(new Line(node1, node2));
				++edgesSoFar;
			}
		}
	}

	void drawWorld() {
		gpuProgram.setUniform(false, "loadTexture");
		for (const auto& item : lines) {
			item->draw();
		}

		gpuProgram.setUniform(true, "loadTexture");
		for (const auto& item : circles) {
			item->draw();
		}
		gpuProgram.setUniform(false, "loadTexture");
	}

	void translateWorld(vec2 displacement) {
		wCenter = wCenter - displacement;
		gpuProgram.setUniform(pushToHyperbolic({ wCenter.x, wCenter.y, 1.0f }), "hC"); //projToHyperbolic would be too intense
	}

	// Using Potential Calculations (electric)
	void rearrangeWorld() {
		std::vector<Experiment*> experiments;

		for (int i = 0; i < MAX_EXPERIMENT; ++i) {
			Experiment* exp = new Experiment();
			exp->buildCenters();
			experiments.push_back(exp);
		}

		int min = 0;
		float minEnergy = experiments[0]->calcEnergy(circles);
		for (int i = 1; i < MAX_EXPERIMENT; ++i) {
			float energy = experiments[i]->calcEnergy(circles);
			if (minEnergy > energy) {
				min = i;
				minEnergy = energy;
			}
		}

		std::vector<vec3> results = experiments[min]->hCenters;
		for (int i = 0; i < MAX_NODES; ++i) {
			circles[i]->hCenter = results[i];
		}

		for (const auto& item : circles) {
			item->refreshBuffer();
		}
		for (const auto& item : lines) {
			item->refreshBuffer();
		}

		experiments.clear();
	}

	void simulateForces(long delta) {
		std::vector<vec3> newCenters;
		newCenters.resize(MAX_NODES);

		for (int i = 0; i < MAX_NODES; ++i) {
			newCenters[i] = circles[i]->getForced(circles, delta);
		}

		for (int i = 0; i < MAX_NODES; ++i) {
			circles[i]->hCenter = newCenters[i];
			circles[i]->refreshBuffer();
		}

		for (const auto& item : lines) {
			item->refreshBuffer();
		}
	}

	void resetTranslation() {
		wCenter = { 0.0f, 0.0f };
		gpuProgram.setUniform(vec3(wCenter.x, wCenter.y, 1.0f), "hC");
	}

	void resetWorld() {
		for (const auto& item : circles) {
			item->loadOldCenter();
			item->refreshBuffer();
		}
		for (const auto& item : lines) {
			item->refreshBuffer();
		}
	}

	void listCircles() {
		for (const auto& item : circles) {
			printf("#%d center: (%g %g %g)\n", item->id, item->hCenter.x, item->hCenter.y, item->hCenter.z);
		}
	}

	~Collection() {
		int n = lines.size();

		for (int i = 0; i < MAX_NODES; ++i) {
			delete circles[i];
		}

		for (int i = 0; i < n; ++i) {
			delete lines[i];
		}
	}
};
// -------------------------------------- END OF COLLECTION CODE -------------------------------------------

const char * const vertexSource = R"(
	#version 330
	precision highp float;

	uniform vec3 hC;
	layout(location = 0) in vec3 vp;
	layout(location = 1) in vec2 vt;

	out vec2 texCoord;

	float getHyperbDistance(vec3 p, vec3 q) {
		return acosh(-1.0f * ((p.x * q.x) + (p.y * q.y) - (p.z * q.z)));
	}

	vec3 getVectorOfHyperbLine(vec3 p, vec3 result) {
		float dist = getHyperbDistance(p, result);

		vec3 vector = (result - p * cosh(dist)) / sinh(dist);
		return vector;
	}

	vec3 getResultOfHyperbLine(vec3 p, vec3 v, float dist) {
		vec3 r = p * cosh(dist) + v * sinh(dist);
		return r;
	}

	vec3 mirrorTwice(vec3 p, vec3 m1, vec3 m2) {
		vec3 v, mirrored;

		float dist1 = getHyperbDistance(p, m1);
		v = getVectorOfHyperbLine(p, m1);
		mirrored = getResultOfHyperbLine(p, v, dist1 * 2);

		float dist2 = getHyperbDistance(mirrored, m2);
		v = getVectorOfHyperbLine(mirrored, m2);
		mirrored = getResultOfHyperbLine(mirrored, v, dist2 * 2);
	
		return mirrored;
	}

	void main() {
		if (hC != vec3(0, 0, 1)) {
			vec3 origo = vec3(0, 0, 1);
			vec3 v = getVectorOfHyperbLine(origo, hC);
			float dist = getHyperbDistance(origo, hC);
			vec3 m1 = getResultOfHyperbLine(origo, v, dist / 4);
			vec3 m2 = getResultOfHyperbLine(origo, v, (3/4) * dist);	
			vec3 t = mirrorTwice(vp, m1, m2);
			gl_Position = vec4(t.x / t.z, t.y / t.z, 1, 1);
		}
		else {
			gl_Position = vec4(vp.x / vp.z, vp.y / vp.z, 1, 1);
		}
		texCoord = vt;
	}
)";

const char * const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform sampler2D textureUnit;
	uniform vec3 color;
	uniform bool loadTexture;

	in vec2 texCoord;
	out vec4 outColor;

	void main() {
		if (loadTexture)
			outColor = texture(textureUnit, texCoord);
		else
			outColor = vec4(color, 1);
	}
)";

Collection collection;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
	gpuProgram.setUniform(false, "loadTexture");
	gpuProgram.setUniform(vec3(0.1059f, 0.6039f, 0.8666f), "color");
	// To let fragment shader blend transparency, source: https://community.khronos.org/t/transparency-does-not-work/75406
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// -------
	collection.initWorld();
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	collection.drawWorld();
	glutSwapBuffers();
}

bool canSimulate = false;
bool buttonPressed = false;
vec2 savedPos;
vec2 nextPos;
long lastTime = 0;
int simulationCount = 0;

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case ' ':
		collection.resetWorld();
		collection.rearrangeWorld();
		canSimulate = true;
		break;
	case 'r':
		collection.resetTranslation();
		break;
	case 'R':
		collection.resetWorld();
		break;
	case 'l':
		collection.listCircles();
		break;
	}
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	nextPos = { cX, cY };

	if (buttonPressed) {
		vec2 displacement = nextPos - savedPos;
		if (length(displacement) < 1e-9f)
			return;

		collection.translateWorld(displacement);
		savedPos = nextPos;
		glutPostRedisplay();
	}
}

void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	savedPos = { cX, cY };
	buttonPressed = (state == GLUT_DOWN && button == GLUT_RIGHT_BUTTON);
}

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	if (canSimulate) {
		++simulationCount;
		collection.simulateForces(time - lastTime);
		glutPostRedisplay();
	}
	if (simulationCount > MAX_SIMULATION) {
		canSimulate = false;
		simulationCount = 0;
	}
	lastTime = time;
}
