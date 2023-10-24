

/* vertex and triangle layout */
struct Vertex
{
    float x, y, z;
};
struct Face
{
    int v0, v1, v2;
};

struct Parameters
{
    float xMin, xMax;
    float yMin, yMax;
    float xPadding, yPadding;
    int xCount, yCount;
};
