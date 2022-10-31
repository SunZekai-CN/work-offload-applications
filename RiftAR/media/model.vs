#version 330

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

out vec3 oNormal;

uniform mat4 modelMatrix;
uniform mat4 modelViewProjectionMatrix;

void main()
{
    gl_Position = modelViewProjectionMatrix * vec4(position, 1.0);
    oNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
}