#pragma once

enum ShaderType
{
    VERTEX_SHADER = GL_VERTEX_SHADER,
    FRAGMENT_SHADER = GL_FRAGMENT_SHADER
};

class Shader
{
public:
    Shader(const string& vs, const string& fs);
    ~Shader();

    void bind();
    template <class T> void setUniform(const string& name, const T& value);

private:
    GLuint mProgram;

    string readEntireFile(const string& filename);
    GLuint compileShader(ShaderType type, const string& sourceFile);
    void link();

    GLint getUniformLocation(const string& name);

};

template <> inline void Shader::setUniform(const string& name, const bool& value)
{
    bind();
    GL_CHECK(glUniform1i(getUniformLocation(name), (int)value));
}

template <> inline void Shader::setUniform(const string& name, const int& value)
{
    bind();
    GL_CHECK(glUniform1i(getUniformLocation(name), value));
}

template <> inline void Shader::setUniform(const string& name, const float& value)
{
    bind();
    GL_CHECK(glUniform1f(getUniformLocation(name), value));
}

template <> inline void Shader::setUniform(const string& name, const glm::vec2& value)
{
    bind();
    GL_CHECK(glUniform2f(getUniformLocation(name), value.x, value.y));
}

template <> inline void Shader::setUniform(const string& name, const glm::vec3& value)
{
    bind();
    GL_CHECK(glUniform3f(getUniformLocation(name), value.x, value.y, value.z));
}

template <> inline void Shader::setUniform(const string& name, const glm::vec4& value)
{
    bind();
    GL_CHECK(glUniform4f(getUniformLocation(name), value.x, value.y, value.z, value.w));
}

template <> inline void Shader::setUniform(const string& name, const glm::mat2& value)
{
    bind();
    GL_CHECK(glUniformMatrix2fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value)));
}

template <> inline void Shader::setUniform(const string& name, const glm::mat3& value)
{
    bind();
    GL_CHECK(glUniformMatrix3fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value)));
}

template <> inline void Shader::setUniform(const string& name, const glm::mat4& value)
{
    bind();
    GL_CHECK(glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value)));
}