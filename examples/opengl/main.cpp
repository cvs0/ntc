// OpenGL 4.6 surfaceless host. Same fp32 decode path as the Vulkan example.
// The driver include extension is not required: the host splices ntc_decode.glsl.

#include "ntc/ntc.hpp"

#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif

static std::string read_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::string splice(const std::string& entry_in, const std::string& library) {
    std::string entry = entry_in;
    const std::string ext = "#extension GL_GOOGLE_include_directive : require\n";
    auto epos = entry.find(ext);
    if (epos != std::string::npos) entry.erase(epos, ext.size());
    const std::string key = "#include \"ntc_decode.glsl\"";
    auto pos = entry.find(key);
    if (pos == std::string::npos) return entry;
    return entry.substr(0, pos) + "\n" + library + "\n" + entry.substr(pos + key.size());
}

static void check_shader(GLuint s) {
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        GLsizei n = 0;
        glGetShaderInfoLog(s, sizeof(log), &n, log);
        throw std::runtime_error(std::string("shader: ") + log);
    }
}

static GLuint program_from(const std::string& comp_src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    const char* p = comp_src.c_str();
    glShaderSource(sh, 1, &p, nullptr);
    glCompileShader(sh);
    check_shader(sh);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        GLsizei n = 0;
        glGetProgramInfoLog(prog, sizeof(log), &n, log);
        throw std::runtime_error(std::string("link: ") + log);
    }
    glDeleteShader(sh);
    return prog;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ntc_gl file.ntc shader_dir\n";
        return 2;
    }
    try {
        ntc::TextureSet ts = ntc::load_file(argv[1]);
        const std::string dir = argv[2];
        auto cpu = ntc::decode_mip(ts, 0);
        auto up = ntc::make_gpu_upload(ts, 0, ntc::kFilterPoint, 4);
        const int w = static_cast<int>(ts.width);
        const int h = static_cast<int>(ts.height);
        const int c = ts.num_channels();

        auto eglGetPlatformDisplayEXT_p = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));
        if (!eglGetPlatformDisplayEXT_p) throw std::runtime_error("no egl platform display");
        EGLDisplay dpy =
            eglGetPlatformDisplayEXT_p(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        if (dpy == EGL_NO_DISPLAY) throw std::runtime_error("no egl display");
        if (!eglInitialize(dpy, nullptr, nullptr)) throw std::runtime_error("eglInitialize");
        if (!eglBindAPI(EGL_OPENGL_API)) throw std::runtime_error("eglBindAPI");
        EGLint ncfg = 0;
        eglGetConfigs(dpy, nullptr, 0, &ncfg);
        std::vector<EGLConfig> cfgs(static_cast<size_t>(ncfg));
        eglGetConfigs(dpy, cfgs.data(), ncfg, &ncfg);
        EGLConfig cfg = nullptr;
        for (EGLConfig c : cfgs) {
            EGLint rt = 0;
            eglGetConfigAttrib(dpy, c, EGL_RENDERABLE_TYPE, &rt);
            if (rt & EGL_OPENGL_BIT) {
                cfg = c;
                break;
            }
        }
        if (!cfg) throw std::runtime_error("no opengl config");
        // Request 4.6. llvmpipe on this host stops at 4.5; the shaders are
        // GLSL 450, which both versions accept. No 4.6-only feature is used.
        EGLContext ctx = EGL_NO_CONTEXT;
        for (int minor : {6, 5}) {
            const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION,
                                       4,
                                       EGL_CONTEXT_MINOR_VERSION,
                                       minor,
                                       EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                       EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                                       EGL_NONE};
            ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
            if (ctx) break;
        }
        if (!ctx) throw std::runtime_error("no gl context");
        if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) throw std::runtime_error("make current");

        const GLubyte* vendor = glGetString(GL_VENDOR);
        const GLubyte* renderer = glGetString(GL_RENDERER);
        const GLubyte* version = glGetString(GL_VERSION);
        std::cout << "gl vendor " << vendor << " renderer " << renderer << " version " << version << "\n";
        GLint subgroup = 0;
        GLint n = 0;
        glGetIntegerv(GL_NUM_EXTENSIONS, &n);
        for (GLint i = 0; i < n; ++i) {
            const char* e = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i)));
            if (e && std::string(e) == "GL_KHR_shader_subgroup") {
                glGetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroup);
            }
        }
        std::cout << "gl_subgroup " << subgroup << "\n";

        const std::string lib = read_text(dir + "/ntc_decode.glsl");
        const std::string comp = splice(read_text(dir + "/ntc_decompress_gl.comp"), lib);
        GLuint prog = program_from(comp);

        GLuint ssbo[3];
        glGenBuffers(3, ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[0]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(up.desc.size() * 4), up.desc.data(),
                     GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[1]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(up.blob.size()), up.blob.data(),
                     GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(w * h * c * 4), nullptr, GL_DYNAMIC_READ);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo[2]);

        GLuint ubo = 0;
        glGenBuffers(1, &ubo);
        uint32_t pc[4] = {0, 0, 0, 0};
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(pc), pc, GL_STATIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, ubo);

        glUseProgram(prog);
        glDispatchCompute((w + 7) / 8, (h + 7) / 8, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glFinish();

        std::vector<float> gpu(static_cast<size_t>(w * h * c));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(gpu.size() * 4), gpu.data());
        float err = 0.f;
        for (size_t i = 0; i < gpu.size(); ++i) err = std::max(err, std::fabs(gpu[i] - cpu[i]));
        std::cout << "gl_compute_max_abs " << err << "\n";
        if (err > 1e-3f) {
            std::cerr << "gl mismatch\n";
            return 1;
        }

        // Fragment sample into an RGBA32F renderbuffer. Channels beyond 4 are
        // not written; the first three are compared to the reference.
        const std::string frag = splice(read_text(dir + "/ntc_sample_gl.frag"), lib);
        const std::string vert = read_text(dir + "/fullscreen_gl.vert");
        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        const char* vp = vert.c_str();
        const char* fp = frag.c_str();
        glShaderSource(vs, 1, &vp, nullptr);
        glCompileShader(vs);
        check_shader(vs);
        glShaderSource(fs, 1, &fp, nullptr);
        glCompileShader(fs);
        check_shader(fs);
        GLuint gp = glCreateProgram();
        glAttachShader(gp, vs);
        glAttachShader(gp, fs);
        glLinkProgram(gp);
        GLint linked = 0;
        glGetProgramiv(gp, GL_LINK_STATUS, &linked);
        if (!linked) throw std::runtime_error("graphics link");

        GLuint fbo = 0, rb = 0;
        glGenFramebuffers(1, &fbo);
        glGenRenderbuffers(1, &rb);
        glBindRenderbuffer(GL_RENDERBUFFER, rb);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, w, h);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("framebuffer");
        }
        glViewport(0, 0, w, h);
        glUseProgram(gp);
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, ubo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1]);
        GLuint vao = 0;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        std::vector<float> rgba(static_cast<size_t>(w * h * 4));
        glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, rgba.data());
        float ferr = 0.f;
        const int ncmp = std::min(c, 3);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int k = 0; k < ncmp; ++k) {
                    float got = rgba[(static_cast<size_t>(y) * w + x) * 4 + k];
                    float want = cpu[(static_cast<size_t>(y) * w + x) * c + k];
                    ferr = std::max(ferr, std::fabs(got - want));
                }
            }
        }
        std::cout << "gl_fragment_max_abs " << ferr << "\n";
        if (ferr > 1e-3f) {
            std::cerr << "fragment mismatch\n";
            return 1;
        }
        std::cout << "ok\n";
        eglDestroyContext(dpy, ctx);
        eglTerminate(dpy);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
