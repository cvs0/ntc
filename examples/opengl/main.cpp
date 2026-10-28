// OpenGL host. Linux uses a surfaceless EGL context. Windows uses a hidden
// WGL window. The driver include extension is not required: the host splices
// ntc_decode.glsl. Shaders are GLSL 450.

#include "ntc/ntc.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <GL/gl.h>
#else
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
#endif

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#if !defined(GL_VERSION_2_0)
typedef char GLchar;
#endif
#if !defined(GL_VERSION_1_5)
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;
#endif
typedef uint64_t GLuint64;
#ifndef GL_COMPUTE_SHADER
#define GL_COMPUTE_SHADER 0x91B9
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_SHADER_STORAGE_BARRIER_BIT 0x00002000
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_VERTEX_SHADER 0x8B31
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_RENDERBUFFER 0x8D41
#define GL_FRAMEBUFFER 0x8D40
#define GL_RGBA32F 0x8814
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#define GL_NUM_EXTENSIONS 0x821D
#define GL_UNIFORM_BUFFER 0x8A11
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_READ 0x88E9
#define GL_TIME_ELAPSED 0x88BF
#define GL_QUERY_RESULT 0x8866
#endif
#define NTC_GL_API(ret, name, ...) static ret(APIENTRY* name)(__VA_ARGS__) = nullptr
NTC_GL_API(GLuint, glCreateShader, GLenum);
NTC_GL_API(void, glShaderSource, GLuint, GLsizei, const char* const*, const GLint*);
NTC_GL_API(void, glCompileShader, GLuint);
NTC_GL_API(void, glGetShaderiv, GLuint, GLenum, GLint*);
NTC_GL_API(void, glGetShaderInfoLog, GLuint, GLsizei, GLsizei*, GLchar*);
NTC_GL_API(void, glDeleteShader, GLuint);
NTC_GL_API(GLuint, glCreateProgram, void);
NTC_GL_API(void, glAttachShader, GLuint, GLuint);
NTC_GL_API(void, glLinkProgram, GLuint);
NTC_GL_API(void, glGetProgramiv, GLuint, GLenum, GLint*);
NTC_GL_API(void, glGetProgramInfoLog, GLuint, GLsizei, GLsizei*, GLchar*);
NTC_GL_API(void, glUseProgram, GLuint);
NTC_GL_API(void, glDeleteProgram, GLuint);
NTC_GL_API(void, glGenBuffers, GLsizei, GLuint*);
NTC_GL_API(void, glBindBuffer, GLenum, GLuint);
NTC_GL_API(void, glBufferData, GLenum, GLsizeiptr, const void*, GLenum);
NTC_GL_API(void, glBindBufferBase, GLenum, GLuint, GLuint);
NTC_GL_API(void, glGetBufferSubData, GLenum, GLintptr, GLsizeiptr, void*);
NTC_GL_API(void, glDispatchCompute, GLuint, GLuint, GLuint);
NTC_GL_API(void, glMemoryBarrier, GLbitfield);
NTC_GL_API(void, glGenFramebuffers, GLsizei, GLuint*);
NTC_GL_API(void, glBindFramebuffer, GLenum, GLuint);
NTC_GL_API(void, glGenRenderbuffers, GLsizei, GLuint*);
NTC_GL_API(void, glBindRenderbuffer, GLenum, GLuint);
NTC_GL_API(void, glRenderbufferStorage, GLenum, GLenum, GLsizei, GLsizei);
NTC_GL_API(void, glFramebufferRenderbuffer, GLenum, GLenum, GLenum, GLuint);
NTC_GL_API(GLenum, glCheckFramebufferStatus, GLenum);
NTC_GL_API(void, glGenVertexArrays, GLsizei, GLuint*);
NTC_GL_API(void, glBindVertexArray, GLuint);
// glDrawArrays is OpenGL 1.1. Windows gl.h declares it and opengl32.dll
// exports it. A function-pointer redefinition fails on clang++ (MSVC ABI)
// and on cl.exe. Do not load it with wglGetProcAddress.
NTC_GL_API(const GLubyte*, glGetStringi, GLenum, GLuint);
NTC_GL_API(void, glGenQueries, GLsizei, GLuint*);
NTC_GL_API(void, glBeginQuery, GLenum, GLuint);
NTC_GL_API(void, glEndQuery, GLenum);
NTC_GL_API(void, glGetQueryObjectui64v, GLuint, GLenum, GLuint64*);
NTC_GL_API(void, glCompressedTexImage2D, GLenum, GLint, GLenum, GLsizei, GLsizei, GLint, GLsizei, const void*);
NTC_GL_API(void, glActiveTexture, GLenum);
static void load_gl() {
#define LOAD(name) name = reinterpret_cast<decltype(name)>(wglGetProcAddress(#name)); if (!name) throw std::runtime_error("missing " #name)
    LOAD(glCreateShader);
    LOAD(glShaderSource);
    LOAD(glCompileShader);
    LOAD(glGetShaderiv);
    LOAD(glGetShaderInfoLog);
    LOAD(glDeleteShader);
    LOAD(glCreateProgram);
    LOAD(glAttachShader);
    LOAD(glLinkProgram);
    LOAD(glGetProgramiv);
    LOAD(glGetProgramInfoLog);
    LOAD(glUseProgram);
    LOAD(glDeleteProgram);
    LOAD(glGenBuffers);
    LOAD(glBindBuffer);
    LOAD(glBufferData);
    LOAD(glBindBufferBase);
    LOAD(glGetBufferSubData);
    LOAD(glDispatchCompute);
    LOAD(glMemoryBarrier);
    LOAD(glGenFramebuffers);
    LOAD(glBindFramebuffer);
    LOAD(glGenRenderbuffers);
    LOAD(glBindRenderbuffer);
    LOAD(glRenderbufferStorage);
    LOAD(glFramebufferRenderbuffer);
    LOAD(glCheckFramebufferStatus);
    LOAD(glGenVertexArrays);
    LOAD(glBindVertexArray);
    LOAD(glGetStringi);
    LOAD(glGenQueries);
    LOAD(glBeginQuery);
    LOAD(glEndQuery);
    LOAD(glGetQueryObjectui64v);
    LOAD(glCompressedTexImage2D);
    LOAD(glActiveTexture);
#undef LOAD
}
#ifndef WGL_CONTEXT_MAJOR_VERSION_ARB
#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#endif
typedef HGLRC(WINAPI* PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC, HGLRC, const int*);
#endif

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

static std::string inject_after_version(const std::string& src, const std::string& extra) {
    const auto pos = src.find('\n');
    if (pos == std::string::npos) return extra + src;
    return src.substr(0, pos + 1) + extra + src.substr(pos + 1);
}

static bool gl_ext(const char* name) {
    GLint n = 0;
    glGetIntegerv(0x821D, &n);  // GL_NUM_EXTENSIONS
    for (GLint i = 0; i < n; ++i) {
        const auto* s = reinterpret_cast<const char*>(glGetStringi(0x1F03, static_cast<GLuint>(i)));
        if (s && std::strcmp(s, name) == 0) return true;
    }
    return false;
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

#ifndef GL_SUBGROUP_SIZE_KHR
#define GL_SUBGROUP_SIZE_KHR 0x9532
#endif
#ifndef GL_TIME_ELAPSED
#define GL_TIME_ELAPSED 0x88BF
#endif
#ifndef GL_QUERY_RESULT
#define GL_QUERY_RESULT 0x8866
#endif

static bool renderer_is_software(const std::string& renderer) {
    std::string low = renderer;
    for (char& ch : low) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return low.find("llvmpipe") != std::string::npos || low.find("softpipe") != std::string::npos ||
           low.find("swiftshader") != std::string::npos;
}

struct PassStats {
    double median_ms = 0;
    double p95_ms = 0;
};

static PassStats summarize(std::vector<double> ms) {
    PassStats s;
    if (ms.empty()) return s;
    std::sort(ms.begin(), ms.end());
    s.median_ms = ms[ms.size() / 2];
    const size_t i95 = std::min(ms.size() - 1, static_cast<size_t>(std::ceil(0.95 * ms.size())) - 1);
    s.p95_ms = ms[i95];
    return s;
}

static void report_pass(const char* name, const PassStats& s, double texels, double out_bytes, bool software) {
    const double sec = s.median_ms / 1e3;
    const double tps = sec > 0 ? texels / sec : 0;
    const double gbs = sec > 0 ? (out_bytes / sec) / 1e9 : 0;
    std::cout << "bench " << name << " median_ms " << s.median_ms << " p95_ms " << s.p95_ms << " texels_s " << tps
              << " gb_s " << gbs;
    if (software) std::cout << " not_a_performance_claim";
    std::cout << "\n";
}

static void write_f32(const std::string& path, const float* data, size_t n) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
}

struct BlockFile {
    uint32_t w = 0, h = 0, format = 0;
    std::vector<uint8_t> payload;
};

static BlockFile load_blocks(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (raw.size() < 20) throw std::runtime_error("short block file");
    BlockFile f;
    std::memcpy(&f.w, raw.data(), 4);
    std::memcpy(&f.h, raw.data() + 4, 4);
    std::memcpy(&f.format, raw.data() + 8, 4);
    uint32_t n = 0, magic = 0;
    std::memcpy(&n, raw.data() + 12, 4);
    std::memcpy(&magic, raw.data() + 16, 4);
    if (magic != 0x4243544Eu) throw std::runtime_error("bad block magic");
    if (20u + n > raw.size()) throw std::runtime_error("short block payload");
    f.payload.assign(raw.begin() + 20, raw.begin() + 20 + n);
    return f;
}

static std::vector<double> time_query(int warmup, int iters, const std::function<void()>& body) {
    for (int i = 0; i < warmup; ++i) {
        body();
        glFinish();
    }
    std::vector<GLuint> qs(static_cast<size_t>(iters));
    glGenQueries(iters, qs.data());
    std::vector<double> ms;
    ms.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        glBeginQuery(GL_TIME_ELAPSED, qs[static_cast<size_t>(i)]);
        body();
        glEndQuery(GL_TIME_ELAPSED);
        glFinish();
        GLuint64 ns = 0;
        glGetQueryObjectui64v(qs[static_cast<size_t>(i)], GL_QUERY_RESULT, &ns);
        ms.push_back(static_cast<double>(ns) / 1e6);
    }
    return ms;
}

static void run_gl_bench(const ntc::TextureSet& ts, const ntc::GpuUpload& up, const std::string& dir,
                         const std::string& out_dir, int warmup, int iters, bool software, const std::string& bc7,
                         const std::string& bc1, const std::string& bc5) {
    std::filesystem::create_directories(out_dir);
    const int w = static_cast<int>(ts.width);
    const int h = static_cast<int>(ts.height);
    const int c = ts.num_channels();
    const std::string lib = read_text(dir + "/ntc_decode.glsl");
    GLuint prog = program_from(splice(read_text(dir + "/ntc_decompress_gl.comp"), lib));
    GLuint sample = program_from(splice(read_text(dir + "/ntc_sample_res_gl.comp"), lib));
    GLuint stage = 0;
    try {
        std::string stage_src = inject_after_version(read_text(dir + "/ntc_decompress_stage.comp"), "#define NTC_GL 1\n");
        stage = program_from(splice(stage_src, lib));
    } catch (const std::exception& ex) {
        std::cout << "bench load_fp32_stage skipped " << ex.what() << "\n";
    }
    GLuint lit = 0;
    if (c >= 6) lit = program_from(splice(read_text(dir + "/ntc_lit_gl.comp"), lib));

    uint64_t mip_texels = 0;
    int nmips = 0;
    for (int m = 0; m < static_cast<int>(ts.num_mips); ++m) {
        const int mw = std::max(1, w >> m);
        const int mh = std::max(1, h >> m);
        mip_texels += static_cast<uint64_t>(mw) * static_cast<uint64_t>(mh);
        nmips = m + 1;
        if (mw == 1 && mh == 1) break;
    }
    const uint64_t max_out = static_cast<uint64_t>(std::max(w * h * c, w * h * 4));
    GLuint ssbo[3];
    glGenBuffers(3, ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(up.desc.size() * 4), up.desc.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(up.blob.size()), up.blob.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(max_out * 4), nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo[2]);
    GLuint ubo = 0;
    glGenBuffers(1, &ubo);

    auto set_pc = [&](uint32_t a, uint32_t b, uint32_t cw, uint32_t ch) {
        uint32_t pc[4] = {a, b, cw, ch};
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(pc), pc, GL_STATIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, ubo);
    };

    auto load_body = [&]() {
        glUseProgram(prog);
        for (int m = 0; m < nmips; ++m) {
            const int mw = std::max(1, w >> m);
            const int mh = std::max(1, h >> m);
            set_pc(static_cast<uint32_t>(m), 0, 0, 0);
            glDispatchCompute((mw + 7) / 8, (mh + 7) / 8, 1);
        }
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    const double load_bytes = static_cast<double>(mip_texels) * c * 4.0;
    std::cout << "note load_pass one_dispatch_per_mip 8x8 point_ntc_eval ssbo_f32\n";
    report_pass("load_fp32", summarize(time_query(warmup, iters, load_body)), static_cast<double>(mip_texels),
                load_bytes, software);
    if (stage) {
        auto stage_body = [&]() {
            glUseProgram(stage);
            for (int m = 0; m < nmips; ++m) {
                const int mw = std::max(1, w >> m);
                const int mh = std::max(1, h >> m);
                set_pc(static_cast<uint32_t>(m), 0, 0, 0);
                glDispatchCompute((mw + 7) / 8, (mh + 7) / 8, 1);
            }
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        };
        report_pass("load_fp32_stage", summarize(time_query(warmup, iters, stage_body)),
                    static_cast<double>(mip_texels), load_bytes, software);
    }
    // float16_t from, in order: the EXT arithmetic types, AMD half, NV gpu shader 5.
    GLuint fp16 = 0;
    GLuint fp16_stage = 0;
    const char* fp16_tag = nullptr;
    const char* fp16_def = nullptr;
    if (gl_ext("GL_EXT_shader_explicit_arithmetic_types_float16") ||
        gl_ext("GL_EXT_shader_explicit_arithmetic_types")) {
        fp16_tag = "GL_EXT_shader_explicit_arithmetic_types_float16";
        fp16_def = "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
                   "#define NTC_FP16 1\n#define NTC_FP16_EXT_DONE 1\n";
    } else if (gl_ext("GL_AMD_gpu_shader_half_float")) {
        fp16_tag = "GL_AMD_gpu_shader_half_float";
        fp16_def = "#extension GL_AMD_gpu_shader_half_float : require\n"
                   "#define NTC_FP16 1\n#define NTC_FP16_AMD 1\n#define NTC_FP16_EXT_DONE 1\n";
    } else if (gl_ext("GL_NV_gpu_shader5")) {
        fp16_tag = "GL_NV_gpu_shader5";
        fp16_def = "#extension GL_NV_gpu_shader5 : require\n"
                   "#define NTC_FP16 1\n#define NTC_FP16_NV 1\n#define NTC_FP16_EXT_DONE 1\n";
    }
    if (fp16_def) {
        try {
            std::string src = inject_after_version(read_text(dir + "/ntc_decompress_gl.comp"), fp16_def);
            fp16 = program_from(splice(src, lib));
            std::string ssrc = inject_after_version(read_text(dir + "/ntc_decompress_stage.comp"),
                                                    std::string("#define NTC_GL 1\n") + fp16_def);
            fp16_stage = program_from(splice(ssrc, lib));
            std::cout << "fp16_ext " << fp16_tag << "\n";
        } catch (const std::exception& ex) {
            std::cout << "bench load_fp16 skipped compile\n";
            std::cout << "bench load_fp16_stage skipped compile\n";
            std::cout << "fp16_ext_error " << fp16_tag << "\n";
            fp16 = 0;
            fp16_stage = 0;
        }
    } else {
        std::cout << "bench load_fp16 skipped no_gl_fp16\n";
        std::cout << "bench load_fp16_stage skipped no_gl_fp16\n";
    }
    if (fp16) {
        auto body = [&]() {
            glUseProgram(fp16);
            for (int m = 0; m < nmips; ++m) {
                const int mw = std::max(1, w >> m);
                const int mh = std::max(1, h >> m);
                set_pc(static_cast<uint32_t>(m), 0, 0, 0);
                glDispatchCompute((mw + 7) / 8, (mh + 7) / 8, 1);
            }
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        };
        report_pass("load_fp16", summarize(time_query(warmup, iters, body)), static_cast<double>(mip_texels),
                    load_bytes, software);
    }
    if (fp16_stage) {
        auto body = [&]() {
            glUseProgram(fp16_stage);
            for (int m = 0; m < nmips; ++m) {
                const int mw = std::max(1, w >> m);
                const int mh = std::max(1, h >> m);
                set_pc(static_cast<uint32_t>(m), 0, 0, 0);
                glDispatchCompute((mw + 7) / 8, (mh + 7) / 8, 1);
            }
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        };
        report_pass("load_fp16_stage", summarize(time_query(warmup, iters, body)), static_cast<double>(mip_texels),
                    load_bytes, software);
    }

    std::vector<float> gpu(static_cast<size_t>(w * h * c));
    load_body();
    glFinish();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(gpu.size() * 4), gpu.data());
    write_f32(out_dir + "/gl_ntc_fp32.f32", gpu.data(), gpu.size());
    std::cout << "wrote " << out_dir << "/gl_ntc_fp32.f32 " << w << " " << h << " " << c << "\n";

    auto sample_res = [&](int ow, int oh, const char* name) {
        if (software && (ow > 512 || oh > 512)) {
            std::cout << "bench " << name << " skipped cpu_device\n";
            return;
        }
        const uint64_t nfloat = static_cast<uint64_t>(ow) * static_cast<uint64_t>(oh) * static_cast<uint64_t>(c);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(nfloat * 4), nullptr, GL_DYNAMIC_READ);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo[2]);
        auto body = [&]() {
            glUseProgram(sample);
            set_pc(0, 0, static_cast<uint32_t>(ow), static_cast<uint32_t>(oh));
            glDispatchCompute((ow + 7) / 8, (oh + 7) / 8, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        };
        report_pass(name, summarize(time_query(warmup, iters, body)), static_cast<double>(ow) * oh,
                    static_cast<double>(nfloat) * 4.0, software);
    };
    std::cout << "note sample_pass one_dispatch 8x8 point_ntc_eval mip0 ssbo_f32 no_filter\n";
    sample_res(w, h, "sample_native_fp32");
    sample_res(1920, 1080, "sample_1920x1080_fp32");
    sample_res(3840, 2160, "sample_3840x2160_fp32");

    if (lit) {
        auto body = [&]() {
            glUseProgram(lit);
            set_pc(0, 0, static_cast<uint32_t>(w), static_cast<uint32_t>(h));
            glDispatchCompute((w + 7) / 8, (h + 7) / 8, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        };
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(w * h * 3 * 4), nullptr, GL_DYNAMIC_READ);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo[2]);
        report_pass("lit_native_fp32", summarize(time_query(warmup, iters, body)), static_cast<double>(w) * h,
                    static_cast<double>(w) * h * 12.0, software);
        body();
        glFinish();
        std::vector<float> rgb(static_cast<size_t>(w * h * 3));
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(rgb.size() * 4), rgb.data());
        write_f32(out_dir + "/gl_lit_fp32.f32", rgb.data(), rgb.size());
        std::cout << "wrote " << out_dir << "/gl_lit_fp32.f32\n";
    } else {
        std::cout << "bench lit skipped channels\n";
    }

    auto sample_blocks = [&](const std::string& path, const char* tag) {
        if (path.empty()) {
            std::cout << "bench " << tag << " skipped no_file\n";
            return;
        }
        BlockFile bf = load_blocks(path);
        GLenum fmt = 0;
        if (bf.format == 1) fmt = 0x8E8C;
        else if (bf.format == 2) fmt = 0x83F1;
        else if (bf.format == 3) fmt = 0x8DBD;
        if (!fmt) {
            std::cout << "bench " << tag << " skipped format\n";
            return;
        }
        const std::string src = read_text(dir + "/tex_sample_gl.comp");
        GLuint tp = program_from(src);
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glActiveTexture(0x84C0);
        glBindTexture(0x0DE1, tex);
        glTexParameteri(0x0DE1, 0x2801, 0x2601);
        glTexParameteri(0x0DE1, 0x2800, 0x2601);
        glTexParameteri(0x0DE1, 0x2802, 0x812F);
        glTexParameteri(0x0DE1, 0x2803, 0x812F);
        glPixelStorei(0x0CF5, 1);
        glGetError();
        glCompressedTexImage2D(0x0DE1, 0, fmt, static_cast<GLsizei>(bf.w), static_cast<GLsizei>(bf.h), 0,
                               static_cast<GLsizei>(bf.payload.size()), bf.payload.data());
        if (glGetError() != 0) {
            std::cout << "bench " << tag << " skipped format\n";
            glDeleteTextures(1, &tex);
            glDeleteProgram(tp);
            return;
        }
        auto time_one = [&](int ow, int oh, const char* suffix) {
            if (software && (ow > 512 || oh > 512)) {
                std::cout << "bench " << tag << suffix << " skipped cpu_device\n";
                return;
            }
            const uint64_t nfloat = static_cast<uint64_t>(ow) * oh * 4;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(nfloat * 4), nullptr, GL_DYNAMIC_READ);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[2]);
            auto body = [&]() {
                glUseProgram(tp);
                set_pc(static_cast<uint32_t>(ow), static_cast<uint32_t>(oh), 0, 0);
                glDispatchCompute((ow + 7) / 8, (oh + 7) / 8, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            };
            report_pass((std::string(tag) + suffix).c_str(), summarize(time_query(warmup, iters, body)),
                        static_cast<double>(ow) * oh, static_cast<double>(nfloat) * 4.0, software);
        };
        time_one(1920, 1080, "_1920x1080");
        time_one(3840, 2160, "_3840x2160");
        const uint64_t native = static_cast<uint64_t>(bf.w) * bf.h * 4;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(native * 4), nullptr, GL_DYNAMIC_READ);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[2]);
        glUseProgram(tp);
        set_pc(bf.w, bf.h, 0, 0);
        glDispatchCompute((bf.w + 7) / 8, (bf.h + 7) / 8, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glFinish();
        std::vector<float> px(static_cast<size_t>(native));
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(px.size() * 4), px.data());
        write_f32(out_dir + "/" + tag + ".f32", px.data(), px.size());
        std::cout << "wrote " << out_dir << "/" << tag << ".f32 " << bf.w << " " << bf.h << "\n";
        glDeleteTextures(1, &tex);
        glDeleteProgram(tp);
    };
    sample_blocks(bc7, "bc7");
    sample_blocks(bc1, "bc1");
    sample_blocks(bc5, "bc5");
    glDeleteProgram(prog);
    glDeleteProgram(sample);
    if (lit) glDeleteProgram(lit);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ntc_gl file.ntc shader_dir [--bench outdir] [--warmup N] [--iters N] "
                     "[--bc7 f] [--bc1 f] [--bc5 f]\n";
        return 2;
    }
    std::string bench_dir, bc7_path, bc1_path, bc5_path;
    int warmup = 8;
    int iters = 20;
    try {
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            auto need = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value");
                return argv[++i];
            };
            if (a == "--bench") bench_dir = need();
            else if (a == "--warmup") warmup = std::stoi(need());
            else if (a == "--iters") iters = std::stoi(need());
            else if (a == "--bc7") bc7_path = need();
            else if (a == "--bc1") bc1_path = need();
            else if (a == "--bc5") bc5_path = need();
            else {
                std::cerr << "unknown " << a << "\n";
                return 2;
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
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

#ifdef _WIN32
        WNDCLASSW wc{};
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"ntc_gl_hidden";
        RegisterClassW(&wc);
        HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"ntc", WS_OVERLAPPEDWINDOW, 0, 0, 32, 32, nullptr,
                                    nullptr, wc.hInstance, nullptr);
        if (!hwnd) throw std::runtime_error("CreateWindow");
        HDC hdc = GetDC(hwnd);
        PIXELFORMATDESCRIPTOR pfd{};
        pfd.nSize = sizeof(pfd);
        pfd.nVersion = 1;
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
        pfd.iPixelType = PFD_TYPE_RGBA;
        pfd.cColorBits = 32;
        const int pf = ChoosePixelFormat(hdc, &pfd);
        if (!pf || !SetPixelFormat(hdc, pf, &pfd)) throw std::runtime_error("SetPixelFormat");
        HGLRC dummy = wglCreateContext(hdc);
        if (!dummy || !wglMakeCurrent(hdc, dummy)) throw std::runtime_error("wgl dummy context");
        auto create_attribs = reinterpret_cast<PFNWGLCREATECONTEXTATTRIBSARBPROC>(
            wglGetProcAddress("wglCreateContextAttribsARB"));
        if (!create_attribs) throw std::runtime_error("no wglCreateContextAttribsARB");
        HGLRC hrc = nullptr;
        for (int minor : {6, 5}) {
            const int attrs[] = {WGL_CONTEXT_MAJOR_VERSION_ARB, 4, WGL_CONTEXT_MINOR_VERSION_ARB, minor,
                                 WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB, 0};
            hrc = create_attribs(hdc, nullptr, attrs);
            if (hrc) break;
        }
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(dummy);
        if (!hrc || !wglMakeCurrent(hdc, hrc)) throw std::runtime_error("no gl context");
        load_gl();
#else
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
#endif

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
        const bool software = renderer_is_software(reinterpret_cast<const char*>(renderer));
        std::cout << "gl_software " << (software ? 1 : 0) << "\n";

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
        if (!bench_dir.empty()) {
            run_gl_bench(ts, up, dir, bench_dir, warmup, iters, software, bc7_path, bc1_path, bc5_path);
        }
        std::cout << "ok\n";
#ifdef _WIN32
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(hrc);
        ReleaseDC(hwnd, hdc);
        DestroyWindow(hwnd);
#else
        eglDestroyContext(dpy, ctx);
        eglTerminate(dpy);
#endif
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
