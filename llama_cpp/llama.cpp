#define LLAMA_API_INTERNAL
#include "llama.h"

#include "unicode.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

#ifdef GGML_USE_CUBLAS
#  include "ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#  include "ggml-opencl.h"
#elif defined(GGML_USE_VULKAN)
#  include "ggml-vulkan.h"
#elif defined(GGML_USE_SYCL)
#  include "ggml-sycl.h"
#elif defined(GGML_USE_KOMPUTE)
#   include "ggml-kompute.h"
#endif

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif
#ifdef GGML_USE_MPI
#  include "ggml-mpi.h"
#endif
#ifndef QK_K
#  ifdef GGML_QKK_64
#    define QK_K 64
#  else
#    define QK_K 256
#  endif
#endif

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
            #include <fcntl.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <io.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <forward_list>
#include <fstream>
#include <functional>
//#include <filesystem>
//#include <initializer_list>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <iostream>
#include <fstream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

#define LLAMA_MAX_NODES   8192
#define LLAMA_MAX_EXPERTS 8


//
// logging
//

LLAMA_ATTRIBUTE_FORMAT(2, 3)
static void llama_log_internal        (ggml_log_level level, const char* format, ...);
static void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

void LLAMA_TRACK(std::string tag, size_t value);
void llama_enable_tracker(std::string tag);
void llama_disable_tracker(std::string tag);

//
// helpers
//
#include <Windows.h>
#include <DbgHelp.h>
#include <iostream>

#pragma comment(lib, "Dbghelp.lib")
void PrintStackTrace() {
    void* stack[100];
    int frames = CaptureStackBackTrace(0, 100, stack, NULL);
    LLAMA_LOG_INFO("Stack: %d frames\n", frames);

    // Initialize symbol handler
    SymInitialize(GetCurrentProcess(), NULL, TRUE);

    // Print stack trace
    SYMBOL_INFO* symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256, 1);
    for (int i = 0; i < frames; ++i) {
        symbol->MaxNameLen = 255;
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

        if (SymFromAddr(GetCurrentProcess(), (DWORD64)(stack[i]), 0, symbol)) {
            LLAMA_LOG_INFO("%s: %p\n", symbol->Name, (void*)symbol->Address);
        } else {
            LLAMA_LOG_INFO("Unknown: %p\n", symbol->Name, (void*)symbol->Address);
        }
    }
    free(symbol);

    // Cleanup
    SymCleanup(GetCurrentProcess());
}


static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static const int kv_context = 1;
std::string quick_ts();

static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    std::string result;
    for (size_t pos = 0; ; pos += search.length()) {
        auto new_pos = s.find(search, pos);
        if (new_pos == std::string::npos) {
            result += s.substr(pos, s.size() - pos);
            break;
        }
        result += s.substr(pos, new_pos - pos) + replace;
        pos = new_pos;
    }
    s = std::move(result);
}

static bool is_float_close(float a, float b, float abs_tol) {
    // Check for non-negative tolerance
    if (abs_tol < 0.0) {
        throw std::invalid_argument("Tolerance must be non-negative");
    }

    // Exact equality check
    if (a == b) {
        return true;
    }

    // Check for infinities
    if (std::isinf(a) || std::isinf(b)) {
        return false;
    }

    // Regular comparison using the provided absolute tolerance
    return std::fabs(b - a) <= abs_tol;
}

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

LLAMA_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

//
// gguf constants (sync with gguf.py)
//

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_PERSIMMON,
    LLM_ARCH_REFACT,
    LLM_ARCH_BERT,
    LLM_ARCH_NOMIC_BERT,
    LLM_ARCH_BLOOM,
    LLM_ARCH_STABLELM,
    LLM_ARCH_QWEN,
    LLM_ARCH_QWEN2,
    LLM_ARCH_PHI2,
    LLM_ARCH_PLAMO,
    LLM_ARCH_CODESHELL,
    LLM_ARCH_ORION,
    LLM_ARCH_INTERNLM2,
    LLM_ARCH_MINICPM,
    LLM_ARCH_GEMMA,
    LLM_ARCH_STARCODER2,
    LLM_ARCH_UNKNOWN,
};

static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,           "llama"      },
    { LLM_ARCH_FALCON,          "falcon"     },
    { LLM_ARCH_GPT2,            "gpt2"       },
    { LLM_ARCH_GPTJ,            "gptj"       },
    { LLM_ARCH_GPTNEOX,         "gptneox"    },
    { LLM_ARCH_MPT,             "mpt"        },
    { LLM_ARCH_BAICHUAN,        "baichuan"   },
    { LLM_ARCH_STARCODER,       "starcoder"  },
    { LLM_ARCH_PERSIMMON,       "persimmon"  },
    { LLM_ARCH_REFACT,          "refact"     },
    { LLM_ARCH_BERT,            "bert"       },
    { LLM_ARCH_NOMIC_BERT,      "nomic-bert" },
    { LLM_ARCH_BLOOM,           "bloom"      },
    { LLM_ARCH_STABLELM,        "stablelm"   },
    { LLM_ARCH_QWEN,            "qwen"       },
    { LLM_ARCH_QWEN2,           "qwen2"      },
    { LLM_ARCH_PHI2,            "phi2"       },
    { LLM_ARCH_PLAMO,           "plamo"      },
    { LLM_ARCH_CODESHELL,       "codeshell"  },
    { LLM_ARCH_ORION,           "orion"      },
    { LLM_ARCH_INTERNLM2,       "internlm2"  },
    { LLM_ARCH_MINICPM,         "minicpm"    },
    { LLM_ARCH_GEMMA,           "gemma"      },
    { LLM_ARCH_STARCODER2,      "starcoder2" },
    { LLM_ARCH_UNKNOWN,         "(unknown)"  },
};

enum llm_kv {
    LLM_KV_GENERAL_ARCHITECTURE,
    LLM_KV_GENERAL_QUANTIZATION_VERSION,
    LLM_KV_GENERAL_ALIGNMENT,
    LLM_KV_GENERAL_NAME,
    LLM_KV_GENERAL_AUTHOR,
    LLM_KV_GENERAL_URL,
    LLM_KV_GENERAL_DESCRIPTION,
    LLM_KV_GENERAL_LICENSE,
    LLM_KV_GENERAL_SOURCE_URL,
    LLM_KV_GENERAL_SOURCE_HF_REPO,

    LLM_KV_CONTEXT_LENGTH,
    LLM_KV_EMBEDDING_LENGTH,
    LLM_KV_BLOCK_COUNT,
    LLM_KV_FEED_FORWARD_LENGTH,
    LLM_KV_USE_PARALLEL_RESIDUAL,
    LLM_KV_TENSOR_DATA_LAYOUT,
    LLM_KV_EXPERT_COUNT,
    LLM_KV_EXPERT_USED_COUNT,
    LLM_KV_POOLING_TYPE,

    LLM_KV_ATTENTION_HEAD_COUNT,
    LLM_KV_ATTENTION_HEAD_COUNT_KV,
    LLM_KV_ATTENTION_MAX_ALIBI_BIAS,
    LLM_KV_ATTENTION_CLAMP_KQV,
    LLM_KV_ATTENTION_KEY_LENGTH,
    LLM_KV_ATTENTION_VALUE_LENGTH,
    LLM_KV_ATTENTION_LAYERNORM_EPS,
    LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,
    LLM_KV_ATTENTION_CAUSAL,

    LLM_KV_ROPE_DIMENSION_COUNT,
    LLM_KV_ROPE_FREQ_BASE,
    LLM_KV_ROPE_SCALE_LINEAR,
    LLM_KV_ROPE_SCALING_TYPE,
    LLM_KV_ROPE_SCALING_FACTOR,
    LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,
    LLM_KV_ROPE_SCALING_FINETUNED,

    LLM_KV_TOKENIZER_MODEL,
    LLM_KV_TOKENIZER_LIST,
    LLM_KV_TOKENIZER_TOKEN_TYPE,
    LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,
    LLM_KV_TOKENIZER_SCORES,
    LLM_KV_TOKENIZER_MERGES,
    LLM_KV_TOKENIZER_BOS_ID,
    LLM_KV_TOKENIZER_EOS_ID,
    LLM_KV_TOKENIZER_UNK_ID,
    LLM_KV_TOKENIZER_SEP_ID,
    LLM_KV_TOKENIZER_PAD_ID,
    LLM_KV_TOKENIZER_ADD_BOS,
    LLM_KV_TOKENIZER_ADD_EOS,
    LLM_KV_TOKENIZER_ADD_PREFIX,
    LLM_KV_TOKENIZER_HF_JSON,
    LLM_KV_TOKENIZER_RWKV,
};

static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_ARCHITECTURE,          "general.architecture"                  },
    { LLM_KV_GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },
    { LLM_KV_GENERAL_ALIGNMENT,             "general.alignment"                     },
    { LLM_KV_GENERAL_NAME,                  "general.name"                          },
    { LLM_KV_GENERAL_AUTHOR,                "general.author"                        },
    { LLM_KV_GENERAL_URL,                   "general.url"                           },
    { LLM_KV_GENERAL_DESCRIPTION,           "general.description"                   },
    { LLM_KV_GENERAL_LICENSE,               "general.license"                       },
    { LLM_KV_GENERAL_SOURCE_URL,            "general.source.url"                    },
    { LLM_KV_GENERAL_SOURCE_HF_REPO,        "general.source.huggingface.repository" },

    { LLM_KV_CONTEXT_LENGTH,                "%s.context_length"        },
    { LLM_KV_EMBEDDING_LENGTH,              "%s.embedding_length"      },
    { LLM_KV_BLOCK_COUNT,                   "%s.block_count"           },
    { LLM_KV_FEED_FORWARD_LENGTH,           "%s.feed_forward_length"   },
    { LLM_KV_USE_PARALLEL_RESIDUAL,         "%s.use_parallel_residual" },
    { LLM_KV_TENSOR_DATA_LAYOUT,            "%s.tensor_data_layout"    },
    { LLM_KV_EXPERT_COUNT,                  "%s.expert_count"          },
    { LLM_KV_EXPERT_USED_COUNT,             "%s.expert_used_count"     },
    { LLM_KV_POOLING_TYPE ,                 "%s.pooling_type"          },

    { LLM_KV_ATTENTION_HEAD_COUNT,          "%s.attention.head_count"             },
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,       "%s.attention.head_count_kv"          },
    { LLM_KV_ATTENTION_MAX_ALIBI_BIAS,      "%s.attention.max_alibi_bias"         },
    { LLM_KV_ATTENTION_CLAMP_KQV,           "%s.attention.clamp_kqv"              },
    { LLM_KV_ATTENTION_KEY_LENGTH,          "%s.attention.key_length"             },
    { LLM_KV_ATTENTION_VALUE_LENGTH,        "%s.attention.value_length"           },
    { LLM_KV_ATTENTION_LAYERNORM_EPS,       "%s.attention.layer_norm_epsilon"     },
    { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,   "%s.attention.layer_norm_rms_epsilon" },
    { LLM_KV_ATTENTION_CAUSAL,              "%s.attention.causal"                 },

    { LLM_KV_ROPE_DIMENSION_COUNT,          "%s.rope.dimension_count"                 },
    { LLM_KV_ROPE_FREQ_BASE,                "%s.rope.freq_base"                       },
    { LLM_KV_ROPE_SCALE_LINEAR,             "%s.rope.scale_linear"                    },
    { LLM_KV_ROPE_SCALING_TYPE,             "%s.rope.scaling.type"                    },
    { LLM_KV_ROPE_SCALING_FACTOR,           "%s.rope.scaling.factor"                  },
    { LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,     "%s.rope.scaling.original_context_length" },
    { LLM_KV_ROPE_SCALING_FINETUNED,        "%s.rope.scaling.finetuned"               },

    { LLM_KV_TOKENIZER_MODEL,               "tokenizer.ggml.model"              },
    { LLM_KV_TOKENIZER_LIST,                "tokenizer.ggml.tokens"             },
    { LLM_KV_TOKENIZER_TOKEN_TYPE,          "tokenizer.ggml.token_type"         },
    { LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,    "tokenizer.ggml.token_type_count"   },
    { LLM_KV_TOKENIZER_SCORES,              "tokenizer.ggml.scores"             },
    { LLM_KV_TOKENIZER_MERGES,              "tokenizer.ggml.merges"             },
    { LLM_KV_TOKENIZER_BOS_ID,              "tokenizer.ggml.bos_token_id"       },
    { LLM_KV_TOKENIZER_EOS_ID,              "tokenizer.ggml.eos_token_id"       },
    { LLM_KV_TOKENIZER_UNK_ID,              "tokenizer.ggml.unknown_token_id"   },
    { LLM_KV_TOKENIZER_SEP_ID,              "tokenizer.ggml.seperator_token_id" },
    { LLM_KV_TOKENIZER_PAD_ID,              "tokenizer.ggml.padding_token_id"   },
    { LLM_KV_TOKENIZER_ADD_BOS,             "tokenizer.ggml.add_bos_token"      },
    { LLM_KV_TOKENIZER_ADD_EOS,             "tokenizer.ggml.add_eos_token"      },
    { LLM_KV_TOKENIZER_ADD_PREFIX,          "tokenizer.ggml.add_space_prefix"   },
    { LLM_KV_TOKENIZER_HF_JSON,             "tokenizer.huggingface.json"        },
    { LLM_KV_TOKENIZER_RWKV,                "tokenizer.rwkv.world"              },
};

struct LLM_KV {
    LLM_KV(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    std::string operator()(llm_kv kv) const {
        return ::format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch));
    }
};

enum llm_tensor {
    LLM_TENSOR_TOKEN_EMBD,
    LLM_TENSOR_TOKEN_EMBD_NORM,
    LLM_TENSOR_TOKEN_TYPES,
    LLM_TENSOR_POS_EMBD,
    LLM_TENSOR_OUTPUT,
    LLM_TENSOR_OUTPUT_NORM,
    LLM_TENSOR_ROPE_FREQS,
    LLM_TENSOR_ATTN_Q,
    LLM_TENSOR_ATTN_K,
    LLM_TENSOR_ATTN_V,
    LLM_TENSOR_ATTN_QKV,
    LLM_TENSOR_ATTN_OUT,
    LLM_TENSOR_ATTN_NORM,
    LLM_TENSOR_ATTN_NORM_2,
    LLM_TENSOR_ATTN_OUT_NORM,
    LLM_TENSOR_ATTN_ROT_EMBD,
    LLM_TENSOR_FFN_GATE_INP,
    LLM_TENSOR_FFN_NORM,
    LLM_TENSOR_FFN_GATE,
    LLM_TENSOR_FFN_DOWN,
    LLM_TENSOR_FFN_UP,
    LLM_TENSOR_FFN_ACT,
    LLM_TENSOR_FFN_DOWN_EXP,
    LLM_TENSOR_FFN_GATE_EXP,
    LLM_TENSOR_FFN_UP_EXP,
    LLM_TENSOR_ATTN_Q_NORM,
    LLM_TENSOR_ATTN_K_NORM,
    LLM_TENSOR_LAYER_OUT_NORM,
};

static const std::map<llm_arch, std::map<llm_tensor, std::string>> LLM_TENSOR_NAMES = {
    {
        LLM_ARCH_LLAMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
        },
    },
    {
        LLM_ARCH_BAICHUAN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_FALCON,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_NORM_2,     "blk.%d.attn_norm_2" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_GPT2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_GPTJ,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        },
    },
    {
        LLM_ARCH_GPTNEOX,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_PERSIMMON,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd"},
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm"},
            { LLM_TENSOR_OUTPUT,          "output"},
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm"},
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv"},
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output"},
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm"},
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm"},
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm"},
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down"},
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up"},
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd"},
        },
    },
    {
        LLM_ARCH_MPT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_ACT,         "blk.%d.ffn.act" },
        },
    },
    {
        LLM_ARCH_STARCODER,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_REFACT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_BERT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.%d.layer_output_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_NOMIC_BERT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.%d.layer_output_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_BLOOM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_STABLELM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_PHI2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_PLAMO,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_CODESHELL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_ORION,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_INTERNLM2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_MINICPM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
        },
    },
    {
        LLM_ARCH_GEMMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_STARCODER2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_UNKNOWN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        },
    },
};

static llm_arch llm_arch_from_string(const std::string & name) {
    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLM_ARCH_UNKNOWN;
}

// helper to handle gguf constants
// usage:
//
//   const auto tn = LLM_TN(LLM_ARCH_LLAMA);
//
//   std::string name = tn(LLM_TENSOR_OUTPUT);                     -> "output"
//   std::string name = tn(LLM_TENSOR_TOKEN_EMBD, "bias");         -> "token_embd.bias"
//   std::string name = tn(LLM_TENSOR_ATTN_NORM, "weight", 3);     -> "blk.3.attn_norm.weight"
//
struct LLM_TN {
    LLM_TN(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    std::string operator()(llm_tensor tensor) const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return LLM_TENSOR_NAMES.at(arch).at(tensor);
    }

    std::string operator()(llm_tensor tensor, const std::string & suffix) const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return LLM_TENSOR_NAMES.at(arch).at(tensor) + "." + suffix;
    }

    std::string operator()(llm_tensor tensor, int bid) const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return ::format(LLM_TENSOR_NAMES.at(arch).at(tensor).c_str(), bid);
    }

    std::string operator()(llm_tensor tensor, const std::string & suffix, int bid) const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return ::format(LLM_TENSOR_NAMES.at(arch).at(tensor).c_str(), bid) + "." + suffix;
    }

    std::string operator()(llm_tensor tensor, const std::string & suffix, int bid, int xid) const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return ::format(LLM_TENSOR_NAMES.at(arch).at(tensor).c_str(), bid, xid) + "." + suffix;
    }
};

//
// gguf helpers
//

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,   "none"   },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR, "linear" },
    { LLAMA_ROPE_SCALING_TYPE_YARN,   "yarn"   },
};

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const bool *)data)[i] ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}

static std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replace_all(val, "\\", "\\\\");
                        replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

//
// ggml helpers
//

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

//
// llama helpers
//

#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

template <typename T>
struct no_init {
    T value;
    no_init() { /* do nothing */ }
};
typedef struct llama_mem Llama_mem;

struct llama_mem {
    void *addr=NULL;
    size_t sz=0;
    uint16_t *refc;
    void *ref;

    llama_mem()
    {
        addr=ref=NULL;
        refc=NULL;
    }

    llama_mem( void *ptr )
    {
        addr = ptr;
        sz = 0;
        refc = (uint16_t*)malloc(sizeof(uint16_t));
        *refc = 1;
        ref = ptr;
    }

    llama_mem( Llama_mem *copy )
    {
        addr = copy->addr;
        sz = copy->sz;
        refc = copy->refc;
        *refc = *refc + 1;
        ref = copy->ref;
    }

    ~llama_mem()
    {
        if( !refc ) return;

        *refc = *refc - 1;
        if( *refc == 0 ) {
            free(ref);
            free(refc);
        }
    }
};

typedef struct senmemory_pool Llama_pool;
typedef struct senmemory_pool _Pool;

typedef struct Sp_searchable sp_searchable;
struct Sp_searchable {
    size_t inds[2];
    uint16_t refs[2];
    void *data;

    Sp_searchable(void *ptr)
    {
        data = ptr;
        inds[0] = 0;
        inds[1] = 0;
        refs[0] = 0;
        refs[1] = 0;
    }
};

typedef struct Sp_linked sp_linked;
typedef void voidcb(void *ptr);
typedef void voidcb2(void *ptr, uint16_t count);

struct Sp_linked {
    sp_linked *next=NULL;
    sp_linked *prev=NULL;
    void *data=NULL;
    uint16_t count=0;

    Sp_linked() {
        next=prev=NULL;
        data=NULL;
        count=0;
    }
    ~Sp_linked() {
        delete next;
    }
    void forall(voidcb2 *func)
    {
        sp_linked *i;
        for( i = this; i; i = i->next ) {
            func(i->data, count);
        }
    }

    sp_linked *insert(void *ptr)
    {
        if( !data ) {
            data=ptr;
            return this;
        }
        sp_linked *i = new sp_linked();
        i->data = ptr;

        i->next = next;
        i->prev = this;

        this->next = i;
        if( next )
            next->prev = i;
        return i;
    }
};
void delete_splinked(void *ptr)
{
    sp_linked *spl = (sp_linked*)ptr;
    delete spl;
}

typedef struct Sp_linkedarray sp_linkedarray;
typedef sp_linkedarray sp_la;
void sp_la_free( void *ptr, uint16_t count )
{
    if( !ptr ) return;
    free(ptr);
}
uint8_t sp_la_searchby;
void sp_la_shifter( void *ptr, uint16_t count )
{
    void **buf = (void**)ptr;
    uint16_t i;
    sp_searchable *sb;

    for( i=0; i<count; i++ ) {
        sb = (sp_searchable*)buf[i];
        sb->inds[ sp_la_searchby ]--;
    }
}

struct Sp_linkedarray {
    sp_linked *head;
    sp_linked *tail;
    uint16_t asize=512;

    Sp_linkedarray(uint16_t as) {
        asize=as;
        head=tail=new sp_linked();
        head->count = 0;
        head->data = malloc(asize*sizeof(void*));
        memset( head->data, 0, sizeof(void*)*asize );
    }
    ~Sp_linkedarray() { //! make sure to destroy the contents first as this only releases the mass pointers.
        head->forall(sp_la_free);
        delete head;
    }


    sp_linked *get(uint16_t tgtlist)
    {
        sp_linked *i;
        uint16_t listno;

        listno=0;
        //LLAMA_LOG_INFO("Get list %u\n", tgtlist);
        for( i = head; i; i = i->next, listno++ ) {
            if( listno == tgtlist ) {
                return i;
            }
            if( i == tail ) {
                listno++;
                sp_linked *item = i->insert(malloc(asize*sizeof(void*)));
                tail = item;
                memset( item->data, 0, sizeof(void*)*asize );
                return item;
            }
        }
        PrintStackTrace();
        throw "Unreachable code area 2\n";
    }
    sp_linked *get(size_t addr, uint16_t &remainder, uint16_t &listno, uint8_t searchby)
    {
        sp_linked *i, *j;
        void **buf;
        uint16_t n=0;
        size_t req=addr;

        if( addr == 0 ) {
            remainder=listno=0;
            return head;
        }

        listno=0;//! todo: remove empty lists
        for( i = head; i; i = i->next, listno++ ) {
            while( i->count == 0 ) {
                // remove empty list
                j = i->next;
                if( i->prev ) {
                    i->prev->next = j;
                } else {
                    head = j;
                }
                if( j ) {
                    j->prev = i->prev;
                    sp_la_searchby=searchby;
                    j->forall( sp_la_shifter );
                    i->next = NULL;
                    delete i;
                    i = j;
                } else {
                    tail = j = i->prev;
                    delete i;
                    addr += j->count;
                    i = j;
                    listno--;
                    break;
                }
            }

            if( addr < i->count ) {
                remainder=addr;
                //LLAMA_LOG_INFO("%s: get(%zu) found %u after %u lists.\n", __func__, req, remainder, listno);
                return i;
            }
            addr -= i->count;
            if( !i->next ) {
                if( addr == 0 && i->count < asize ) {
                    remainder=i->count;
                    LLAMA_LOG_INFO("warning %s: get(%zu) 4 found %u after %u lists.\n", __func__, req, remainder, listno);
                    return i;
                }
                listno++;
                sp_linked *item = i->insert(malloc(asize*sizeof(void*)));
                if( item->next == NULL ) {
                    tail = item;
                }
                memset( item->data, 0, sizeof(void*)*asize );
                remainder = addr;
                LLAMA_LOG_INFO("warning %s: get(%zu) 2 found %u after %u lists.\n", __func__, req, remainder, listno);
                return item;
            }
        }
        PrintStackTrace();
        throw "Unreachable code area 1\n";
    }

    void forall( voidcb *func )
    {
        uint16_t x;
        Sp_linked *i;
        void **buf;

        for( i = head; i; i = i->next ) {
            buf = (void**)i->data;
            for( x = 0; x < asize; x++ ) {
                if( buf[x] == NULL ) continue;
                func(buf[x]);
            }
        }
    }
};

template<typename T, typename S, typename Q>
class Sp_searchnest {
public:
    uint16_t units;
    uint16_t dataptr[2];
    size_t count;
    sp_la *data[2];

    Sp_searchnest(uint16_t u, uint16_t d1, uint16_t d2) {
        units = u;
        dataptr[0] = d1;
        dataptr[1] = d2;
        data[0] = new sp_la(units);
        data[1] = new sp_la(units);
        count = 0;
    }
    ~Sp_searchnest() {
        if( data[0] ) {
            delete data[0];
            delete data[1];
            data[0]=NULL;
        }
    }
    void shift_ind_r( sp_linked *ptr, uint16_t searchby )
    {
        // shift ind right by 1 for all after ptr
        sp_searchable *sb;
        void **buf;
        uint16_t o;
        for( ; ptr; ptr = ptr->next ) {
            buf = (void**)ptr->data;
            for( o=0; o<ptr->count; o++ ) {
                sb = (sp_searchable*)buf[o];
                if( sb )
                    sb->inds[searchby]++;
            }
        }
    }
    void shift_pointers_r( sp_linked *ptr, uint16_t offset, uint8_t searchby )
    {
        uint16_t i;
        sp_searchable *sb;
        void **buf = (void**)ptr->data;
        //LLAMA_LOG_INFO("spr(%u) %u/%u\n", searchby, offset, ptr->count);
        if( ptr->count == units ) { // we have to go to the next list; this one is full
            Sp_linked *next = ptr->next;
            void **b2;
            uint16_t j=units;

            if( next==NULL || next->count == units ) {
                // create a new storage area
                //LLAMA_LOG_INFO("spr: insert new next (%p, %zu)\n", next, next ? next->count : 0);
                sp_linked *item = ptr->insert(malloc(units*sizeof(void*)));
                memset( item->data, 0, sizeof(void*)*units );
                if( next ) {
                    shift_ind_r(next, searchby);
                } else {
                    data[searchby]->tail = item;
                }
                item->count = 1;
                b2 = (void**)item->data;
            } else {
                b2 = (void**)next->data;
                next->count++;
                for( j = next->count-1; j>0; j-- ) {
                    b2[j] = b2[j-1];
                    sb = (sp_searchable*)b2[j];
                    if( sb )
                        sb->refs[searchby]=j;
                }
            }
            b2[0] = buf[units-1];
            sb = (sp_searchable*)b2[0];
            if( sb ) {
                sb->inds[searchby]++;
                sb->refs[searchby]=0;
            }
            //LLAMA_LOG_INFO("overflow complete\n");
        } else {
            ptr->count++;
        }

        for( i = ptr->count-1; i>offset; i-- ) {
            buf[i] = buf[i-1];
            sb = (sp_searchable*)buf[i];
            if( sb )
                sb->refs[searchby]=i;
        }

        buf[offset]=NULL;
        //LLAMA_LOG_INFO("%s: done\n", __func__);
    }
    void shift_pointers_l( sp_linked *ptr, uint16_t offset, uint8_t searchby )
    {
        uint16_t i;
        void **buf = (void**)ptr->data;
        sp_searchable *sb;
        //LLAMA_LOG_INFO("spl %u/%u %u\n", offset, ptr->count, searchby);
        for( i=offset+1; i<ptr->count; i++ ) {
            buf[i-1] = buf[i];
            sb = (sp_searchable*)buf[i];
            if( sb )
                sb->refs[searchby]=i-1;
        }
        ptr->count--;
    }
    sp_searchable *bs( uint8_t searchby, T* ref, size_t &M, uint16_t &miditem, uint16_t &listno )
    {
        S *valS = (S*)(char**)( ((uint8_t*)ref + dataptr[0] ) );
        Q *valQ = (Q*)(char**)( ((uint8_t*)ref + dataptr[1] ) );
        //LLAMA_LOG_INFO("search: %zu %u for %p/%p\n", count, dataptr[searchby], *valS, *valQ);
        sp_linked *mid;
        size_t L,R,Ml,Mr;
        void **buf;
        S *sptr;
        Q *qptr;
        sp_searchable * midref;

        LLAMA_TRACK("sp_search", 0);

        L=M=0;
        R=count>0?count-1:0;
        miditem=0;
        listno=0;
        while(L<R) {
            M=L+floor((float)(R-L)/2.0);
            mid = data[searchby]->get(M,miditem,listno,searchby);
            LLAMA_TRACK("sp_search", 1);
            buf = (void**)mid->data;
            midref = (sp_searchable*)( buf[miditem] );
            Ml=Mr=M;
            LLAMA_TRACK("sp_search", 2);
            if( midref == NULL ) {
                LLAMA_LOG_INFO("%s error: midref==null M=%zu L=%zu R=%zu\n", __func__, M, L, R);
                throw "error\n";
            }
            if( searchby == 0 )
                sptr = (S*)(char**)( (uint8_t*)midref->data + dataptr[0] );
            else
                qptr = (Q*)(char**)( (uint8_t*)midref->data + dataptr[1] );

            LLAMA_TRACK("sp_search", 3);
            if( searchby == 0 ) {
                //LLAMA_LOG_INFO("%s: compare s %p(%zu) vs %p\n", __func__, (void*)*sptr, M, (void*)*valS);
                if( *sptr < *valS ) {
                    L = M = M+1;
                } else if( *sptr == *valS ) {
                    L = M;
                    break;
                } else {
                    R = M = M>0?M - 1:0;
                }
            } else {
                //LLAMA_LOG_INFO("%s: compare q %p vs %p\n", __func__, (void*)*qptr, (void*)*valQ);
                if( *qptr < *valQ ) {
                    L = M;
                } else if( *qptr == *valQ ) {
                    L = M;
                    break;
                } else {
                    R = M = M>0?M - 1:0;
                }
            }
            LLAMA_TRACK("sp_search", 4);
        }
        if( M == 0 ) {
            mid = data[searchby]->head;
            miditem=0;
            listno=0;
            LLAMA_TRACK("sp_search", 5);
        } else if( M == count ) {
            miditem++;
            if( miditem == units ) {
                miditem=0;
                listno++;
            }
            LLAMA_TRACK("sp_search", 6);
            return NULL;
        } else {
            mid = data[searchby]->get( M, miditem, listno, searchby );
        }
        buf = (void**)mid->data;
        midref = (sp_searchable*)( buf[miditem] );
        //LLAMA_LOG_INFO("%s: M=%zu, miditem=%u, listno=%u\n", __func__, M, miditem, listno);
        LLAMA_TRACK("sp_search", 9);
        return midref;
    }
    /*
    sp_searchable *bslb( uint8_t searchby, T* ref, size_t &M, uint16_t &miditem, uint16_t &listno )
    {
        S *valS = (S*)(char**)( ((uint8_t*)ref + dataptr[0] ) );
        Q *valQ = (Q*)(char**)( ((uint8_t*)ref + dataptr[1] ) );
        //LLAMA_LOG_INFO("search: %zu %u for %p/%p\n", count, dataptr[searchby], *valS, *valQ);
        sp_linked *mid;
        size_t L,R,Ml,Mr;
        void **buf;
        S *sptr;
        Q *qptr;
        sp_searchable * midref;

        L=M=0;
        R=count>0?count-1:0;
        miditem=0;
        listno=0;
        while(L<R) {
            M=L+ceil((float)(R-L)/2.0);
            mid = data[searchby]->get(M,miditem,listno,searchby);
            buf = (void**)mid->data;
            midref = (sp_searchable*)( buf[miditem] );
            Ml=Mr=M;
            if( midref == NULL ) {
                LLAMA_LOG_INFO("%s error: midref==null M=%zu L=%zu R=%zu\n", __func__, M, L, R);
                throw "error\n";
            }
            if( searchby == 0 )
                sptr = (S*)(char**)( (uint8_t*)midref->data + dataptr[0] );
            else
                qptr = (Q*)(char**)( (uint8_t*)midref->data + dataptr[1] );

            if( searchby == 0 ) {
                //LLAMA_LOG_INFO("%s: compare s %p(%zu) vs %p\n", __func__, (void*)*sptr, M, (void*)*valS);
                if( *sptr < *valS ) {
                    L = M;
                } else if( *sptr == *valS ) {
                    L = M;
                    break;
                } else {
                    R = M = M>0?M - 1:0;
                }
            } else {
                //LLAMA_LOG_INFO("%s: compare q %p vs %p\n", __func__, (void*)*qptr, (void*)*valQ);
                if( *qptr < *valQ ) {
                    L = M;
                } else if( *qptr == *valQ ) {
                    L = M;
                    break;
                } else {
                    R = M = M>0?M - 1:0;
                }
            }
        }
        if( M == 0 ) {
            mid = data[searchby]->head;
            miditem=0;
            listno=0;
        } else if( M == count ) {
            miditem++;
            if( miditem == units ) {
                miditem=0;
                listno++;
            }
            return NULL;
        } else {
            mid = data[searchby]->get( M, miditem, listno, searchby );
        }
        buf = (void**)mid->data;
        midref = (sp_searchable*)( buf[miditem] );
        //LLAMA_LOG_INFO("%s: M=%zu, miditem=%u, listno=%u\n", __func__, M, miditem, listno);
        return midref;
    }
    */
    sp_searchable *bsub( uint8_t searchby, T* ref, size_t &M, uint16_t &miditem, uint16_t &listno )
    {
        S *valS = (S*)(char**)( ((uint8_t*)ref + dataptr[0] ) );
        Q *valQ = (Q*)(char**)( ((uint8_t*)ref + dataptr[1] ) );
        //LLAMA_LOG_INFO("search: %zu %u for %p/%p\n", count, dataptr[searchby], *valS, *valQ);
        sp_linked *mid;
        size_t L,R;
        void **buf;
        S *sptr;
        Q *qptr;
        sp_searchable * midref;
        LLAMA_TRACK("sp_search_ub", 0);

        L=M=0;
        R=count>0?count-1:0;
        miditem=0;
        listno=0;
        while(L<R) {
            M=L+floor((float)(R-L)/2.0);
            mid = data[searchby]->get(M,miditem,listno,searchby);
            buf = (void**)mid->data;
            midref = (sp_searchable*)( buf[miditem] );
            if( midref == NULL ) {
                LLAMA_LOG_INFO("%s error: midref==null M=%zu L=%zu R=%zu\n", __func__, M, L, R);
                throw "error\n";
            }
            if( searchby == 0 )
                sptr = (S*)(char**)( (uint8_t*)midref->data + dataptr[0] );
            else
                qptr = (Q*)(char**)( (uint8_t*)midref->data + dataptr[1] );

            if( searchby == 0 ) {
                //LLAMA_LOG_INFO("%s: compare s %p(%zu) vs %p\n", __func__, (void*)*sptr, M, (void*)*valS);
                if( *sptr < *valS ) {
                    L = M = M+1;
                } else if( *sptr == *valS ) {
                    L = M;
                    break;
                } else {
                    R = M;
                }
            } else {
                //LLAMA_LOG_INFO("%s: compare q %p vs %p\n", __func__, (void*)*qptr, (void*)*valQ);
                if( *qptr < *valQ ) {
                    L = M = M+1;
                } else if( *qptr == *valQ ) {
                    L = M;
                    break;
                } else {
                    R = M;
                }
            }
        }
        if( M == count ) {
            miditem++;
            if( miditem == units ) {
                miditem=0;
                listno++;
            }
            LLAMA_TRACK("sp_search_ub", 6);
            return NULL;
        } else if( M == 0 ) {
            mid = data[searchby]->head;
            miditem=0;
            listno=0;
        } else {
            LLAMA_TRACK("sp_search_ub", 7);
            mid = data[searchby]->get( M, miditem, listno, searchby );
            LLAMA_TRACK("sp_search_ub", 8);
        }
        buf = (void**)mid->data;
        midref = (sp_searchable*)( buf[miditem] );
        //LLAMA_LOG_INFO("%s: M=%zu, miditem=%u, listno=%u\n", __func__, M, miditem, listno);
        LLAMA_TRACK("sp_search_ub", 9);
        return midref;
    }
    void insert( T *ptr )
    {
        void **buf;
        size_t M;
        uint16_t mi, ln;
        //LLAMA_LOG_INFO("pool insert count=%zu: %p\n", count, ptr);
        sp_searchable *sb = new sp_searchable(ptr);
        sp_searchable *res;

        LLAMA_TRACK("sp_insert", 0);
        res = bsub(0, ptr, M, mi, ln);
        sp_linked *arr = data[0]->get(ln);
        if( M != count ) {
            shift_pointers_r(arr, mi, 0);
        } else {
            arr->count++;
        }
        LLAMA_TRACK("sp_insert", 10);
        buf = (void**)arr->data;
        buf[mi] = (void*)sb;
        sb->inds[0] = ln;
        sb->refs[0] = mi;

        LLAMA_TRACK("sp_insert", 20);
        res = bsub(1, ptr, M, mi, ln);
        arr = data[1]->get(ln);
        if( M != count ) {
            shift_pointers_r(arr, mi, 1);
        } else {
            arr->count++;
        }
        LLAMA_TRACK("sp_insert", 40);
        buf = (void**)arr->data;
        buf[mi] = (void*)sb;
        sb->inds[1] = ln;
        sb->refs[1] = mi;

        count++;
        LLAMA_TRACK("sp_insert", 9);
    }
    void erase( T *ptr )
    { //! todo: reduce gaps after erasure to increase speed
        uint16_t mi, ln;
        size_t M;
        void **buf;
        sp_searchable *sp;
        sp_linked *arr;

        LLAMA_TRACK("sp_erase", 0);
        //LLAMA_LOG_INFO("%s: count=%zu\n", __func__, count);
        sp_searchable *pre = bs(0, ptr, M, mi, ln);
        if( !pre ) {
            LLAMA_LOG_ERROR("%s: not found 0 %p\n", __func__, ptr);
            PrintStackTrace();
            throw "not found on erase";
            return;
        }
        LLAMA_TRACK("sp_erase", 1);
        arr = data[0]->get(ln);
        buf = (void**)arr->data;
        sp = (sp_searchable*)buf[mi];
        LLAMA_TRACK("sp_erase", 2);
        if( !sp ) {
            LLAMA_LOG_INFO("%s: not found 2 %p (found %p)\n", __func__, ptr, sp->data);
            PrintStackTrace();
            throw "not found on erase";
            return;
        }
        if( sp->data != (void*)ptr ) {
            LLAMA_LOG_INFO("%s: not found 1 %p (found %p)\n", __func__, ptr, sp->data);
            PrintStackTrace();
            throw "not found on erase";
            return;
        }
        shift_pointers_l(arr, mi, 0);

        LLAMA_TRACK("sp_erase", 3);
        arr = data[1]->get( sp->inds[1] );
        buf = (void**)arr->data;
        if( buf[sp->refs[1]] != (void*)sp ) {
            LLAMA_LOG_INFO("%s: mismatch %p vs %p\n", __func__, sp, buf[sp->refs[1]]);
            PrintStackTrace();
            throw "not found on erase";
            return;
        }
        LLAMA_TRACK("sp_erase", 4);
        shift_pointers_l(arr, sp->refs[1], 1);

        count--;
        LLAMA_TRACK("sp_erase", 5);
    }
};

void report_pool_bysize(void*);
void report_pool_contig(void*);

struct senmemory_pool {
    Sp_searchnest<Llama_mem, void*, size_t> *used;
    Sp_searchnest<Llama_mem, void*, size_t> *loose; // for allocation
    size_t total_alloced=0;

    senmemory_pool() {
        Llama_mem *_0=NULL;
        used = new Sp_searchnest<Llama_mem, void*, size_t>((uint16_t)512,(uint16_t)((char*)&(_0->addr) - (char*)_0), (uint16_t)((char*)&(_0->sz) - (char*)_0));
        loose = new Sp_searchnest<Llama_mem, void*, size_t>((uint16_t)512,(uint16_t)((char*)&(_0->addr) - (char*)_0), (uint16_t)((char*)&(_0->sz) - (char*)_0));
    }
    void _report(void)
    {
        void *last=NULL;
        size_t lastsz=0;
        LLAMA_LOG_INFO("loose has %zu entries.\n", loose->count);
        report_pool_contig(NULL);
        report_pool_bysize(NULL);
        loose->data[0]->forall( report_pool_contig );
        loose->data[1]->forall( report_pool_bysize );
    }
    void _record(Llama_mem *mem)
    {
        //LLAMA_LOG_INFO("_record large memory: %p/%zu refc=%u\n", mem->addr, mem->sz, *mem->refc);
        loose->insert(mem);
        if( loose->count > 200 && loose->count%100 == 0 )
            LLAMA_LOG_INFO("%s: loose count %zu\n", __func__, loose->count);
    }
    void *alloc(size_t sz) {
        // shortcut:
        if( sz > 10240 || sz < 128 )
            return malloc(sz);

        Llama_mem ptr_ref, *new_ptr=NULL;
        Llama_mem *remnant=NULL, *usable=NULL, *usable2;
        void *ptr;
        sp_searchable *sp=NULL, *sp2=NULL;
        sp_linked *iter=NULL;
        size_t M;
        uint16_t offset, listno;
        void **buf;

        //llama_enable_tracker("sp_search");
        //llama_enable_tracker("sp_search_ub");
        //llama_enable_tracker("sp_free");
        //llama_enable_tracker("sp_erase");
        //llama_enable_tracker("sp_alloc");
        //llama_enable_tracker("sp_insert");

        LLAMA_TRACK("sp_alloc", 0);

        ptr_ref.sz = sz;
        //LLAMA_LOG_INFO("%s: locate size %zu\n", __func__, sz);
        sp = loose->bsub(1, &ptr_ref, M, offset, listno);
        if( sp ) {
            if( M == 0 ) {
                iter = loose->data[1]->head;
                offset = 0;
            } else {
                iter = loose->data[1]->get(listno);
            }
        } else {
            iter = NULL;
        }
        LLAMA_TRACK("sp_alloc", 1);
        /*
        m = std::lower_bound(loose_bysize.begin(), loose_bysize.end(), &ptr_ref, [](Llama_mem *a, Llama_mem *b) {
            return a->sz < b->sz;
        });*/
        if( iter ) { // && (((void*)iter->data)[offset]) ) {
            buf = (void**)iter->data;
            if( buf[offset] != NULL ) {
                LLAMA_TRACK("sp_alloc", 11);
                //LLAMA_LOG_INFO("%s: iter=%p sz=%zu offset=%u\n", __func__, iter, sz, offset);
                buf = (void**)iter->data;
                sp = (sp_searchable*)buf[offset];
                usable = (Llama_mem*)sp->data;
                LLAMA_TRACK("sp_alloc", 12);
                //LLAMA_LOG_INFO("%s: usable addr=%p sz=%zu\n", __func__, usable->addr, usable->sz);
                if( usable->sz == sz || usable->sz >= sz+144 ) {
                } else {
                    size_t tgtsz = sz+144;
                    ptr_ref.sz = tgtsz;

                    //LLAMA_LOG_INFO("%s: locate double %zu\n", __func__, ptr_ref.sz);
                    LLAMA_TRACK("sp_alloc", 13);
                    sp = loose->bsub(1, &ptr_ref, M, offset, listno);
                    LLAMA_TRACK("sp_alloc", 14);
                    if( sp ) {
                        iter = loose->data[1]->get( listno );
                        buf = (void**)iter->data;
                        if( buf[offset] != NULL ) {
                            sp = (sp_searchable*)buf[offset];
                            usable = (Llama_mem*)sp->data;
                            LLAMA_TRACK("sp_alloc", 15);
                            if( usable->sz < tgtsz || usable->sz >= sz*3 ) {
                                iter = NULL;
                            }
                            LLAMA_TRACK("sp_alloc", 16);
                        }
                    } else {
                        iter = NULL;
                    }
                    usable=NULL;
                    if( iter ) {
                        LLAMA_TRACK("sp_alloc", 17);
                        if( buf[offset] != NULL ) {
                            //LLAMA_LOG_INFO("%s: found double iter=%p offset=%u\n", __func__, iter, offset);
                            sp = (sp_searchable*)buf[offset];
                            usable = (Llama_mem*)sp->data;
                            //LLAMA_LOG_INFO("use %p:%zu\n", usable->addr, usable->sz);
                        } else {
                            usable = NULL;
                        }
                        LLAMA_TRACK("sp_alloc", 18);
                    }
                }
            }
        }
        LLAMA_TRACK("sp_alloc", 2);
        if( usable ) {
            if( usable->sz < sz ) {
            } else {
                //LLAMA_LOG_INFO("%s: erase usable %p (%p: %zu)\n", __func__, usable, usable->addr, usable->sz);
                loose->erase(usable); // we have to erase because we may change its size

                //LLAMA_TRACK("sp_alloc", 21);
                if( usable->sz != sz ) {
                    size_t addr_ptr, new_sz=usable->sz - sz;
                    addr_ptr = (size_t)( (char*)usable->addr + sz );
                    uint8_t offset_sz = 16 - ( addr_ptr%16 );
                    if( (addr_ptr%16) != 0 && new_sz > offset_sz ) {
                        addr_ptr += offset_sz;
                        new_sz = usable->sz - (((char*)addr_ptr) - ((char*)usable->addr));
                    }
                    //LLAMA_TRACK("sp_alloc", 22);
                    if( new_sz >= 16 ) {
                        //LLAMA_LOG_INFO("%s: create remnant size %zu ptr %p\n", __func__, new_sz, addr_ptr);
                        remnant = new Llama_mem(usable);
                        remnant->sz = new_sz;
                        remnant->addr = (void*)( addr_ptr );
                        _record(remnant);
                        usable->sz -= new_sz;
                    }
                    //LLAMA_TRACK("sp_alloc", 23);
                }

                new_ptr = usable;
                //LLAMA_TRACK("sp_alloc", 24);
            }
        }
        if( !new_ptr ) {
            //LLAMA_TRACK("sp_alloc", 3);
            new_ptr = new Llama_mem(calloc(sz,1));
            if( !new_ptr->addr ) {
                LLAMA_LOG_INFO("%s: allocation of %zu bytes failed.\n", __func__, sz);
                throw "allocation error\n";
            }
            new_ptr->sz = sz;
            memset(new_ptr->addr, 0, sz);
            total_alloced += sz/100;
            //LLAMA_TRACK("sp_alloc", 4);
        }

        //llama_enable_tracker("sp_search_ub");
        //llama_enable_tracker("sp_search");

        used->insert( new_ptr );

        //llama_disable_tracker("sp_search_ub");
        //llama_disable_tracker("sp_search");
        //LLAMA_TRACK("sp_alloc", 5);
        /*if( total_alloced > 100000 && (total_alloced/100)%100 == 0 ) {
            LLAMA_LOG_INFO("%s: total_alloced=%zu*100\n", __func__, total_alloced);
        }
        if( used->count > 100 && used->count%100 == 0 )
            LLAMA_LOG_INFO("%s: used count %zu %p (%p) size %zu\n", __func__, used->count, new_ptr, new_ptr->addr, sz);

        */
        LLAMA_TRACK("sp_alloc", 9);
        return new_ptr->addr;
    }
    void release(void *ptr) {
        // shortcut:
        /*
        free(ptr);
        return;
        */

        Llama_mem ptr_ref;
        sp_searchable *sp;
        sp_linked *iter;
        size_t M;
        uint16_t offset, listno;
        void **buf;

        LLAMA_TRACK("sp_free", 0);
        //LLAMA_LOG_INFO("%s: %p\n", __func__, ptr);
        ptr_ref.addr = ptr;
        sp = used->bs(0, &ptr_ref, M, offset, listno);
        iter = used->data[0]->get(listno);
        buf = (void**)iter->data;
        if( !iter || !buf[offset] ) {
            LLAMA_TRACK("sp_free", 2);
            free(ptr);
            LLAMA_TRACK("sp_free", 3);
            return;
            /*
            LLAMA_LOG_ERROR("Invalid free of %p, 1 used size %zu\n", ptr, used->count);
            PrintStackTrace();
            throw "Error";
            return;*/
        }
        Llama_mem *mem = (Llama_mem*)sp->data;
        if( mem->addr != ptr ) {
            LLAMA_TRACK("sp_free", 5);
            free(ptr);
            LLAMA_TRACK("sp_free", 6);
            return;
            /*
            LLAMA_LOG_ERROR("Invalid free of %p, 2 used size %zu (found %p at %zu (%u))\n",
                            ptr, used->count, mem->addr, M, offset);
            PrintStackTrace();
            throw "Error";
            return;
            */
        }
        memset( mem->addr, 0, mem->sz );
        LLAMA_TRACK("sp_free", 7);
        used->erase(mem);
        //LLAMA_LOG_INFO("%s: ->record\n", __func__);
        LLAMA_TRACK("sp_free", 8);
        _record(mem);
        LLAMA_TRACK("sp_free", 9);
    }
};

void report_pool_bysize( void *ptr )
{
    static Llama_mem *lastptr=NULL;

    if( ptr ) {
        Llama_mem *it = (Llama_mem*)ptr;
        LLAMA_LOG_INFO("sz: %p +%zu\n", it->addr, it->sz );
        if( it->sz < lastptr->sz ) {
            LLAMA_LOG_INFO("[error: out of order]\n");
        }
    }
    lastptr = (Llama_mem*)ptr;
}

void report_pool_contig( void *ptr )
{
    static Llama_mem *lastptr=NULL;

    if( ptr ) {
        Llama_mem *it = (Llama_mem*)ptr;
        LLAMA_LOG_INFO("alloc: %p +%zu\n", it->addr, it->sz );
        if( it->addr <= lastptr->addr && lastptr ) {
            LLAMA_LOG_INFO("[error: out of order]\n");
        }
    }
    lastptr = (Llama_mem*)ptr;
}


_Pool myPool;
bool pool_initialized=false;
void *pool_alloc(size_t sz)
{
    if( !pool_initialized ) {
        pool_initialized=true;
        new (&myPool) _Pool;
    }
    void *p = myPool.alloc(sz);
    if( !p ) {
        LLAMA_LOG_INFO("Invalid pointer 2\n");
    }
    return p;
}
void my_pool_free(void *ptr)
{
    if( !ptr ) {
        LLAMA_LOG_INFO("Invalid pointer 3\n");
    }
    myPool.release(ptr);
}
#define pool_free(x) my_pool_free(x); x=NULL


struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        //LLAMA_LOG_INFO("fopen('%s','%s')\n", fname, mode);
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            LLAMA_LOG_INFO("%s: cannot find '%s'\n", __func__, fname);
            return;
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) const {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    std::string read_string() const {
        uint32_t len;
        char *buf;

        read_raw(&len, sizeof(len));
        buf = (char*)pool_alloc(len+1);
        read_raw(buf, len);
        buf[len] = '\0';
        std::string ret = buf;
        pool_free(buf);
        buf=NULL;
        return ret;
    }
    void write_string(std::string buf) const {
        uint32_t len = buf.length();
        write_raw(&len, sizeof(len));
        write_raw(buf.c_str(), len);
    }
    void write_text(std::string buf) const {
        uint32_t len = buf.length();
        write_raw(buf.c_str(), len);
    }

    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, 1, len, fp);
        if (ferror(fp)) {
            LLAMA_LOG_INFO("%s read error: %s", __func__, strerror(errno));
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != len) {
            LLAMA_LOG_INFO("%s read error (EOF) read %zu of %zu", __func__, ret, len);
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }
    uint16_t read_u16() const {
        uint16_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, 1, len, fp);
        if (ret != len) {
            fprintf(stderr,"Write error: %d\n",ferror(fp));
            clearerr(fp);
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }

        if (fflush(fp)) {
          fprintf(stderr,"Flush error: %d\n",ferror(fp));
          clearerr(fp);
          throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }
    void write_u16(std::uint16_t val) const {
        write_raw(&val, sizeof(val));
    }

    void close() {
        if( fp ) {
            if (fflush(fp)) {
              fprintf(stderr,"Flush error: %d\n",ferror(fp));
              clearerr(fp);
              throw std::runtime_error(format("write error: %s", strerror(errno)));
            }
            std::fclose(fp);
            fp=NULL;
        }
    }

    ~llama_file() {
        if (fp) {
            if (fflush(fp)) {
              fprintf(stderr,"Flush error: %d\n",ferror(fp));
              clearerr(fp);
            }
            std::fclose(fp);
        }
    }
};

std::set<std::string> trackers;

void LLAMA_TRACK(std::string tag, size_t value)
{
    if( trackers.count(tag) == 0 ) return;
    llama_file *f = new llama_file(tag.c_str(), "w");
    f->write_text( std::to_string(value) );
    f->close();
}
void llama_enable_tracker(std::string tag)
{
    trackers.insert(tag);
}
void llama_disable_tracker(std::string tag)
{
    trackers.erase(tag);
}


struct llama_mmap {
    void * addr;
    size_t size;

    llama_mmap(const llama_mmap &) = delete;

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;

    // list of mapped fragments (first_offset, last_offset)
    std::vector<std::pair<size_t, size_t>> mapped_fragments;

    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1 /* -1 = max value */, bool numa = false) {
        size = file->size;
        int fd = fileno(file->fp);
        int flags = MAP_SHARED;
        // prefetch/readahead impairs performance on NUMA systems
        if (numa)  { prefetch = 0; }
#ifdef __linux__
        // advise the kernel to read the file sequentially (increases readahead)
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size, PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) { // NOLINT
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            // advise the kernel to preload the mapped memory
            if (posix_madvise(addr, std::min(file->size, prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            // advise the kernel not to use readahead
            // (because the next page might not belong on the same node)
            if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }

        // initialize list of mapped_fragments
        mapped_fragments.emplace_back(0, file->size);
    }

    static void align_range(size_t * first, size_t * last, size_t page_size) {
        // align first to the next page
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        // align last to the previous page
        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    // partially unmap the file in the range [first, last)
    void unmap_fragment(size_t first, size_t last) {
        // note: this function must not be called multiple times with overlapping ranges
        // otherwise, there is a risk of invalidating addresses that have been repurposed for other mappings
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void * next_page_start = (uint8_t *) addr + first;

        // unmap the range
        if (munmap(next_page_start, len)) {
            LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        // update the list of mapped fragments to avoid unmapping the same range again in the destructor
        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto & frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                // the range is in the middle of the fragment, split it
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                // the range starts in the middle of the fragment
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                // the range ends in the middle of the fragment
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
                // the range covers the entire fragment
            } else {
                // the range is outside the fragment
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~llama_mmap() {
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false) {
        GGML_UNUSED(numa);

        size = file->size;

        HANDLE hFile = (HANDLE) _get_osfhandle(_fileno(file->fp));

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(error).c_str()));
        }

        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            // PrefetchVirtualMemory is only present on Windows 8 and above, so we dynamically load it
            BOOL (WINAPI *pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            // may fail on pre-Windows 8 systems
            pPrefetchVirtualMemory = reinterpret_cast<decltype(pPrefetchVirtualMemory)> (GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

            if (pPrefetchVirtualMemory) {
                // advise the kernel to preload the mapped memory
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T) std::min(size, prefetch);
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    LLAMA_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
#else
            throw std::runtime_error("PrefetchVirtualMemory unavailable");
#endif
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        // not supported
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    ~llama_mmap() {
        if (!UnmapViewOfFile(addr)) {
            LLAMA_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    llama_mmap(struct llama_file * file, size_t prefetch = -1, bool numa = false) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);
        GGML_UNUSED(numa);

        throw std::runtime_error("mmap not supported");
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }
#endif
};

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct llama_mlock {
    void * addr = NULL;
    size_t size = 0;

    bool failed_already = false;

    llama_mlock() {}
    llama_mlock(const llama_mlock &) = delete;

    ~llama_mlock() {
        if (size) {
            raw_unlock(addr, size);
        }
    }

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0); // NOLINT
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

#ifdef _POSIX_MEMLOCK_RANGE
    static constexpr bool SUPPORTED = true;

    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

    #ifdef __APPLE__
        #define MLOCK_SUGGESTION \
            "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
            "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
    #else
        #define MLOCK_SUGGESTION \
            "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
    #endif

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);

        // Check if the resource limit is fine after all
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
            suggest = false;
        }

        LLAMA_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    #undef MLOCK_SUGGESTION

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            LLAMA_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                LLAMA_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            // It failed but this was only the first try; increase the working
            // set size and try again.
            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                LLAMA_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            // Per MSDN: "The maximum number of pages that a process can lock
            // is equal to the number of pages in its minimum working set minus
            // a small overhead."
            // Hopefully a megabyte is enough overhead:
            size_t increment = len + 1048576;
            // The minimum must be <= the maximum, so we need to increase both:
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                LLAMA_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            LLAMA_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    static size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) const {
        LLAMA_LOG_WARN("warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void * addr, size_t len) {}
#endif
};

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    }
    else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

static ggml_backend_buffer_type_t llama_default_buffer_type_cpu(bool host_buffer) {
    ggml_backend_buffer_type_t buft = nullptr;

#if defined(GGML_USE_CUBLAS)
    // host buffers should only be used when data is expected to be copied to/from the GPU
    if (host_buffer) {
        buft = ggml_backend_cuda_host_buffer_type();
    }
#elif defined(GGML_USE_SYCL)
    if (host_buffer) {
        buft = ggml_backend_sycl_host_buffer_type();
    }
#elif defined(GGML_USE_CPU_HBM)
    buft = ggml_backend_cpu_hbm_buffer_type();
#elif defined(GGML_USE_VULKAN)
    if (host_buffer) {
        buft = ggml_backend_vk_host_buffer_type();
    }
#endif

    if (buft == nullptr) {
        buft = ggml_backend_cpu_buffer_type();
    }
    return buft;

    GGML_UNUSED(host_buffer);
}

static ggml_backend_buffer_type_t llama_default_buffer_type_offload(int gpu) {
    ggml_backend_buffer_type_t buft = nullptr;

#ifdef GGML_USE_METAL
    buft = ggml_backend_metal_buffer_type();
#elif defined(GGML_USE_CUBLAS)
    buft = ggml_backend_cuda_buffer_type(gpu);
#elif defined(GGML_USE_VULKAN)
    buft = ggml_backend_vk_buffer_type(gpu);
#elif defined(GGML_USE_SYCL)
    buft = ggml_backend_sycl_buffer_type(gpu);
#elif defined(GGML_USE_CLBLAST)
    buft = ggml_backend_opencl_buffer_type();
#elif defined(GGML_USE_KOMPUTE)
    buft = ggml_backend_kompute_buffer_type(gpu);
    if (buft == nullptr) {
        LLAMA_LOG_WARN("%s: cannot use GPU %d, check `vulkaninfo --summary`\n", __func__, gpu);
    }
#endif

    if (buft == nullptr) {
        buft = llama_default_buffer_type_cpu(true);
    }
    return buft;

    GGML_UNUSED(gpu);
}

static ggml_backend_buffer_type_t llama_default_buffer_type_split(int fallback_gpu, const float * tensor_split) {
    ggml_backend_buffer_type_t buft = nullptr;

#ifdef GGML_USE_CUBLAS
    if (ggml_backend_cuda_get_device_count() > 1) {
        buft = ggml_backend_cuda_split_buffer_type(tensor_split);
    }
#endif

#ifdef GGML_USE_SYCL
    if (ggml_backend_sycl_get_device_count() > 1) {
        buft = ggml_backend_sycl_split_buffer_type(tensor_split);
    }
#endif

    if (buft == nullptr) {
        buft = llama_default_buffer_type_offload(fallback_gpu);
    }
    return buft;

    GGML_UNUSED(tensor_split);
}

static size_t llama_get_device_count() {
#if defined(GGML_USE_CUBLAS)
    return ggml_backend_cuda_get_device_count();
#elif defined(GGML_USE_SYCL)
    return ggml_backend_sycl_get_device_count();
#elif defined(GGML_USE_VULKAN)
    return ggml_backend_vk_get_device_count();
#else
    return 1;
#endif
}

static size_t llama_get_device_memory(int device) {
#if defined(GGML_USE_CUBLAS)
    size_t total;
    size_t free;
    ggml_backend_cuda_get_device_memory(device, &total, &free);
    return free;
#elif defined(GGML_USE_SYCL)
    size_t total;
    size_t free;
    ggml_backend_sycl_get_device_memory(device, &total, &free);
    return free;
#elif defined(GGML_USE_VULKAN)
    size_t total;
    size_t free;
    ggml_backend_vk_get_device_memory(device, &total, &free);
    return free;
#else
    return 1;
    GGML_UNUSED(device);
#endif
}

//
// globals
//

struct llama_state {
    llama_state() {
#ifdef GGML_USE_METAL
        ggml_backend_metal_log_set_callback(log_callback, log_callback_user_data);
#endif
    }

    // We save the log callback globally
    ggml_log_callback log_callback = llama_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static llama_state g_state;

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_17M,
    MODEL_22M,
    MODEL_33M,
    MODEL_109M,
    MODEL_137M,
    MODEL_335M,
    MODEL_0_5B,
    MODEL_1B,
    MODEL_2B,
    MODEL_3B,
    MODEL_4B,
    MODEL_7B,
    MODEL_8B,
    MODEL_13B,
    MODEL_14B,
    MODEL_15B,
    MODEL_20B,
    MODEL_30B,
    MODEL_34B,
    MODEL_40B,
    MODEL_65B,
    MODEL_70B,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
    MODEL_XL,
};

static const size_t kiB = 1024;
static const size_t MiB = 1024*kiB;
static const size_t GiB = 1024*MiB;

struct llama_hparams {
    bool vocab_only;
    bool rope_finetuned;

    uint32_t n_vocab;
    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_layer;
    uint32_t n_rot;
    uint32_t n_embd_head_k; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v; // dimension of values (d_v) aka n_embd_head
    uint32_t n_ff;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_vocab_type = 0; // for BERT-style token types

    float f_norm_eps;
    float f_norm_rms_eps;

    float    rope_freq_base_train;
    float    rope_freq_scale_train;
    uint32_t n_yarn_orig_ctx;

    float f_clamp_kqv      = 0.0f;
    float f_max_alibi_bias = 0.0f;

    bool causal_attn = true;
    bool need_kq_pos = false;

    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    bool operator!=(const llama_hparams & other) const {
        if (this->vocab_only    != other.vocab_only)    return true;
        if (this->n_vocab       != other.n_vocab)       return true;
        if (this->n_ctx_train   != other.n_ctx_train)   return true;
        if (this->n_embd        != other.n_embd)        return true;
        if (this->n_head        != other.n_head)        return true;
        if (this->n_head_kv     != other.n_head_kv)     return true;
        if (this->n_layer       != other.n_layer)       return true;
        if (this->n_rot         != other.n_rot)         return true;
        if (this->n_embd_head_k != other.n_embd_head_k) return true;
        if (this->n_embd_head_v != other.n_embd_head_v) return true;
        if (this->n_ff          != other.n_ff)          return true;
        if (this->n_expert      != other.n_expert)      return true;
        if (this->n_expert_used != other.n_expert_used) return true;

        if (this->rope_finetuned  != other.rope_finetuned)  return true;
        if (this->n_yarn_orig_ctx != other.n_yarn_orig_ctx) return true;

        const float EPSILON = 1e-9f;

        if (!is_float_close(this->f_norm_eps,            other.f_norm_eps,            EPSILON)) return true;
        if (!is_float_close(this->f_norm_rms_eps,        other.f_norm_rms_eps,        EPSILON)) return true;
        if (!is_float_close(this->rope_freq_base_train,  other.rope_freq_base_train,  EPSILON)) return true;
        if (!is_float_close(this->rope_freq_scale_train, other.rope_freq_scale_train, EPSILON)) return true;

        return false;
    }

    uint32_t n_gqa() const {
        return n_head/n_head_kv;
    }

    uint32_t n_embd_k_gqa() const { // dimension of key embeddings across all k-v heads
        return n_embd_head_k * n_head_kv;
    }

    uint32_t n_embd_v_gqa() const { // dimension of value embeddings across all k-v heads
        return n_embd_head_v * n_head_kv;
    }
};

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_batch;
    uint32_t n_threads;       // number of threads to use for generation
    uint32_t n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_yarn_orig_ctx;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float defrag_thold;

    bool embeddings;
    bool offload_kqv;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * attn_norm_b;
    struct ggml_tensor * attn_norm_2;
    struct ggml_tensor * attn_norm_2_b;
    struct ggml_tensor * attn_q_norm;
    struct ggml_tensor * attn_q_norm_b;
    struct ggml_tensor * attn_k_norm;
    struct ggml_tensor * attn_k_norm_b;
    struct ggml_tensor * attn_out_norm;
    struct ggml_tensor * attn_out_norm_b;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;
    struct ggml_tensor * wqkv;

    // attention bias
    struct ggml_tensor * bq;
    struct ggml_tensor * bk;
    struct ggml_tensor * bv;
    struct ggml_tensor * bo;
    struct ggml_tensor * bqkv;

    // normalization
    struct ggml_tensor * ffn_norm;
    struct ggml_tensor * ffn_norm_b;
    struct ggml_tensor * layer_out_norm;
    struct ggml_tensor * layer_out_norm_b;

    // ff
    struct ggml_tensor * ffn_gate; // w1
    struct ggml_tensor * ffn_down; // w2
    struct ggml_tensor * ffn_up;   // w3

    // ff MoE
    struct ggml_tensor * ffn_gate_inp;
    struct ggml_tensor * ffn_gate_exp[LLAMA_MAX_EXPERTS];
    struct ggml_tensor * ffn_down_exp[LLAMA_MAX_EXPERTS];
    struct ggml_tensor * ffn_up_exp  [LLAMA_MAX_EXPERTS];

    // ff bias
    struct ggml_tensor * ffn_down_b; // b2
    struct ggml_tensor * ffn_up_b;   // b3
    struct ggml_tensor * ffn_act;
};


using llm_build_cb = std::function<void(struct ggml_tensor * cur, const char * name, int nl)>;

typedef struct system_actor System_actor;
typedef struct system_memory System_memory;

typedef struct llm_org_context Org_context;

ggml_backend_t get_backend(ggml_tensor *);

// ring-buffer of cached KV data
typedef struct llama_kv_cache {
    uint32_t size = 0;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    uint16_t seq=0;
    /*
    uint8_t inuse[512];
    bool kv_in_use( int token ) {
        int spot = token/8;
        uint8_t a = inuse[spot];
        int remn = token - spot*8;
        return( ( ( a >> remn ) & 1 ) == 0x1 );
    }
    */

    std::vector<struct ggml_tensor *> k_l; // per layer
    std::vector<struct ggml_tensor *> v_l;

    struct kv_data {
        size_t start;
        size_t len;
        size_t alloced;
        void *ptr;
    };

    std::vector<kv_data> pre_k[32];
    std::vector<kv_data> pre_v[32];
    void prefit_clear( void )
    {
        std::vector<kv_data>::iterator it;
        for( int il=0; il<32; il++ ) {
            for( it = pre_k[il].begin(); it != pre_k[il].end(); it++ ) {
                if( (*it).alloced > 0 ) {
                    pool_free( (*it).ptr );
                }
            }
            for( it = pre_v[il].begin(); it != pre_v[il].end(); it++ ) {
                if( (*it).alloced > 0 ) {
                    pool_free( (*it).ptr );
                }
            }
            pre_k[il].clear();
            pre_v[il].clear();
        }
    }
    void prefit_set( std::vector<kv_data> *pre, void *data, size_t start, size_t len )
    {
        std::vector<kv_data>::iterator it;
        kv_data z;
        
        if( !data ) {
            LLAMA_LOG_INFO("Invalid pointer 1\n");
            throw "invalid pointer";
        }

        //LLAMA_LOG_INFO("prefit_set(%zu,%zu)\n", start, len);

        for( it = pre->begin(); it != pre->end(); it++ ) {
            /*if( (*it).start + (*it).len == start ) {
                if( (*it).len + len > (*it).alloced ) {
                    size_t newlen = (*it).len + len;
                    if( newlen < 256000 ) newlen *= 2;
                    else if( len < 1024 ) newlen += len*1024;
                    else newlen += len*16;

                    void *buf = pool_alloc( newlen );
                    memcpy( buf, (*it).ptr, (*it).len );
                    memcpy( (void*)( (char*)buf + (*it).len ), data, len );
                    if( (*it).alloced > 0 ) {
                        pool_free( (*it).ptr );
                    }
                    (*it).alloced = newlen;
                    (*it).ptr = buf;
                } else {
                    memcpy( (void*)( (char*)(*it).ptr + (*it).len ), data, len );
                }
                (*it).len += len;
                //LLAMA_LOG_INFO("prefit_set done 1\n", start, len);
                return;
            }*/
            if( (*it).start > start ) {
                z.start = start;
                z.len = len;
                z.ptr = data;
                z.alloced = 0;
                pre->insert(it, z);
                //LLAMA_LOG_INFO("prefit_set done 2\n", start, len);
                return;
            } else if( (*it).start + (*it).len > start ) {
                LLAMA_LOG_INFO("prefit overflow 2!\n");
                throw "prefit overthrow!\n";
            }
        }
        z.start = start;
        z.len = len;
        z.ptr = data;
        z.alloced = 0;
        pre->push_back(z);
        //LLAMA_LOG_INFO("prefit_set done 3\n", start, len);
    }
#define max_buflen 32767
    void prefit_write(void)
    {
        std::vector<kv_data>::iterator it, itprev, itfirst;
        size_t buflen, bufstart;
        uint8_t buf[max_buflen];

        LLAMA_LOG_INFO("prefit_write(%s): %zu k %zu v\n", quick_ts().c_str(), pre_k[0].size(), pre_v[0].size());
        for( int il=0; il<32; il++ ) {
            it = pre_k[il].begin();
            //LLAMA_LOG_INFO("layer %d %zu\n", il, it->len);
            if( it != pre_k[il].end() ) {
                itfirst = itprev = it;
                buflen = it->len;
            }
            while( true ) {
                while( it != pre_k[il].end() && buflen >= max_buflen ) {
                    ggml_backend_tensor_set(k_l[il], it->ptr, it->start, it->len );
                    it++;
                    itprev=itfirst=it;
                    buflen=it->len;
                }
                if( itfirst == pre_k[il].end() ) break;
                if( it == itprev && it != pre_k[il].end()  )
                    it++;
                while( it != pre_k[il].end() && buflen + it->len < max_buflen && itprev->start + itprev->len == it->start ) {
                    itprev = it;
                    buflen += it->len;
                    it++;
                }
                buflen=0;
                for( itprev=itfirst; itprev!=it; itprev++ ) {
                    memcpy(buf+buflen, itprev->ptr, itprev->len);
                    buflen += itprev->len;
                }
                ggml_backend_tensor_set(k_l[il], buf, itfirst->start, buflen );
                if( it == pre_k[il].end() ) break;
                itprev = itfirst = it;
                buflen = it->len;
            }
            //LLAMA_LOG_INFO("prefit_write(%s): %d step 2\n", quick_ts().c_str(), il);

            //LLAMA_LOG_INFO("layer %d v\n", il);
            it = pre_v[il].begin();
            if( it != pre_v[il].end() ) {
                itfirst = itprev = it;
                buflen = it->len;
            }
            while( true ) {
                while( it != pre_v[il].end() && buflen >= max_buflen ) {
                    ggml_backend_tensor_set(v_l[il], it->ptr, it->start, it->len );
                    it++;
                    itprev=itfirst=it;
                    buflen=it->len;
                }
                if( itfirst == pre_v[il].end() ) break;
                if( it == itprev && it != pre_v[il].end() )
                    it++;
                while( it != pre_v[il].end() && buflen + it->len < max_buflen && itprev->start + itprev->len == it->start ) {
                    itprev = it;
                    buflen += it->len;
                    it++;
                }
                buflen=0;

                for( itprev=itfirst; itprev!=it; itprev++ ) {
                    memcpy(buf+buflen, itprev->ptr, itprev->len);
                    buflen += itprev->len;
                }
                ggml_backend_tensor_set(v_l[il], buf, itfirst->start, buflen );
                if( it == pre_v[il].end() ) break;
                itprev = itfirst = it;
                buflen = it->len;
            }
            /*
            for( it = pre_v[il].begin(); it != pre_v[il].end(); it++ ) {
                ggml_backend_tensor_set(v_l[il], (*it).ptr, (*it).start, (*it).len );
            }*/
        }
        LLAMA_LOG_INFO("prefit_write(%s): done\n", quick_ts().c_str());
    }
    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;
    void prepare(void)
    {
        LLAMA_LOG_INFO("Prepare kv_cache\n");
        new (&(k_l)) std::vector<struct ggml_tensor*>;
        new (&(v_l)) std::vector<struct ggml_tensor *>;
        new (&ctxs) std::vector<struct ggml_context *>;
        new (&bufs) std::vector<ggml_backend_buffer_t>;
        for( int il=0; il<32; il++ ) {
            new (&(pre_k[il])) std::vector<kv_data>;
            new (&(pre_v[il])) std::vector<kv_data>;
        }
    }

    void *buffer( size_t len )
    {
        void *x = (void*)pool_alloc( 32 * 2048 * len );
        return x;
    }
    void read( size_t startpt, size_t n_tokens, void *kx, void *vx )
    {
        size_t bufptr;

        size_t v = 2 * n_tokens;
        size_t k = 1024 * v;

        size_t st_v = 2 * startpt;
        size_t st_k = 1024 * st_v;

        size_t p, p_sz = 2 * size;

        LLAMA_LOG_INFO("kv_read(%s): startpt %zu n_tokens %zu\n", quick_ts().c_str(), startpt, n_tokens);

        ggml_backend_t backend_res = get_backend(k_l[0]);

        for( int il=bufptr=0; il<32; il++ ) {
            //LLAMA_LOG_INFO("kv_read(%s): %d\n", quick_ts().c_str(), il);
            ggml_backend_tensor_get_async(backend_res, k_l[il], (void*)((char*)kx + bufptr), st_k, k );
            for( int i=0, p=st_v; i<1024; i++, p += p_sz, bufptr += v ) {
                ggml_backend_tensor_get_async(backend_res, v_l[il], (void*)((char*)vx + bufptr), p, v );
            }
        }

        ggml_backend_synchronize(backend_res);
        LLAMA_LOG_INFO("kv_read(%s): done\n", quick_ts().c_str());
    }

    void read2( size_t startpt, size_t offset, size_t n_tokens, std::vector<ggml_fp16_t> kx[32], std::vector<ggml_fp16_t> vx[32] )
    {
        size_t bufptr;

        size_t v = 2 * n_tokens;
        size_t k = 1024 * v;

        size_t st_t = 2048 * startpt;
        size_t off_t = 2048 * offset;

        size_t p, p_sz = 2 * size;
        size_t st_v = startpt * 2;

        size_t endpt = startpt+n_tokens;

        LLAMA_LOG_INFO("kv_read2(%s): startpt %zu offset %zu n_tokens %zu\n", quick_ts().c_str(), startpt, offset, n_tokens);

        ggml_backend_t backend_res = get_backend(k_l[0]);
        // create a temporary buffer to hold the data before parsing it into the end of the vx lists
        void *vx_buffers = pool_alloc( 32 * 2048 * n_tokens );

        for( int il=0; il<32; il++, bufptr += k ) {
            bufptr=off_t;
            ggml_backend_tensor_get_async(backend_res, k_l[il], (void*)((char*)kx[il].data() + bufptr), st_t, k );

            for( int i=0, p=startpt*2; i<1024; i++, p += p_sz ) {
                ggml_backend_tensor_get_async(backend_res, v_l[il], (void*)((char*)vx_buffers + i*v + k*il), p, v );
            }
        }

        ggml_backend_synchronize(backend_res);

        LLAMA_LOG_INFO("kv_read2(%s): transfer\n", quick_ts().c_str());
        for( int il=0; il<32; il++ ) {

            size_t tx = k*il;
            for( int i=0, p=0; i<1024; i++, p += v ) {
                bufptr = i * 2;
                for( int t=startpt; t<endpt; t++, bufptr += 2048, tx += 2 ) {
                    *(ggml_fp16_t *)((char*)vx[il].data()+bufptr) = *(ggml_fp16_t *)((char*)vx_buffers + tx + p);
                    //memcpy(backend_res, v_l[il], (void*)((char*)vx[il].data() + bufptr), p, 2 );
                }
            }
        }
        LLAMA_LOG_INFO("kv_read2(%s): done\n", quick_ts().c_str());
    }

    /*
    void writestr( std::string who, std::string what, int n_tokens )
    {
        // write and evaluate token message
        std::string message = "<|im_start|>";
        message.append(who);
        message.append("\n");
        message.append(what);
        message.append("<|im_end|>");

        std::vector<int> tokens = new std::vector<int>();
        llama_tokenstr(current_model, message, tokens, false);
    }
    */

    void write( int startpt, size_t n_tokens, void *kx, void *vx )
    {
        size_t bufptr;

        size_t v = 2 * n_tokens;
        size_t k = 1024 * v;

        size_t st_v = 2 * startpt;
        size_t st_k = 1024 * st_v;

        size_t p, p_sz = size*2; // size refers to total # of tokens in this kb

        LLAMA_LOG_INFO("kv_write(%s): startpt %d n_tokens %zu\n", quick_ts().c_str(), startpt, n_tokens);
        for( int il=bufptr=0; il<32; il++ ) {
            //LLAMA_LOG_INFO("kv_write(%s): %d step 1\n", quick_ts().c_str(), il);
            //LLAMA_LOG_INFO("pre_k %d: %zu %zu\n", il, st_k, st_k+k);
            prefit_set( &(pre_k[il]), (void*)((char*)kx + bufptr), st_k, k );
            //ggml_backend_tensor_set( k_l[il], (void*)((char*)kx + bufptr), st_k, k );
            //LLAMA_LOG_INFO("kv_write(%s): %d step 2\n", quick_ts().c_str(), il);
            for( int i=0, p=st_v; i<1024; i++, p += p_sz, bufptr += v ) {
                //LLAMA_LOG_INFO("pre_v %d: %zu %zu\n", il, p, p+v);
                prefit_set( &(pre_v[il]), (void*)((char*)vx + bufptr), p, v );
                //ggml_backend_tensor_set( v_l[il], (void*)((char*)vx + bufptr), p, v );
            }
        }

        LLAMA_LOG_INFO("kv_write(%s): done\n", quick_ts().c_str());
    }

    void write2( size_t startpt, size_t offset, size_t n_tokens, std::vector<ggml_fp16_t> kx[32], std::vector<ggml_fp16_t> vx[32] )
    {
        size_t bufptr;

        size_t v = 2 * n_tokens;
        size_t k = 1024 * v;

        size_t st_t = 2048 * startpt;
        size_t off_t = 2048 * offset;

        size_t p, p_sz = 2 * size;
        size_t st_v = startpt * 2;

        size_t endpt = startpt+n_tokens;

        LLAMA_LOG_INFO("kv_write2: startpt %zu offset %zu n_tokens %zu\n", startpt, offset, n_tokens);

        ggml_backend_t backend_res = get_backend(k_l[0]);

        void *vx_buffers = pool_alloc(n_tokens*2048*32);
        bufptr=0;
        for( int il=0; il<32; il++ ) {
            for( int t=startpt; t<endpt; t++, st_v += 2 ) {
                for( int i=0, p=st_v; i<1024; i++, p += p_sz, bufptr += 2 ) {

                }
            }
        }
        for( int il=0; il<32; il++ ) {
            bufptr=off_t;
            ggml_backend_tensor_set_async(backend_res, k_l[il], (void*)((char*)kx[il].data() + bufptr), st_t, k );
            //! explore: it might be faster for short generations to write all of v over again instead of per-token per-head
            for( int t=startpt; t<endpt; t++, st_v += 2 ) {
                for( int i=0, p=st_v; i<1024; i++, p += p_sz, bufptr += 2 ) {
                    ggml_backend_tensor_set_async(backend_res, v_l[il], (void*)((char*)vx[il].data() + bufptr), p, 2 );
                }
            }
        }

        ggml_backend_synchronize(backend_res);
    }

    /*
    void use_tokens(size_t startpt, size_t n_tokens, bool used=true)
    {
        int end_pt = startpt+n_tokens;
        if( end_pt >= size ) {
            end_pt = size;
            n_tokens = size - startpt;
        }
        int spot = startpt/8;
        int endspot = end_pt/8;
        int remn = startpt - spot*8;
        int endremn = end_pt - endspot*8;

        if( n_tokens > 4 )
            LLAMA_LOG_INFO("%s: %s startpt %zu n_tokens %zu\n", __func__, used ? "use" : "blank", startpt, n_tokens);

        for( int i=spot; i<=endspot; i++ ) {
            if( i != spot && i != endspot ) {
                inuse[i] = used ? 0xff : 0;
            } else {
                int start_point = (i!=spot) ? 0 : ( remn );
                int stop_point = (i!=endspot) ? 8 : ( endremn );
                if( used ) {
                    for( int bit=start_point; bit<stop_point; bit++ ) {
                        inuse[i] |= ( 1 << (7-bit) );
                    }
                } else {
                    for( int bit=start_point; bit<stop_point; bit++ ) {
                        inuse[i] &= ~( 1 << (7-bit) );
                    }
                }
            }
        }
    }
    */

    size_t total_size() const {
        size_t size = 0;
        for (ggml_backend_buffer_t buf : bufs) {
            size += ggml_backend_buffer_get_size(buf);
        }
        return size;
    }

    ~llama_kv_cache() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }
} Kv_cache;

struct llm_org_context {
    Kv_cache *kv_self;
    std::vector<uint8_t> & buf_compute_meta;
    ggml_backend_sched *sched;

    struct ggml_context * ctx0 = nullptr;
    uint16_t n_embd_k_gqa, n_embd_v_gqa;

    llm_org_context( Kv_cache *kv,
                     std::vector<uint8_t> &compute_meta,
                     ggml_backend_sched *scheduler,
                     llama_hparams hparams
                     )
        :
        kv_self          (kv),
        buf_compute_meta (compute_meta),
        sched ( scheduler ),
        n_embd_k_gqa ( hparams.n_embd_head_k * hparams.n_head_kv ),
        n_embd_v_gqa ( hparams.n_embd_head_v * hparams.n_head_kv )
    {

    }

    void init() {
        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_compute_meta.size(),
            /*.mem_buffer =*/ buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };
        LLAMA_LOG_INFO("Prepare buf_compute_meta size %zu\n", buf_compute_meta.size());
        ctx0 = ggml_init(params);
    }

    void free() {
        if (ctx0) {
            ggml_free(ctx0);
            ctx0 = nullptr;
        }
    }

    struct ggml_cgraph *shuffler=NULL;

    int buffer_size=0;
    int buffer_target=0;
    void *k_buffer_layers[32];
    void *v_buffer_layers[32];

    ggml_backend_t backend_cpu = nullptr;
    int n_threads;

    struct ggml_cgraph *get_shuffler()
    {
        if( !shuffler )
            shuffler = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        return shuffler;
    }

    void run_kv_shuffler(void) {
        LLAMA_LOG_INFO("%s: run shuffler\n", __func__);

        backend_cpu = ggml_backend_cpu_init();
        if (backend_cpu != nullptr) {
            ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
        }

        ggml_cgraph *gf = get_shuffler();
        ggml_backend_sched_graph_compute(sched, gf);

        if( buffer_size > 0 ) {
            LLAMA_LOG_INFO("%s: adjust buffer %d\n", __func__, buffer_size);
            size_t size_v = buffer_size * 2;
            size_t size_k = size_v * 1024;
            size_t tgt_v = buffer_target * 2;
            size_t tgt_k = tgt_v * 1024;
            size_t p_size = kv_self->size * 2;
            ggml_backend_t backend_res = get_backend(kv_self->k_l[0]);
            for( int il = 0; il < 32; ++il ) { // it's the same amt of memory, 1024*overlap_v
                ggml_backend_tensor_set_async(backend_res, kv_self->k_l[il], k_buffer_layers[il], tgt_k, size_k );
                for( int i=0; i<1024; i++ ) {
                    ggml_backend_tensor_set_async(backend_res, kv_self->v_l[il], (void*)((char*)v_buffer_layers[il]+(size_v*i)), tgt_v+(p_size*i), size_v );
                }
                pool_free(k_buffer_layers[il]);
                pool_free(v_buffer_layers[il]);
            }
            ggml_backend_synchronize(backend_res);
            buffer_size = 0;
        }
        shuffler=NULL;
    }

    int swap_left_kv_now(int from_st, int from_en, int to_st, int used_start, int empty_start) {
        int rv = swap_left_kv(from_st, from_en, to_st, used_start, empty_start);
        run_kv_shuffler();
        return rv;
    }

    int swap_left_kv(int from_st, int from_en, int to_st, int used_start, int empty_start) {
        struct ggml_cgraph * gf = get_shuffler();

        if( from_st < to_st ) {
            LLAMA_LOG_ERROR("You shouldn't have done that!\n");
            return -1;
        }

        int empty_sz = kv_self->size - empty_start;
        int from_sz = from_en - from_st;
        int overlap = from_sz - ( used_start - to_st );
        if( overlap < 0 ) overlap = 0;
        int overlap_end = used_start + overlap;
        if( overlap_end > from_st ) {
            LLAMA_LOG_INFO("%s: overlap_end %d > from_st %d\n", __func__, overlap_end, from_st);
            overlap -= ( overlap_end - from_st );
            overlap_end = from_st;
        }
        size_t overlap_tgt = overlap_end + overlap; // something about this seems strange or weird or something...
        size_t used_en = from_st;
        size_t remnant = used_en - overlap_end;

        size_t overlap_v = 2 * overlap;
        size_t overlap_k = 1024 * overlap_v;

        size_t to_st_v = 2 * to_st;
        size_t to_st_k = 1024 * to_st_v;

        size_t p_size = kv_self->size * 2;

        // record overlapping area for replay at end of run
        buffer_size = overlap;
        if( overlap > 0 ) {
            buffer_target = to_st + from_sz;
            LLAMA_LOG_INFO("%s: set buffer target %d size %d\n", __func__, buffer_target, buffer_size);
        }

        void *nd;
        ggml_tensor *view_k_src, *view_k_dst, *view_v_src, *view_v_dst;

        LLAMA_LOG_INFO("%s: setting up swap_left from %d+%d to %d and from %d+%d to %d\n", from_st, from_sz, to_st, overlap_end, remnant, overlap_tgt );
        for( int il = 0; il < 32; ++il ) {
            if( overlap > 0 ) {
                k_buffer_layers[il] = pool_alloc( overlap_k );
                v_buffer_layers[il] = pool_alloc( overlap_k ); // it's the same amt of memory, 1024*overlap_v
                ggml_backend_tensor_get(kv_self->k_l[il], k_buffer_layers[il], to_st_k, overlap_k );
                for( size_t i=0; i<1024; i++ ) {
                    ggml_backend_tensor_get(kv_self->v_l[il], (void*)((char*)v_buffer_layers[il]+(overlap_v*i)), to_st_v+(p_size*i), overlap_v );
                }
            }

            view_k_src = ggml_view_2d(ctx0, kv_self->k_l[il],
                    n_embd_k_gqa, from_sz,
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*from_st));

            view_k_dst = ggml_view_2d(ctx0, kv_self->k_l[il],
                    n_embd_k_gqa, from_sz,
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*to_st));

            view_v_src = ggml_view_2d(ctx0, kv_self->v_l[il],
                    from_sz, n_embd_v_gqa,
                    ggml_row_size(kv_self->v_l[il]->type, kv_self->size),
                    ggml_row_size(kv_self->v_l[il]->type, from_st));

            view_v_dst = ggml_view_2d(ctx0, kv_self->v_l[il],
                    from_sz, n_embd_v_gqa,
                    ggml_row_size(kv_self->v_l[il]->type, kv_self->size),
                    ggml_row_size(kv_self->v_l[il]->type, to_st));

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_k_src, view_k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_v_src, view_v_dst));

            if( remnant > 0 ) {
                view_k_src = ggml_view_2d(ctx0, kv_self->k_l[il],
                        n_embd_k_gqa, remnant,
                        ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),
                        ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*overlap_end));

                view_k_dst = ggml_view_2d(ctx0, kv_self->k_l[il],
                        n_embd_k_gqa, remnant,
                        ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),
                        ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*overlap_tgt));

                view_v_src = ggml_view_2d(ctx0, kv_self->v_l[il],
                        remnant, n_embd_v_gqa,
                        ggml_row_size(kv_self->v_l[il]->type, kv_self->size),
                        ggml_row_size(kv_self->v_l[il]->type, overlap_end));

                view_v_dst = ggml_view_2d(ctx0, kv_self->v_l[il],
                        remnant, n_embd_v_gqa,
                        ggml_row_size(kv_self->v_l[il]->type, kv_self->size),
                        ggml_row_size(kv_self->v_l[il]->type, overlap_tgt));

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_k_src, view_k_dst));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_v_src, view_v_dst));
            }
        }

        // nothing changes in inuse?

        return 0;
    }

    int shuffle_kv_now(int start, int end, int delta) {
        int rv = shuffle_kv(start,end,delta);
        run_kv_shuffler();
        return rv;
    }

    int shuffle_kv(int start, int end, int delta) {
        struct ggml_cgraph * gf = get_shuffler();
        if( delta > 0 && end+delta > kv_self->size ) {
            end = kv_self->size - delta;
        }
        int tgt = start + delta;
        if( tgt < 0 ) {
            LLAMA_LOG_INFO("%s: start=%d, end=%d, delta=%d invalid operation < 0\n", __func__, start, end, delta);
            start = 0-delta;
            tgt = 0;
        }
        int range = end-start;

        LLAMA_LOG_INFO("%s: %d-%d ++ %d -> %d\n", __func__, start, end, delta, tgt);

        for (int il = 0; il < 32; ++il) {
            ggml_tensor * view_k_src = ggml_view_2d(ctx0, kv_self->k_l[il],
                    n_embd_k_gqa, range,
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*start));

            ggml_tensor * view_k_dst = ggml_view_2d(ctx0, kv_self->k_l[il],
                    n_embd_k_gqa, range,
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*tgt));

            ggml_tensor * view_v_src = ggml_view_2d(ctx0, kv_self->v_l[il],
                    range, n_embd_v_gqa,
                    ggml_row_size(kv_self->v_l[il]->type, kv_self->size),
                    ggml_row_size(kv_self->v_l[il]->type, start));

            ggml_tensor * view_v_dst = ggml_view_2d(ctx0, kv_self->v_l[il],
                    range, n_embd_v_gqa,
                    ggml_row_size(kv_self->v_l[il]->type, kv_self->size),
                    ggml_row_size(kv_self->v_l[il]->type, tgt));

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_k_src, view_k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_v_src, view_v_dst));
        }

        // let's move inuse while we're at it

        int i, t=start/8;
        int on_i, on_t = tgt/8;
        uint8_t v;
/*
        for( i=start; i<end; i+=8, t++ ) {
            if( i >= tgt && i+8 <= tgt+range ) continue;
            if( i+8 < end ) {
                v = 0;
            } else {
                v = kv_self->inuse[t];
                for( int b=0; b+i<end; b++ ) {
                    if( b+i >= tgt && b+i <= tgt+range ) break;
                    v &= ~( 1 << (7-b) );
                }
            }
            kv_self->inuse[t] = v;
        }
        end = tgt + range;

        for( i = tgt, t = on_t; i<end; i+=8, on_t++ ) {
            if( i+8 < end ) {
                v = 0xff;
            } else {
                v = kv_self->inuse[t];
                for( int b=0; b+i<end; b++ ) {
                    v |= ( 1 << (7-b) );
                }
            }
            kv_self->inuse[t] = v;
        }
        */

        return delta;
    }
};

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;

    struct token_data {
        token text;
        float score;
        ttype type;
    };

    enum llama_vocab_type type = LLAMA_VOCAB_TYPE_SPM;

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::unordered_map<token, id> special_tokens_cache;

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    id special_bos_id = 1;
    id special_eos_id = 2;
    id special_unk_id = 0;
    id special_sep_id = -1;
    id special_pad_id = -1;

    int special_add_bos = -1; // -1 unknown, 1 add, 0 don't add.
    int special_add_eos = -1; // -1 unknown, 1 add, 0 don't add.

    id linefeed_id       = 13;
    id special_prefix_id = 32007;
    id special_middle_id = 32009;
    id special_suffix_id = 32008;
    id special_eot_id    = 32010;

    bool add_space_prefix = true;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
        GGML_ASSERT(token_left.find(' ') == std::string::npos);
        GGML_ASSERT(token_left.find('\n') == std::string::npos);
        GGML_ASSERT(token_right.find(' ') == std::string::npos);
        GGML_ASSERT(token_right.find('\n') == std::string::npos);

        auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
        if (it == bpe_ranks.end()) {
            return -1;
        }

        return it->second;
    }
};


#define PROCPATH "c:\\lore\\mem\\";

static inline float fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}



typedef struct llama_context _Context;
typedef struct system_eidet System_eidet;
typedef struct system_kb System_kb;
typedef struct llama_model Llama_model;

Llama_model *current_model=NULL;
System_kb *current_kb=NULL;
_Context *current_context=NULL;

typedef struct system_timestamp {
    int year, month, day;
    int hour, min, second;

    std::string to_readable(void)
    {
        std::string when = "";
        int hr;
        if( hour >= 12 ) hr = hour-12;
        else hr = hour;

        when.append(std::to_string(month+1));
        when.append("/");
        when.append(std::to_string(day));
        when.append("/");
        int yr;
        if( year >= 100 )
            yr = 1900 + year;
        else
            yr = year;
        when.append(std::to_string(yr));

        when.append(" at ");

        when.append(std::to_string(hr));
        if( min < 10 )
            when.append(":0");
        else
            when.append(":");
        when.append(std::to_string(min));
        when.append(hour>=12?"pm":"am");
        return when;
    }
    std::string to_string(void)
    {
        std::string when = "time_";
        when.append(std::to_string(year));
        when.append("_");
        when.append(std::to_string(month));
        when.append("_");
        when.append(std::to_string(day));
        when.append("_");
        when.append(std::to_string(hour));
        when.append("_");
        when.append(std::to_string(min));
        when.append("_");
        when.append(std::to_string(second));
        when.append("_");

        return when;
    }
    std::string to_quick(void)
    {
        std::string when = "";
        when.append(std::to_string(min));
        when.append("_");
        when.append(std::to_string(second));
        return when;
    }
} System_timestamp;
System_timestamp *llama_ts_now()
{
    std::time_t t = std::time(0);   // get time now
    std::tm* now = std::localtime(&t);
    System_timestamp *ts = (System_timestamp*)pool_alloc(sizeof(System_timestamp));

    ts->year = now->tm_year;
    ts->month = now->tm_mon;
    ts->day = now->tm_mday;
    ts->hour = now->tm_hour;
    ts->min = now->tm_min;
    ts->second = now->tm_sec;

    return ts;
}
System_timestamp *llama_string_to_ts( std::string str )
{
    //chat_2024_3_22_15_18_46
    int ipos, npos;
    std::string conv;
    System_timestamp *ts = (System_timestamp*)pool_alloc(sizeof(System_timestamp));

    ipos = str.find("_");
    ipos++;
    npos = str.find("_", ipos);
    conv = str.substr(ipos, npos-ipos);
    ts->year = std::stoi( conv );
    ipos = npos+1;
    npos = str.find("_", ipos);
    conv = str.substr(ipos, npos-ipos);
    ts->month = std::stoi( conv );
    ipos = npos+1;
    npos = str.find("_", ipos);
    conv = str.substr(ipos, npos-ipos);
    ts->day = std::stoi( conv );
    ipos = npos+1;
    npos = str.find("_", ipos);
    conv = str.substr(ipos, npos-ipos);
    ts->hour = std::stoi( conv );
    ipos = npos+1;
    npos = str.find("_", ipos);
    conv = str.substr(ipos, npos-ipos);
    ts->min = std::stoi( conv );
    ipos = npos+1;
    npos = str.find("_", ipos);
    conv = str.substr(ipos, npos-ipos);
    ts->second = std::stoi( conv );

    return ts;
}

std::string quick_ts()
{
    System_timestamp *ts = llama_ts_now();
    std::string res = ts->to_quick();
    pool_free(ts);
    return res;
}

std::string readable_ts()
{
    System_timestamp *ts = llama_ts_now();
    std::string res = ts->to_readable();
    pool_free(ts);
    return res;
}

char toLowerCase(char c)
{
    char a = 'a', A = 'A', Z = 'Z';

    if( c >= A && c <= Z ) {
        return c - (A-a);
    }
    return c;
}
void llama_quick_tokenize( std::string raw_text, std::vector<llama_vocab::id> &output );



struct system_memory {
    std::set<std::string> keywords;

    std::string what;
    std::string who;
    System_timestamp *when=NULL;

    uint16_t n_tokens;
    std::vector<int> tokens;

    //std::string where; // add with location

    void prepare()
    {
        new (&what) std::string;
        new (&who) std::string;
        new (&keywords) std::set<std::string>;
        new (&tokens) std::vector<int>;
        when = NULL;
        n_tokens=0;
    }
    void release()
    {
        keywords.clear();
        tokens.clear();
    }
    void readfile( llama_file &file )
    {
        who = file.read_string();
        what = file.read_string();
        n_tokens = file.read_u16();
        std::string strWhen = file.read_string();

        if( tokens.size() != 0 ) tokens.clear();
        llama_quick_tokenize( what, tokens );

        when = llama_string_to_ts(strWhen);

        buildsearch();
    }
    void writefile( llama_file &file )
    {
        //uint16_t sign = (uint16_t)0x10011;
        //file.write_u16(sign);
        file.write_string(who);
        file.write_string(what);
        file.write_u16(n_tokens);
        file.write_string(when->to_string());
    }

    void build( std::string actor, std::string input )
    {
        who = actor;
        when = llama_ts_now();
        what = input;

        if( tokens.size() != 0 ) tokens.clear();
        llama_quick_tokenize( what, tokens );
        n_tokens = tokens.size();

        buildsearch();
    }
    void buildsearch( void )
    {
        const char *instr; // now index the words for search:
        char word[128], *wptr;
        wptr = word;
        *wptr = '\0';

        //LLAMA_LOG_INFO("%s: memory what=%s\n", __func__, what.c_str());

        int iptr, len = what.length();
        char c;

        for( iptr=0; iptr<len; iptr++ ) {
            c = what[iptr]; // = what.c_str() ;; instr++ ) {
            if( c == ' ' || c == '\n' || c == '\0' ) {
                if( *word != '\0' ) {
                    *wptr='\0';
                    keywords.insert( std::string(word) );
                    wptr=word;
                    *wptr='\0';
                }
                if( c == '\0' ) break;
            } else {
                *wptr = toLowerCase(c);
                wptr++;
            }
        }
        if( *word != '\0' ) {
            *wptr='\0';
            //LLAMA_LOG_INFO("%s: add keyword %s\n", __func__, word);
            keywords.insert( std::string(word) );
            wptr=word;
            *wptr='\0';
        }
    }

    int scan( std::vector<std::string *> search )
    {
        std::vector<std::string *>::iterator it;
        int results=0;

        for( it = search.begin(); it != search.end(); it++ ) {
            std::string *keyword = *it;

            if( keywords.find(*keyword) != keywords.end() ) {
                results++;
            }
        }

        return results;
    }
};


struct system_eidet {
    uint16_t n_tokens;
    ggml_fp16_t *kbuf=NULL, *vbuf=NULL;

    std::set<std::string> keywords;

    std::string what;
    std::string who;
    System_timestamp *when=NULL;
    //std::string where; // add with location

    void prepare()
    {
        new (&what) std::string;
        new (&who) std::string;
        new (&keywords) std::set<std::string>;
        when = NULL;
        kbuf = NULL;
        vbuf = NULL;
        n_tokens = 0;
    }

    void release()
    {
        if( kbuf != NULL ) pool_free(kbuf);
        if( vbuf != NULL ) pool_free(vbuf);
        if( when != NULL ) pool_free(when);
        kbuf=NULL;
        vbuf=NULL;
        when=NULL;
        keywords.clear();
    }
    void readfile( llama_file &file )
    {
        who = file.read_string();
        what = file.read_string();
        std::string strWhen = file.read_string();
        when = llama_string_to_ts(strWhen);
        //LLAMA_LOG_INFO("%s\neidet_read: n_tokens\n", strWhen.c_str());
        n_tokens = file.read_u16();

        kbuf = (ggml_fp16_t*)pool_alloc( 32*2048*n_tokens );
        vbuf = (ggml_fp16_t*)pool_alloc( 32*2048*n_tokens );

        file.read_raw( kbuf, 32*2048*n_tokens );
        file.read_raw( vbuf, 32*2048*n_tokens );
    }
    void writefile( llama_file &file )
    {
        //uint16_t sign = (uint16_t)0x10001;
        //file.write_u16(sign);
        file.write_string(who);
        file.write_string(what);
        std::string whenStr = when->to_string();
        file.write_string(whenStr);
        file.write_u16(n_tokens);

        //LLAMA_LOG_INFO("Writing eidet kbuf: %zu bytes for %u tokens (%s)\n", 32*2048*n_tokens, n_tokens, what.c_str());
        file.write_raw( kbuf, 32*2048*n_tokens );
        file.write_raw( vbuf, 32*2048*n_tokens );
    }

    void build( llama_kv_cache *kv, std::string actor, std::string input, uint16_t start, uint16_t used_tokens)
    {
        who = actor;
        what = input;
        when = llama_ts_now();

        n_tokens = used_tokens; /// read in the used tokens:
        LLAMA_LOG_INFO("build eidet: %u start +%u tokens\n%s\n", start, n_tokens, input.c_str());

        kbuf = (ggml_fp16_t*)pool_alloc( 32 * 2048 * n_tokens );
        vbuf = (ggml_fp16_t*)pool_alloc( 32 * 2048 * n_tokens );

        kv->read( start, n_tokens, (void*)kbuf, (void*)vbuf);
    }

    void build2( llama_kv_cache *kv, std::string actor, std::string input, size_t startpt,
                 std::vector<int> tokens,
                 std::vector<ggml_fp16_t> kx[32], std::vector<ggml_fp16_t> vx[32])
    {
        who = actor;
        what = input;
        when = llama_ts_now();

        //LLAMA_LOG_INFO("build2 eidet: %zu start +%zu tokens\n", startpt, tokens.size());

        n_tokens = tokens.size(); /// read in the used tokens:
        kbuf = (ggml_fp16_t*)pool_alloc( 32 * 2048 * n_tokens );
        vbuf = (ggml_fp16_t*)pool_alloc( 32 * 2048 * n_tokens );
        LLAMA_LOG_INFO("eidet %p: copy raw tokens * %u\n", this, n_tokens);

        size_t bufptr;

        size_t v = 2 * n_tokens;
        size_t k = 1024 * v;

        size_t st_t = 2048 * startpt;

        size_t p, p_sz = 2 * n_tokens;
        size_t st_v = startpt * 2;

        size_t endpt = startpt+n_tokens;

        LLAMA_LOG_INFO("%s: startpt %zu n_tokens %zu\n", __func__, startpt, n_tokens);

        for( int il=0; il<32; il++ ) {
            memcpy(kbuf+il*k, kx[il].data(), k );
            //! explore: it might be faster for short generations to read all of v over again instead of per-token per-head
            st_v = startpt * 2;
            bufptr=0;
            for( int t=startpt; t<endpt; t++, st_v += 2 ) {
                for( int i=0, p=st_v; i<1024; i++, p += p_sz, bufptr += 2 ) {
                    memcpy(vbuf+bufptr, vx[il].data()+p, 2 );
                }
            }
        }
    }

    int write( Kv_cache *kv, int start )
    {
        kv->write( start, n_tokens, (void*)kbuf, (void*)vbuf);
        return n_tokens;
    }

    System_memory *convert()
    {
        System_memory *mem = (System_memory*)pool_alloc(sizeof(System_memory));

        new (mem) System_memory;

        mem->prepare();
        mem->who = who;
        mem->what = what;
        mem->when = (System_timestamp*)pool_alloc(sizeof(System_timestamp));
        memcpy( mem->when, when, sizeof(System_timestamp));
        mem->buildsearch();

        return mem;
    }
};
typedef struct kv_mem {
    bool is_full = false;
    bool is_active = false;
    System_memory *m=NULL;
    System_eidet *e=NULL;
    uint16_t first; // token location
    uint16_t last; // inclusive

    void prepare()
    {
        m = NULL;
        e = NULL;
        is_full = false;
        is_active = false;
        first = last = 0;
    }

    System_timestamp *when()
    {
        return is_full ? e->when : m->when;
    }

    void writefile(llama_file &file)
    {
        uint16_t type = !is_full ? 1 : 2;
        file.write_u16( type );

        LLAMA_LOG_INFO("write memory type %u (%s)\n", type, type==1?m->what.c_str():e->what.c_str());

        if( is_full ) e->writefile(file);
        else m->writefile(file);
    }
    void readfile(llama_file &file)
    {
        uint16_t type = file.read_u16();
        first=last=0;
        LLAMA_LOG_INFO("read memory type %u\n", type);
        if( type == 1 ) {
            m = (System_memory*)pool_alloc(sizeof(System_memory));
            new (m) System_memory;
            m->prepare();
            m->readfile(file);
            is_full = false;
            is_active = false;
        } else if( type == 2 ) {
            e = (System_eidet*)pool_alloc(sizeof(System_eidet));
            new (e) System_eidet;
            e->prepare();
            e->readfile(file);
            is_full = true;
            is_active = false;
        } else {
            LLAMA_LOG_INFO("%s: wrong type %d", __func__, type);
        }
    }

    void release(bool release_contents=true)
    {
        if( m != NULL && release_contents ) {
            m->release();
            pool_free(m);
        }
        m=NULL;
        if( e != NULL && release_contents ) {
            e->release();
            pool_free(e);
        }
        e=NULL;
        is_active = is_full = false;
    }
} Kv_mem;

Kv_mem *new_kv_mem( void );
Kv_mem *new_kv_mem( System_memory *memory );
Kv_mem *new_kv_mem( System_eidet *memory );

void llama_backup_file( const char *filepath )
{
    FILE *fp;
    char fp2[256];
    int backupno=0;
    char parsed_path[256], parsed_fn[256];

    fp = fopen(filepath, "rb");
    if( fp ) {
        fclose(fp);
    } else {
        return;
    }

    const char *ptr;
    int len = strlen(filepath), len2;
    for( ptr = filepath+len-1; ptr != filepath; ptr-- ) {
        if( *ptr == '\\' ) {
            len2 = ptr-filepath;
            strncpy(parsed_path, filepath, len2);
            parsed_path[len2] = '\0';
            strncpy(parsed_fn, filepath+(len2)+1, len-(len2+1) );
            parsed_fn[ len - (len2+1) ] = '\0';
            break;
        }
    }
    if( !*parsed_fn ) {
        *parsed_path = '\0';
        strcpy(parsed_fn, filepath);
    }

    while(true) {
        sprintf(fp2, "%s\\%d_%s", parsed_path, backupno, parsed_fn);
        fp = fopen(fp2, "rb");
        if( !fp ) break;
        fclose(fp);
        backupno++;
    }

    rename(filepath, fp2);

    /*

    std::ifstream sourceFile(filepath, std::ios::binary);
    std::ofstream destinationFile(fp2, std::ios::binary);

    if (!sourceFile.is_open() || !destinationFile.is_open()) {
        LLAMA_LOG_INFO("%s: error opening files (%s,%s)!\n", __func__, filepath, fp2);
        return;
    }

    // Copy using rdbuf() for efficiency
    destinationFile << sourceFile.rdbuf();

    sourceFile.close();
    destinationFile.close();
    LLAMA_LOG_INFO("%s: binary copy complete. (%s,%s)\n", __func__, filepath, fp2);
    _unlink(filepath);
    */
}
void loadmemories( const char *filepath, std::vector<Kv_mem*> &mems )
{
    llama_file datafile(filepath, "rb");
    Kv_mem *m;

    if( datafile.fp == NULL ) {
        //! Todo: auto-write description
        // for now we'll just leave them blank.
        LLAMA_LOG_INFO("%s: no file '%s'\n", __func__, filepath);
        return;
    }
    uint32_t count;
    LLAMA_LOG_INFO("%s: reading file %s\n", __func__, filepath);
    count = datafile.read_u32();
    LLAMA_LOG_INFO("%s: load %zu entries.\n", __func__, count);
    while( count > 0 ) {
        m = new_kv_mem();
        m->readfile(datafile);
        mems.push_back(m);
        count--;
    }
}
void savememories( const char *filepath, std::vector<Kv_mem*> &mems )
{
    llama_file datafile(filepath, "wb");
    Kv_mem *m;
    std::vector<Kv_mem*>::iterator it;
    uint32_t count = mems.size();

    LLAMA_LOG_INFO("%s: writing file %s\n", __func__, filepath);
    LLAMA_LOG_INFO("%s: save count:%zu entries.\n", __func__, count);
    datafile.write_u32(count);
    for( it = mems.begin(); it != mems.end(); it++ ) {
        m = *it;
        m->writefile(datafile);
    }

    datafile.close();
}

struct system_actor {
    std::string name;
    System_eidet *self=NULL; // self description data
    bool self_changed=false;
    Kv_mem *mine=NULL;
    std::vector<Kv_mem *> mem; // identity variables & persona variables
    bool mem_changed=false;
    std::vector<Kv_mem *> rags;
    std::set<std::string> ragged; // already in rags
    bool rags_changed=false;
    std::vector<Kv_mem *> history; // things you have seen happen long ago (used for pulling RAG)
    std::vector<Kv_mem *> recent; // things you have seen happen recently

    // idioms;
    //

    std::unordered_map<std::string, Kv_mem*> keys;

    void prepare(void)
    {
        new (&mem) std::vector<Kv_mem *>; // identity variables & persona variables
        new (&rags) std::vector<Kv_mem *>;
        new (&history) std::vector<Kv_mem *>; // things you have seen happen long ago (used for pulling RAG)
        new (&recent) std::vector<Kv_mem *>;
        new (&keys) std::unordered_map<std::string, Kv_mem*>;
        new (&ragged) std::set<std::string>;
        new (&name) std::string;
        mine=NULL;
        self=NULL;
        self_changed=rags_changed=mem_changed=false;
    }

    void release()
    {
        std::vector<Kv_mem*>::iterator i;
        Kv_mem *m;

        for( i = mem.begin(); i != mem.end(); i++ ) {
            m = *i;
            m->release();
        }
        mem.clear();

        for( i = history.begin(); i != history.end(); i++ ) {
            m = *i;
            m->release();
        }
        history.clear();

        for( i = recent.begin(); i != recent.end(); i++ ) {
            m = *i;
            m->release();
        }
        recent.clear();

        for( i = rags.begin(); i != rags.end(); i++ ) {
            m = *i;
            m->release();
        }
        rags.clear();
        ragged.clear();

        if( mine != NULL ) {
            mine->release();
            mine = NULL;
        }
    }

    void appendhist(Kv_mem *m)
    {
        char *datapath = (char*) pool_alloc(name.length() + 10);
        strcpy(datapath, "char\\");
        strcat(datapath, name.c_str());
        strcat(datapath, ".hst");

        llama_file datafile(datapath, "ab");
        m->writefile(datafile);
        datafile.close();
    }

    Kv_mem *addrecent(System_eidet *m)
    {
        LLAMA_LOG_INFO("Add recent eidet %s (%zu tokens)\n", m->what.c_str(), m->n_tokens);
        Kv_mem *mem = new_kv_mem(m);
        recent.push_back(mem);
        return mem;
    }
    Kv_mem *addrecent(System_memory *m)
    {
        LLAMA_LOG_INFO("Add recent memory %s\n", m->what.c_str());
        Kv_mem *mem = new_kv_mem(m);
        recent.push_back(mem);
        return mem;
    }
    Kv_mem *addhist(System_memory *hist)
    {
        Kv_mem *m = new_kv_mem(hist);
        history.push_back(m);
        appendhist(m);

        return m;
    }
    Kv_mem *addrag(System_memory *rag)
    {
        Kv_mem *mem = new_kv_mem(rag);
        rags.push_back(mem);
        ragged.insert( rag->what );
        return mem;
    }
    Kv_mem *addmem(System_eidet *m)
    {
        Kv_mem *mx = new_kv_mem(m);
        mem.push_back(mx);
        return mx;
    }

    void loadfile(void)
    {
        LLAMA_LOG_INFO("%s: start\n", __func__);
        char *rootpath = (char*) pool_alloc(name.length() + 6);
        strcpy(rootpath, "char\\");
        strcat(rootpath, name.c_str());

        char *filepath = (char*) pool_alloc(strlen(rootpath) + 5);
        strcpy(filepath, rootpath);
        strcat(filepath, ".def");
        char *ragpath = (char*) pool_alloc(strlen(rootpath) + 5);
        strcpy(ragpath, rootpath);
        strcat(ragpath, ".rag");
        char *mempath = (char*) pool_alloc(strlen(rootpath) + 5);
        strcpy(mempath, rootpath);
        strcat(mempath, ".mem");
        char *hstpath = (char*) pool_alloc(strlen(rootpath) + 5);
        strcpy(hstpath, rootpath);
        strcat(hstpath, ".hst");
        char *rctpath = (char*) pool_alloc(strlen(rootpath) + 5);
        strcpy(rctpath, rootpath);
        strcat(rctpath, ".rec");

        llama_file playerfile(filepath, "rb");
        Kv_mem *m;

        if( playerfile.fp != NULL ) {
            uint16_t self_found = playerfile.read_u16();
            if( self_found == 1 ) {
                self = (System_eidet*)pool_alloc(sizeof(System_eidet));
                new (self) System_eidet;
                self->prepare();
                self->readfile(playerfile);
                mine = new_kv_mem(self);
            }
            playerfile.close();
        }

        loadmemories(ragpath, rags);
        loadmemories(mempath, mem);
        loadmemories(hstpath, history);
        loadmemories(rctpath, recent);

        LLAMA_LOG_INFO("%s: load complete.\n", __func__);

        self_changed=rags_changed=mem_changed=false; // these refer to the on-disk details
    }
    void savefile()
    {
        char *rootpath = (char*) pool_alloc(name.length() + 6);
        strcpy(rootpath, "char\\");
        strcat(rootpath, name.c_str());

        char *filepath = (char*) pool_alloc(strlen(rootpath) + 5);
        strcpy(filepath, rootpath);
        strcat(filepath, ".def");
        llama_backup_file(filepath);
        llama_file datafile(filepath, "wb");
        if( mine ) {
            datafile.write_u16(1);
            mine->e->writefile(datafile);
        } else {
            datafile.write_u16(0);
        }

        char *mempath = (char*) pool_alloc(strlen(rootpath) + 5);
        strcpy(mempath, rootpath);
        strcat(mempath, ".mem");
        llama_backup_file(mempath);
        savememories(mempath, mem);

        char *ragpath = (char*) pool_alloc(strlen(rootpath) + 5);
        strcpy(ragpath, rootpath);
        strcat(ragpath, ".rag");
        llama_backup_file(ragpath);
        savememories(ragpath, rags);

        char *datapath = (char*) pool_alloc(name.length() + 10);
        strcpy(datapath, "char\\");
        strcat(datapath, name.c_str());
        strcat(datapath, ".hst");

        llama_backup_file(datapath);
        savememories(datapath, history);

        char *recpath = (char*) pool_alloc(name.length() + 10);
        strcpy(recpath, "char\\");
        strcat(recpath, name.c_str());
        strcat(recpath, ".rec");

        llama_backup_file(recpath);
        savememories(recpath, recent);
    }


    std::vector<Kv_mem*> *build_map1()
    {
        Kv_mem *me;
        Kv_mem *src;
        std::vector<Kv_mem*>::iterator it;
        std::vector<Kv_mem*> *res;
        uint16_t token=0;

        LLAMA_LOG_INFO("%s(%s): build map 1\n", __func__, quick_ts().c_str());
        res = (std::vector<Kv_mem*> *)pool_alloc(sizeof(std::vector<Kv_mem*>));
        new (res) std::vector<Kv_mem*>;

        if( mine ) {
            // note: we use a new copy instead of the old copy to make sure nothing has moved or changed shape.
            // if we use the same pointer the comparison will not work.
            me = new_kv_mem(mine->e);
            me->first = 0;
            me->last = mine->e->n_tokens-1;
            token = mine->e->n_tokens;
            if( mine->is_active && mine->first == me->first && mine->last == me->last ) {
                me->is_active = true;
                LLAMA_LOG_INFO("%s: mine is active\n", __func__);
            }
            res->push_back( me );
        }

        for( it = mem.begin(); it != mem.end(); it++ ) {
            src = *it;

            me = new_kv_mem(src->e);
            me->first = token;
            me->last = token + src->e->n_tokens-1;
            if( src->is_active && me->first == src->first ) {
                me->is_active = true; // already written in this placement
            }
            token = me->last+1;

            res->push_back( me );
            LLAMA_LOG_INFO("%s: add mem %zu-%zu\n", __func__, me->first, me->last);
        }

        for( it = rags.begin(); it != rags.end(); it++ ) {
            src = *it;

            if( src->is_full ) {
                me = new_kv_mem(src->e);
                me->first = token;
                me->last = token + src->e->n_tokens - 1;
                token += src->e->n_tokens;
            } else {
                me = new_kv_mem(src->m);
                me->first = token;
                me->last = token + src->m->n_tokens - 1;
                token += src->m->n_tokens;
            }
            if( src->is_active && me->is_full && me->first == src->first ) {
                me->is_active = true; // already written in this placement
            }

            res->push_back( me );
            LLAMA_LOG_INFO("%s: add rag %zu-%zu\n", __func__, me->first, me->last);
        }

        LLAMA_LOG_INFO("%s(%s): done\n", __func__, quick_ts().c_str());
        return res;
    }

    std::vector<System_memory*> *build_map2( uint16_t use_space, std::vector<Kv_mem*> *res )
    {
        Kv_mem *me, *src;
        uint16_t token = 0;
        LLAMA_LOG_INFO("%s(%s): build map 2 use %u\n", __func__, quick_ts().c_str(), use_space);
        if( res->size() > 0 ) {
            src = res->at( res->size()-1 );
            token = src->last+1;
        }
        int it;
        int itBoundary = res->size(), itBound2;
        uint16_t token_cpy = token;
        std::unordered_map<Kv_mem *, Kv_mem *> new_to_old;

        // insert history in reverse from end to beginning as far as we can go
        if( recent.size() > 0 ) {
            LLAMA_LOG_INFO("Recent entries: %zu remnant token: %u\n", recent.size(), token_cpy);
        }
        for( it = recent.size()-1; it >= 0; it-- ) {
            src = recent[it];

            if( src->is_full ) {
                if( token+src->e->n_tokens > use_space ) break;
                token += src->e->n_tokens;
                me = new_kv_mem(src->e);
            } else {
                if( token+src->m->n_tokens > use_space ) break;
                token += src->m->n_tokens;
                me = new_kv_mem(src->m);
            }
            new_to_old[me] = src;
            res->insert( res->begin()+itBoundary, me );
        }

        itBound2 = history.size();
        std::vector<System_memory*> *histcopy;
        histcopy = (std::vector<System_memory*>*)pool_alloc(sizeof(std::vector<System_memory*>));
        new (histcopy) std::vector<System_memory*>;


        if( it >= 0 )
            LLAMA_LOG_INFO("Move %d recent entries to history.\n", (it+1));
        while( it >= 0 ) { // move overflow to history
            src = recent[it];
            me = new_kv_mem((System_memory*)pool_alloc(sizeof(System_memory)));
            new (me->m) System_memory;
            me->m->prepare();
            if( src->is_full ) {
                me->m->build( src->e->who, src->e->what );
            } else {
                me->m->build( src->m->who, src->m->what );
            }
            me->is_active = false;
            history.insert(history.begin() + itBound2, me);
            histcopy->insert(histcopy->begin(), me->m);

            recent.erase(recent.begin()+it);
            it--;
        }

        char buf[60];
        token = token_cpy; // final token count & organize results:
        for( it = itBoundary; it < res->size(); it++ ) {
            me = res->at(it);
            src = new_to_old.at(me);

            me->first = token;
            if( me->is_full )
                token += src->e->n_tokens;
            else
                token += src->m->n_tokens;
            me->last = token==0?0:token-1;

            strncpy(buf, me->e->what.c_str(), 50);
            buf[ me->e->what.length() < 50 ? me->e->what.length() : 50 ] = '\0';
            LLAMA_LOG_INFO("%s: set token start %u-end %u for %s message %s\n", __func__, me->first, me->last, me->is_full?"(full)":"(partial)", buf);

            if( src->is_full && src->is_active && me->first == src->first ) {
                me->is_active=true;
            }
        }
        LLAMA_LOG_INFO("%s(%s): done\n", __func__, quick_ts().c_str());

        return histcopy;
    }
};

struct llama_model {
    e_model     type  = MODEL_UNKNOWN;
    llm_arch    arch  = LLM_ARCH_UNKNOWN;
    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::string name = "n/a";

    std::vector<ggml_backend_t> backends;
    std::string impressionspath;

    llama_hparams hparams = {};
    llama_vocab   vocab;

    struct ggml_tensor * tok_embd;
    struct ggml_tensor * type_embd;
    struct ggml_tensor * pos_embd;
    struct ggml_tensor * tok_norm;
    struct ggml_tensor * tok_norm_b;

    struct ggml_tensor * output_norm;
    struct ggml_tensor * output_embd;
    struct ggml_tensor * output_norm_b;
    struct ggml_tensor * output;
    struct ggml_tensor * output_b;

    std::vector<llama_layer> layers;

    llama_split_mode split_mode;
    int main_gpu;
    int n_gpu_layers;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // layer -> buffer type mapping
    struct layer_buft {
        layer_buft() : buft_matrix(nullptr), buft(nullptr) {}
        layer_buft(ggml_backend_buffer_type_t matrix) : buft_matrix(matrix), buft(matrix) {}
        layer_buft(ggml_backend_buffer_type_t matrix, ggml_backend_buffer_type_t other) : buft_matrix(matrix), buft(other) {}

        ggml_backend_buffer_type_t buft_matrix; // matrices only - used by split buffers and backends that support only matrix multiplication
        ggml_backend_buffer_type_t buft;        // everything else
    };

    layer_buft buft_input;
    layer_buft buft_output;
    std::vector<layer_buft> buft_layer;

    // contexts where the model tensors metadata is stored
    std::vector<struct ggml_context *> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_t> bufs;

    // model memory mapped file
    std::unique_ptr<llama_mmap> mapping;

    // objects representing data potentially being locked in memory
    std::vector<std::unique_ptr<llama_mlock>> mlock_bufs;
    llama_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    ~llama_model() {
        //for (auto it : contexts) {
        //    llama_free(it);
        //}
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }

        if( current_model == this )
            current_model=NULL;
    }
};


#define CONTEXT_DEFAULT 0
#define CONTEXT_ZERO 1
#define CONTEXT_USER 2
#define CONTEXT_WORKER 3


struct llama_context {
    bool is_clone=false;
    llama_context(llama_model & model) : model(model), t_start_us(model.t_start_us), t_load_us(model.t_load_us) {
        //memset( kv_self->inuse, 0, 512 );
    }
    struct llama_kv_cache *kv_self;
    ~llama_context() {
        LLAMA_LOG_INFO("%s: eliminate %p\n", __func__, this);

        ggml_backend_sched_free(sched);
        backends.clear();

#ifdef GGML_USE_VULKAN
        ggml_vk_free_cpu_assist();
#endif
        ggml_backend_buffer_free(buf_input);
        ggml_free(ctx_input);
        //llama_release_context2((llama_model&)model, this);
    }

    llama_cparams cparams;

    //std::vector<llama_context *> children;
    //std::vector<_Keyed*> locks;
    //std::unordered_map<std::string, int> doors;

    int type = CONTEXT_DEFAULT;

    int sequential_start=0;
    int seq_end=0;

    std::vector<ggml_backend *> backends;
#ifdef GGML_USE_METAL
    ggml_backend_t backend_metal = nullptr;
#endif
    ggml_backend *backend_cpu = nullptr;

    llama_model & model;

    std::string charname = "";
    float imp_scale_factor=0.1;
    //std::vector<Llama_processed *> fullreport;

    std::mt19937 rng;

    bool has_evaluated_once = false;
    bool ctx_ready = false;
    int record_all = 0;

    int64_t t_start_us;
    int64_t t_load_us;
    int64_t t_sample_us = 0;
    int64_t t_p_eval_us = 0;
    int64_t t_eval_us   = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    int32_t n_eval   = 0; // number of eval calls

    std::vector<float> logits;
#ifndef NDEBUG
    // guard against access to unset logits
    std::vector<bool>  logits_valid;
#endif
    bool logits_all = false;

    std::vector<float> embd;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_t sched = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    struct ggml_tensor * tok_embd;

    // input tensors
    ggml_backend_buffer_t buf_input = nullptr;
    ggml_context * ctx_input = nullptr;
    struct ggml_tensor * inp_tokens;    // I32 [n_batch]
    struct ggml_tensor * inp_embd;      // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos;       // I32 [n_batch]
    struct ggml_tensor * inp_KQ_mask;   // F32 [n_ctx, n_batch]
    struct ggml_tensor * inp_KQ_pos;    // F32 [n_ctx]
    struct ggml_tensor * inp_K_shift;   // I32 [n_ctx]
    struct ggml_tensor * inp_mean;      // F32 [n_batch, n_batch]

    struct ggml_tensor * inp_cls;       // I32 [n_batch]

#ifdef GGML_USE_MPI
    ggml_mpi_context * ctx_mpi = NULL;
#endif
};

void prepare_kv_cache(struct llama_context *ctx, int n_ctx, int n_batch);

typedef struct system_kb System_kb;
struct system_kb {
    std::vector<System_actor*> actors;
    std::unordered_map<std::string, System_actor*> players;
    std::unordered_map<std::string, std::vector<System_memory*> *> ragwordmap;
    std::vector<Kv_mem*> allmessages;
    struct llama_kv_cache kv[3];

    std::string writinguser;

    uint16_t kv_extent[3] = {0,0,0};
    uint16_t seq_start[3] = {0,0,0};
    std::vector<Kv_mem*> *kvmap[3] = {NULL,NULL,NULL};
    System_actor *kvuser[3] = {NULL,NULL,NULL};
    bool kv_ready[3] = {false,false,false};
    uint8_t current_kv;
    int16_t seq_mark[3] = {-1,-1,-1};
    int16_t gen_mark[3] = {-1,-1,-1};
    int16_t gen_prev[3] = {0, 0, 0};
    std::string gen_str_so_far[3];

    /*
    std::vector<int> gen_tokens_so_far[3];
    std::vector<ggml_fp16_t> gen_k_so_far[3][32];
    std::vector<ggml_fp16_t> gen_v_so_far[3][32];
    */

    void prepare(void)
    {
        new (&actors)     std::vector<System_actor*>;
        new (&players) std::unordered_map<std::string, System_actor*>;
        new (&ragwordmap) std::unordered_map<std::string, std::vector<Kv_mem*>> ;
        new (&active_actor) std::string;
        active_actor = "System";
        new (&allmessages) std::vector<Kv_mem*>;

        for( int i=0; i<3; i++ ) {
            new (&(kv[i])) struct llama_kv_cache;
            //kv[i].prepare();
            kv_extent[i] = 0;
            seq_start[i] = 0;
            kvmap[i] = NULL;
            kvuser[i] = NULL;
            kv_ready[i] = false;
            gen_mark[i] = seq_mark[i] = -1;
            gen_prev[i] = 0;
            new (&gen_str_so_far[i]) std::string;
            gen_str_so_far[i] = "";
            /*
            for( int j=0; j<32; j++ ) {
                new (&gen_k_so_far[i][j]) std::vector<ggml_fp16_t>;
                new (&gen_v_so_far[i][j]) std::vector<ggml_fp16_t>;
            }
            new (&gen_tokens_so_far[i]) std::vector<int>;
            */
        }
        new (&writinguser) std::string;

        writinguser = "";
        LLAMA_LOG_INFO("%s: prepared system_kb\n", __func__);
    }
    void mark_rewind(void)
    {
        for( int i=0; i<3; i++ ) {
            seq_mark[i] = seq_start[i];
        }
    }
    void rewind_to_mark(void)
    {
        for( int i=0; i<3; i++ ) {
            seq_start[i] = seq_mark[i];
            seq_mark[i] = -1;
        }
    }
    void mark_generation(std::string author)
    {
        writinguser = author;
        for( int i=0; i<3; i++ ) {
            if( !kvuser[i] ) continue;
            if( kvuser[i]->name != author ) continue;

            gen_mark[i] = seq_start[i];
            gen_prev[i] = 0;
            gen_str_so_far[i] = "";
            LLAMA_LOG_INFO("set gen_mark[%d] to seq_start %u\n", i, seq_start[i]);
            /*
            for( int j=0; j<32; j++ ) {
                gen_k_so_far[i][j].clear();
                gen_v_so_far[i][j].clear();
            }
            gen_tokens_so_far[i].clear();
            */
        }
    }
    void rewind_generation(std::string message, std::vector<int> &tokens)
    {
        // create memories:
        std::set<System_actor *> messaged;
        Kv_mem *mem;
        bool is_author;

        LLAMA_LOG_INFO("%s: msg '%s', tokens %zu\n", __func__, message.c_str(), tokens.size());

        for( int loop=0; loop<2; loop++ ) {
            for( int i=0; i<3; i++ ) {
                if( loop==0 && ( !kv_ready[i] || !kvuser[i] ) ) {
                    gen_mark[i] = -1;
                    gen_str_so_far[i] = "";
                }
                if( !kv_ready[i] || !kvuser[i] ) {
                    continue;
                }

                is_author = ( writinguser == kvuser[i]->name );

                if( loop == 0 && !is_author ) continue;
                if( loop == 1 && is_author ) continue;

                messaged.insert(kvuser[i]);
                useactor(kvuser[i]->name);
                tokens.clear();
                if( is_author )
                    llama_quick_tokenize(gen_str_so_far[i], tokens);
                else
                    llama_quick_tokenize(message, tokens);
                LLAMA_LOG_INFO("%d(%s): msg '%s', scan '%s', tokens %zu\n", i, kvuser[i]->name.c_str(), message.c_str(), gen_str_so_far[i].c_str(), tokens.size());

                if( kvuser[i]->name == writinguser ) {
                    System_eidet *e = (System_eidet*)pool_alloc(sizeof(System_eidet));
                    new (e) System_eidet;
                    e->prepare();
                    e->build(&(kv[i]), writinguser, gen_str_so_far[i], gen_mark[i], tokens.size());
                    mem = kvuser[i]->addrecent(e);
                    mem->first = gen_mark[i];
                    mem->last = gen_mark[i] + tokens.size() - 1;
                    mem->is_active = true;
                } else {
                    processtokens(writinguser, message, tokens, false, tokens.size());
                    /*
                    System_memory *m = (System_memory*)pool_alloc(sizeof(System_memory));
                    new (m) System_memory;
                    m->prepare();
                    m->build(writinguser, message);
                    mem = kvuser[i]->addrecent(m);
                    processtokens_inplace(i, mem); //! todo: we may not need this.
                    */
                }
                gen_str_so_far[i] = "";
                gen_mark[i] = -1;
                ragunmap(kvuser[i], message); // search for any ragged messages in the past
            }
        }

        //LLAMA_LOG_INFO("%s(%s): ragunmap\n", __func__, quick_ts().c_str());

        //LLAMA_LOG_INFO("%s(%s): finish\n", __func__, quick_ts().c_str());
        std::vector<System_actor *>::iterator it;
        for( it = actors.begin(); it != actors.end(); it++ ) {
            if( messaged.contains( *it ) ) continue;

            System_actor *a = *it;
            System_memory *m = (System_memory*)pool_alloc(sizeof(System_memory));
            new (m) System_memory;
            m->prepare();
            m->build(writinguser, message);
            LLAMA_LOG_INFO("store message for %s: %s\n", a->name.c_str(), message.c_str());
            a->addrecent(m);
        }

        writinguser = "";
        LLAMA_LOG_INFO("%s(%s): done\n", __func__, quick_ts().c_str());
    }
    void query_actor_names(std::vector<std::string> &names)
    {
        std::vector<System_actor *>::iterator it;
        System_actor *a;
        names.clear();

        for( it = actors.begin(); it != actors.end(); it++ ) {
            a = *it;
            names.push_back(a->name); // does not include "System" name
        }

        return;
    }

    Kv_mem *getmem(void)
    {
        Kv_mem *memitem = (Kv_mem*)pool_alloc(sizeof(Kv_mem));
        new (memitem) Kv_mem;
        memitem->prepare();
        allmessages.push_back(memitem);
        return memitem;
    }
    void freemem(Kv_mem *memitem, bool withmem)
    {
        std::vector<Kv_mem*>::iterator it;
        for( it = allmessages.begin(); it != allmessages.end(); it++ ) {
            if( *it == memitem ) {
                allmessages.erase(it);
                break;
            }
        }
        memitem->release(withmem);
        pool_free(memitem);
    }
    void freemems(std::set<Kv_mem *>items, bool withmem)
    {
        size_t it;
        Kv_mem *memitem;

        for( it = 0; it < allmessages.size(); it++ ) {
            if( items.contains(allmessages[it]) ) {
                memitem = allmessages[it];
                allmessages.erase(allmessages.begin() + it);
                it--;
                memitem->release(withmem);
                pool_free(memitem);
            }
        }
    }

    llama_hparams           hparams;

    std::string active_actor;
    int n_batch = 64;

    // save all actors' current data
    void saveall(void)
    {
        std::vector<System_actor*>::iterator it;
        System_actor *a;

        LLAMA_LOG_INFO("Saving all...\n");
        for( it = actors.begin(); it != actors.end(); it++ ) {
            a = *it;
            a->savefile();
        }
        LLAMA_LOG_INFO("Save complete.\n");
    }
    void unload( std::string actor )
    {
        if( players.contains(actor) ) {
            LLAMA_LOG_INFO("%s: found actor %s\n", __func__, actor.c_str());
            System_actor *a = players[actor];
            a->savefile();
            a->release();
            players.erase(actor);
            std::vector<System_actor*>::iterator it;
            for( it = actors.begin(); it != actors.end(); it++ ) {
                if( *it == a ) {
                    actors.erase(it);
                    break;
                }
            }
            pool_free(a);
        }
    }

    void release()
    {
        int i;
        Kv_mem *mem;
        Kv_mem *eid;

        System_actor *actor;

        std::vector<System_actor*>::iterator itActor;
        std::vector<Kv_mem*>::iterator itMap;

        for( i=0; i<3; i++ ) {
            if( kv_ready[i] ) kv_ready[i]=false;
            if( kvmap[i] != NULL ) {
                for( itMap = kvmap[i]->begin(); itMap != kvmap[i]->end(); itMap++ ) {
                    eid = *itMap;
                    eid->release(false);
                    pool_free(eid);
                }
                delete kvmap[i];
                kvmap[i] = NULL;
            }
            if( kvuser[i] != NULL )
                kvuser[i] = NULL;
        }
        for( itActor = actors.begin(); itActor != actors.end(); itActor++ ) {
            actor = *itActor;
            actor->release();
            pool_free(actor);
        }
    }

    struct llama_kv_cache *usekv( int kvno )
    {
        current_kv = kvno;

        if( !kvuser[kvno] ) {
            LLAMA_LOG_INFO("usekv %d: no actor found!\n", kvno);
            throw "Actor not resolved.";
        }

        //LLAMA_LOG_INFO("%s: use kv %d (actor %s)\n", __func__, kvno, kvuser[kvno]->name.c_str());

        if( !kv_ready[kvno] ) {
            int n = kvno;
            int n_ctx = n==0?1024:4096;

            kv_extent[n] = n_ctx;
            kv[n].prepare();

            current_context->kv_self = &(kv[n]);
            current_context->cparams.n_ctx = kv_extent[n];

            if (!llama_kv_cache_init((kv[n]), *current_model,
                    GGML_TYPE_F16, GGML_TYPE_F16, kv_context*n_ctx,
                    true)) {
                LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
                throw "Could not allocate kv cache.";
            }

            current_context->kv_self = &(kv[n]);
            prepare_kv_cache(current_context, n_ctx, n_batch);
            LLAMA_LOG_INFO("%s: prepared kv %d\n", __func__, n);

            seq_start[n] = 0;

            size_t memory_size_k = 0;
            size_t memory_size_v = 0;

            for (auto & k : kv[n].k_l) {
                memory_size_k += ggml_nbytes(k);
            }
            for (auto & v : kv[n].v_l) {
                memory_size_v += ggml_nbytes(v);
            }

            LLAMA_LOG_INFO("%s: KV[%d] self size  = %7.2f MiB, K (f16): %7.2f MiB, V (f16): %7.2f MiB\n", __func__,
                           n,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                (float)memory_size_k / (1024.0f * 1024.0f),
                (float)memory_size_v / (1024.0f * 1024.0f));

            kv_ready[n] = true;
        } else {
            current_context->kv_self = &(kv[kvno]);
            current_context->cparams.n_ctx = kv_extent[kvno];
        }
        current_context->seq_end = seq_start[kvno];

        return &kv[kvno];
    }

    void usemap( int kvno, std::vector<Kv_mem*> *map, bool finalize )
    {
        std::vector<Kv_mem*>::iterator it, it1, it2;
        Kv_mem *e1, *e2, *eid;
        uint16_t n_tokens=0;

        std::vector<Kv_mem*> *prev = kvmap[kvno];

        bool wrote = false;

        // decide where everything goes (including partial memories)
        int token=0;
        if( map->size() > 0 )
            LLAMA_LOG_INFO("%s(%s): map kv(%d) has %d entries.\n", __func__, quick_ts().c_str(), kvno, map->size());

        std::set<Kv_mem*> movable;
        std::unordered_map<Kv_mem*, Kv_mem*> new_to_old;
        if( prev ) {
            // anything in the old map that exists in the new map will be 'movable'
            // unless it is already written in the proper location
            for( it1 = prev->begin(); it1 != prev->end(); it1++ ) {
                e1 = *it1;
                for( it2 = map->begin(); it2 != map->end(); it2++ ) {
                    e2 = *it2;
                    if( !e2->e || e2->is_active ) continue;
                    if( e1->e == e2->e ) { // they must be literally the same pointer internally
                        movable.insert(e2);
                        new_to_old[e2] = e1;
                        break;
                    }
                }
            }
        }
        if( movable.size() != 0 ) {
            //LLAMA_LOG_INFO("Movable entries: %zu\n", movable.size());
        }

        // setup a plan to move movable memories
        std::set<Kv_mem*> moving;

        // first loop: check for anything that can be moved into non-reserved areas
        // anything that passes this first check or the second check below goes into 'moving'
        bool found=false;
        for( it = map->begin(); it != map->end(); it++ ) {
            eid = *it;
            if( !eid->e || eid->is_active ) continue;
            if( movable.contains(eid) ) {
                found=false;
                for( it1 = prev->begin(); it1 != prev->end(); it1++ ) { // check for overlaps:
                    e1 = *it1;
                    if( ( eid->first >= e1->first && eid->first <= e1->last ) ||
                        ( eid->last >= e1->first && eid->last <= e1->last ) ) {
                        if( eid->e == e1->e && eid->first == e1->first ) {
                            continue;
                        }
                        found=true;
                        break;
                    }
                }
                if( !found ) {
                    moving.insert(eid);
                }
            }
        }

        // second loop: weigh options between moving and writing directly
        found=false;
        std::set<System_eidet*> removed;
        std::vector<System_eidet*> removing;
        std::vector<System_eidet*>::iterator it3;

        for( it = map->begin(); it != map->end(); it++ ) {
            eid = *it;
            if( !eid->e || eid->is_active ) continue;
            if( movable.contains(eid) && !moving.contains(eid) && !removed.contains(eid->e) ) {
                // locate what we are removing
                int weight_for = eid->e->n_tokens;
                removing.clear();
                for( it1 = prev->begin(); it1 != prev->end(); it1++ ) {
                    e1 = *it1;
                    if( ( eid->first >= e1->first && eid->first <= e1->last ) ||
                        ( eid->last >= e1->first && eid->last <= e1->last ) ) {
                        // overlap found
                        int weight_counter = e1->e->n_tokens;
                        for( it2 = map->begin(); it2 != map->end(); it2++ ) {
                            e2 = *it2;
                            if( e2->e == eid->e || e2->e == e1->e ) continue;
                            int overlap = 0;
                            if( e1->first >= e2->first && e1->first <= e2->last ) {
                                if( e1->last < e2->last ) {
                                    overlap += 1 + e1->last - e1->first;
                                } else {
                                    overlap += 1 + e2->last - e1->first;
                                }
                            } else if( e1->last >= e2->first && e1->last <= e2->last ) {
                                if( e1->first < e2->first ) {
                                    overlap += 1 + e1->last - e2->first;
                                } else {
                                    overlap += 1 + e1->last - e1->first;
                                }
                            } else if( e1->first < e2->first && e1->last > e2->last ) {
                                overlap = 1 + e2->last - e2->first;
                            }
                            weight_counter -= overlap;
                        }
                        weight_for -= weight_counter;
                        removing.push_back(e1->e); // make sure we don't try to move this one
                    }
                }
                if( weight_for > 0 ) {
                    moving.insert(eid);
                    for( it3 = removing.begin(); it3 != removing.end(); it3++ ) {
                        removed.insert( *it3 );
                    }
                }
            }
        }

        if( moving.size() != 0 ) {
            //LLAMA_LOG_INFO("Moving entries: %zu\n", moving.size());
        }

        //LLAMA_LOG_INFO("%s: prepare llm for shift\n", __func__);
        Org_context llm(&kv[kvno], current_context->buf_compute_meta, current_context->sched, hparams);
        bool initialized=false;
        // shuffle memories that can be moved
        n_tokens=0;
        for( it = map->begin(); it != map->end(); it++ ) {
            eid = *it;
            if( !eid->e || eid->is_active ) continue;
            if( moving.contains(eid) && !removed.contains(eid->e) ) {
                e1 = new_to_old[eid];
                if( e1->first != eid->first ) {
                    if( !initialized ) {
                        llm.init();
                        initialized=true;
                    }
                    int delta = (int)eid->first - (int)e1->first;
                    llm.shuffle_kv( e1->first, e1->last, delta );
                }
            }
        }
        if( initialized ) {
            llm.run_kv_shuffler();
        }
        if( initialized ) {
            LLAMA_LOG_INFO("%s(%s): release shuffler\n", __func__, quick_ts().c_str());
            llm.free();
        }

        // write unmovable memories
        std::set<ggml_backend_t> backend_list;
        wrote=false;
        int entry_count=0;
        for( it = map->begin(); it != map->end(); it++ ) {
            eid = *it;
            n_tokens = n_tokens>=eid->last+1?n_tokens:eid->last+1;

            if( !eid->is_full ) continue;
            if( eid->is_active ) {
                int32_t len = eid->last - eid->first;
                LLAMA_LOG_INFO("%s(%s): active @ %zu tokens %ld\n", __func__, quick_ts().c_str(), eid->first, len);
                continue;
            }
            if( !movable.contains(eid) || removed.contains(eid->e) ) {
                if( !wrote ) {
                    LLAMA_LOG_INFO("%s(%s): begin write\n", __func__, quick_ts().c_str());
                    kv[kvno].prefit_clear();
                    wrote=true;
                }
                eid->e->write(&(kv[kvno]), eid->first);
                entry_count++;
            }
        }
        if( wrote ) {
            kv[kvno].prefit_write();
            LLAMA_LOG_INFO("%s(%s): write complete, %d entries\n", __func__, quick_ts().c_str(), entry_count);
        }

        // translate any partial memories into the kb
        bool writing_partial=false;
        for( it = map->begin(); it != map->end(); it++ ) {
            eid = *it;
            n_tokens = n_tokens>=eid->last+1?n_tokens:eid->last+1;
            if( eid->is_full ) continue;

            if( !writing_partial ) {
                LLAMA_LOG_INFO("%s: partial memories found\n", __func__);
                throw "Partial memories in recent\n";
                //writing_partial=true;
            }

            //seq_start[kvno] = n_tokens;
            //processtokens_inplace(kvno, eid);
        }

        if( finalize ) {
            LLAMA_LOG_INFO("%s(%s): finalize (transfer map)\n", __func__, quick_ts().c_str());
            kvmap[kvno] = map;
        }
        if( map->size() == 0 ) { // reset n_tokens... the previous methods skip over counting some entries
            n_tokens = 0;
        } else {
            eid = map->at(map->size()-1);
            if( n_tokens != eid->last+1 ) {
                LLAMA_LOG_INFO("[[[[[[%s: token count mismatch: %zu expected %zu]]]]]]\n", __func__, n_tokens, eid->last+1);
            }
            n_tokens = eid->last+1;
        }
        seq_start[kvno] = kv[kvno].seq = current_context->seq_end = n_tokens;
        LLAMA_LOG_INFO("%s(%s): done, n_tokens=%u\n", __func__, quick_ts().c_str(), n_tokens);
    }

    uint8_t useactor( std::string actorname, bool quadruple_space=false )
    {
        System_actor *a = getactor(actorname);
        bool is_system_user = ( actorname == "System" );
        uint8_t tgt_kv;

        if( is_system_user ) {
            tgt_kv=0;
            kvuser[0] = a;
        } else if( kvuser[1] == a ) {
            tgt_kv=1;
        } else if( kvuser[2] == a ) {
            tgt_kv=2;
        } else if( kvuser[1] == NULL ) {
            tgt_kv=1;
            kvuser[tgt_kv] = a;
        } else if( kvuser[2] == NULL ) {
            tgt_kv=2;
            kvuser[tgt_kv] = a;
        } else {
            tgt_kv=2;
            kvuser[tgt_kv] = a;
        }
        //if( current_kv == tgt_kv ) return seq_start[tgt_kv];
        LLAMA_LOG_INFO("%s: pick %s for %d\n", __func__, actorname.c_str(), tgt_kv);

        uint16_t reserve_space=64;
        usekv(tgt_kv);

        uint16_t use_space = kv[tgt_kv].size - reserve_space;
        if( quadruple_space ) use_space -= 3*reserve_space;
        use_space-=64; // extra tokens in case of additions or queries
        uint16_t pad_space = 0;
        System_eidet *pad_eidet=NULL;

        // generate a's map
        // first we need to see how much space we're already using in the current gen
        /* this way was too slow:
        if( gen_mark[tgt_kv] != -1 && gen_mark[tgt_kv] != seq_start[tgt_kv] ) {
            pad_space = ( seq_start[tgt_kv] - gen_mark[tgt_kv] );
            use_space -= pad_space + 65; // speed up a little bit by providing extra space while generating long
            // queries.
            LLAMA_LOG_INFO("reserve space for %u: %u (%u) mark: %ld\n", tgt_kv, use_space, pad_space, gen_mark[tgt_kv]);
        }
        */
        //! Todo: maintain the map itself instead of building it every time we select an actor
        std::vector<Kv_mem*> *eidmap = a->build_map1();
        usemap( tgt_kv, eidmap, false );
        std::vector<System_memory*> *new_histories = a->build_map2(use_space, eidmap);
        usemap( tgt_kv, eidmap, true );

        /*if( pad_space > 0 && seq_start[tgt_kv] != gen_mark[tgt_kv] ) {
            LLAMA_LOG_INFO("re-write reserve space [%u tokens]\n", pad_space);
            gen_mark[tgt_kv] = seq_start[tgt_kv];
            kv[tgt_kv].write2( gen_mark[tgt_kv], 0, pad_space, gen_k_so_far[tgt_kv], gen_v_so_far[tgt_kv]);
            seq_start[tgt_kv] += pad_space;
        }*/

        // add to ragwordmap if memories were cycled
        std::string srch;
        std::vector<System_memory*>::iterator it;
        for( it = new_histories->begin(); it != new_histories->end(); it++ ) {
            System_memory *m = *it;
            for( auto srch : m->keywords ) {
                std::vector<System_memory *> *x;

                if( ragwordmap.contains(srch) ) {
                    x = ragwordmap.at(srch);
                } else {
                    x = (std::vector<System_memory *>*)pool_alloc(sizeof(std::vector<System_memory *>));
                    new (x) std::vector<System_memory *>;
                    ragwordmap[srch]=x;
                }
                x->push_back( m );
            }
        }

        return tgt_kv;
    }

    void ragunmap( System_actor *a, std::string what )
    {
        std::unordered_map< std::string, std::vector<System_memory *> *> results;
        std::string *pstr;

        char word[128], *wptr;
        wptr = word;
        *wptr = '\0';

        LLAMA_LOG_INFO("%s: unmap what=%s\n", __func__, what.c_str());

        int iptr, len = what.length();
        char c;

        for( iptr=0; iptr<len; iptr++ ) {
            c = what[iptr];
            //LLAMA_LOG_INFO("%s: char=%c\n", __func__, c);
            if( c == ' ' || c == '\n' || c == 0 ) {
                if( *word != '\0' ) {
                    *wptr='\0';
                    if( ragwordmap.contains(word) && !results.contains(word) ) {
                        results[word] = ragwordmap[word];
                    }
                    wptr=word;
                    *wptr='\0';
                }
                if( c == '\0' ) break;
            } else {
                *wptr = toLowerCase(c);
                wptr++;
            }
        }
        if( *word != '\0' ) {
            *wptr='\0';
            if( ragwordmap.contains(word) && !results.contains(word) ) {
                results[word] = ragwordmap[word];
            }
            wptr=word;
            *wptr='\0';
        }

        std::unordered_map<System_memory *, int> counts;
        std::vector<System_memory *>::iterator it;
        for( const auto &pair : results ) {
            for( it = pair.second->begin(); it != pair.second->end(); it++ ) {
                if( !(counts.contains(*it)) )
                    counts[*it]=1;
                else
                    counts[*it] = counts[*it]+1;
            }
        }
        int desired_adds=2;
        System_memory *highest;
        std::vector<System_actor *>::iterator itAct;
        while( desired_adds > 0 ) {
            int highest_count=0;
            for( const auto &pair : counts ) {
                if( pair.second > highest_count ) {
                    highest_count = pair.second;
                    highest = pair.first;
                }
            }
            if( highest_count <= 2 ) break;
            LLAMA_LOG_INFO("%s: unmap highest=%d\n", __func__, highest_count);
            counts[highest] = 0;
            desired_adds--;
            // add rag to actor's map
            Kv_mem *memitem = getmem();
            LLAMA_LOG_INFO("%s: unmap memory=%s\n", __func__, highest->what.c_str());
            if( a->ragged.contains(highest->what) ) continue;
            a->ragged.insert(highest->what);
            memitem->e = translate(highest);
            memitem->m = NULL;
            memitem->is_active = false;
            memitem->is_full = true;
            if( a->rags.size() > 2 ) {
                a->rags.erase( a->rags.begin(), a->rags.begin()+a->rags.size()-2 );
            }
            a->rags.push_back( memitem );
            a->rags_changed = true;
        }

        LLAMA_LOG_INFO("%s(%s): done\n", __func__, quick_ts().c_str());
    }

    System_eidet *translate(System_memory *mem)
    {
        std::vector<llama_token> tokens;
        Kv_mem *newmem;
        if( !(newmem=processtokens(mem->who, mem->what, tokens, false, 0, true)) )
        {
            LLAMA_LOG_INFO("%s: failed to translate %s\n", __func__, mem->what.c_str());
            throw "Couldn't translate tokens\n";
        }

        return newmem->e;
    }

    // send a message to all agents
    int process_tokens( std::string toname, std::string fromname, std::string message,
                        std::vector<llama_token> &tokens )
    {
        std::set<System_actor*> messaged;
        System_actor *a;

        if( fromname.length() == 0 ) {
            LLAMA_LOG_INFO("%s: Specify who message is from.\n", __func__);
            return 0;
        }

        int ts_prev = tokens.size();
        llama_quick_tokenize( message, tokens );
        int ts_addit = tokens.size() - ts_prev;
        int ts_rewind = 0;
        if( ts_addit == 0 ) {
            ts_addit=1;
            ts_rewind=1;
        }

        int batches = floor(ts_addit/64.0);
        int n_last_batch = ts_addit - 64 * batches;
        LLAMA_LOG_INFO("%s: tokens=%d, additional=%d, rewind=%d to: %s\n", __func__, tokens.size(), ts_addit, ts_rewind, toname.c_str());

        if( toname != "all" ) {
            uint8_t tgt_kv;
            if( kvuser[current_kv]->name != toname )
                useactor(toname);
            processtokens(fromname, message, tokens, false, ts_addit);
            return n_last_batch;
        }
        // likewise if we are generating we only use that kv
        int genkv=99;
        for( int tgt=0; tgt<3; tgt++ ) {
            if( kvuser[tgt] && gen_mark[tgt] != -1 ) {
                genkv = tgt;
                break;
            }
        }

        for( int tgt=0; tgt<3; tgt++ ) {
            if( genkv != 99 && genkv != tgt ) continue;

            if( kvuser[tgt] ) {
                //LLAMA_LOG_INFO("Send to %d\n", tgt);
                seq_start[tgt] -= ts_rewind;
                uint16_t seq_was = seq_start[tgt];
                useactor(kvuser[tgt]->name);
                processtokens(fromname, message, tokens, false, ts_addit); // fully process
                // we want to add to a buffer if this is a gen_mark sequence
                /* -- we do this in processtokens now
                if( gen_mark[tgt] != -1 ) {
                    // add ts_addit tokens to size
                    LLAMA_LOG_INFO("%s: read ts_addit(%u) tokens after previous of %u\n", __func__, ts_addit, ts_prev);
                    for( int j=0; j<32; j++ ) {
                        gen_k_so_far[tgt][j].resize( gen_k_so_far[tgt][j].size() + ts_addit*1024 );
                        gen_v_so_far[tgt][j].resize( gen_v_so_far[tgt][j].size() + ts_addit*1024 );
                    }
                    // read the tokens
                    for( uint16_t k=0; k<ts_addit; k++ ) {
                        gen_tokens_so_far[tgt].push_back( tokens[ts_prev+k] );
                    }
                    kv[tgt].read2( seq_was+ts_prev, ts_prev, ts_addit, gen_k_so_far[tgt], gen_v_so_far[tgt]);
                    LLAMA_LOG_INFO("%s: done reading ts_addit\n", __func__);
                }
                */
                messaged.insert( kvuser[tgt] );
                if( gen_mark[tgt] != -1 )
                    ragunmap(kvuser[tgt], message); // search for any ragged messages in the past
            }
        }

        if( seq_mark[current_kv] != -1 || gen_mark[current_kv] != -1 ) {
            return n_last_batch; // Don't generate memories if we're going to rewind.
        }

        std::vector<System_actor*>::iterator it; // just say that we're sending it since they're not active yet


        for( it = actors.begin(); it != actors.end(); it++ ) {
            if( messaged.contains( *it ) ) continue;

            a = *it;
            if( a->name == fromname ) continue;

            System_memory *m = (System_memory*)pool_alloc(sizeof(System_memory));
            new (m) System_memory;
            m->prepare();
            m->build(fromname, message);
            LLAMA_LOG_INFO("store message for %s: %s\n", a->name.c_str(), message.c_str());
            a->addrecent(m);
        }

        return n_last_batch;
    }

    // processtokens sends a message to one (already-selected) agent

    Kv_mem *processtokens(std::string fromname, std::string message,
                          std::vector<llama_token> &tokens, bool iskey=false, uint16_t ts_addit=0, bool force_encode=false)
    {
        size_t i;
        int n_batch = 64;
        uint16_t startpt;
        uint16_t ts_prev = tokens.size();
        Kv_mem *mem;
        System_eidet *eid;

        if( ts_prev == 0 && ts_addit == 0 ) {
            llama_quick_tokenize( message, tokens );
            ts_addit = tokens.size();

            if( gen_mark[current_kv] != -1 )
                gen_str_so_far[current_kv] = message;
        } else {
            ts_prev -= ts_addit;

            if( gen_mark[current_kv] != -1 ) {
                //LLAMA_LOG_INFO("%s gen: add string '%s'\n", __func__, message.c_str());
                gen_str_so_far[current_kv].append(message);
            }
        }

        uint16_t tgt_kv = current_kv;

        if( gen_mark[tgt_kv] != -1 )
            startpt = gen_mark[tgt_kv];
        else
            startpt = seq_start[tgt_kv];

        LLAMA_LOG_INFO("%s(%u): start %u, process %u (bypass %u) of %zu tokens of %s(+%s)\n", __func__, tgt_kv, startpt, ts_addit, ts_prev, tokens.size(), gen_str_so_far[tgt_kv].c_str(), message.c_str());

        for( i=ts_prev; i < tokens.size(); i += n_batch ) {
            size_t batch_end = std::min(i + n_batch, tokens.size());
            std::vector<int> teabatch(tokens.begin() + i, tokens.begin() + batch_end);

            llama_batch batch = llama_batch_init(teabatch.size(), 0, 1);
            batch.n_tokens = teabatch.size();

            for (int32_t k = 0; k < batch.n_tokens; k++) {
                batch.token[k] = teabatch[k];
            }

            if( seq_start[tgt_kv] + batch.n_tokens >= kv_extent[tgt_kv]-64 ) {
                LLAMA_LOG_INFO("reserve_space from %u (limit %u)\n", seq_start[tgt_kv], kv_extent[tgt_kv]);
                if( gen_mark[tgt_kv] != -1 ) {
                    // add ts_addit tokens to size
                    eid = (System_eidet*)pool_alloc(sizeof(System_eidet));
                    new (eid) System_eidet;
                    eid->prepare();

                    // read from current_kv and build eidet
                    eid->build(&(kv[tgt_kv]), fromname, message, gen_mark[tgt_kv], seq_start[tgt_kv]-gen_mark[tgt_kv]);
                    // add to source
                    mem = kvuser[tgt_kv]->addrecent(eid);
                    mem->is_active = true;

                    LLAMA_LOG_INFO("%s: done reading ts_addit\n", __func__);
                }
                useactor( kvuser[tgt_kv]->name, true ); // this 'true' will allocate 256 tokens, enough for 4 batches.
                startpt = seq_start[tgt_kv];
                if( gen_mark[tgt_kv] != -1 ) { // reset gen mark after rebuilding map
                    gen_mark[tgt_kv] = seq_start[tgt_kv];
                }
            } else {
                //seq_start[tgt_kv] = startpt + tokens.size() - ts_addit;
            }
            current_context->sequential_start = current_context->seq_end = seq_start[tgt_kv];
            int res = llama_decode(current_context, batch);
            seq_start[tgt_kv] += batch.n_tokens;

            llama_batch_free(batch);
        }

        if( !force_encode && ( seq_mark[tgt_kv] != -1 || gen_mark[tgt_kv] != -1 ) ) {
            //LLAMA_LOG_INFO("[skip adding to recent memories for %d]\n", tgt_kv);
            return NULL; // Don't generate memories if we're going to rewind.
        }

        eid = (System_eidet*)pool_alloc(sizeof(System_eidet));
        new (eid) System_eidet;
        eid->prepare();

        // read from current_kv and build eidet
        eid->build(&(kv[tgt_kv]), fromname, message, startpt, tokens.size());
        // add to source
        if( !iskey && !force_encode ) {
            mem = kvuser[tgt_kv]->addrecent(eid);
        } else {
            mem = (Kv_mem*)pool_alloc(sizeof(Kv_mem));
            mem->prepare();
            mem->is_full = true;
            mem->is_active = true;
            mem->m = NULL;
            mem->e = eid;
        }
        mem->first = startpt;
        mem->last = startpt+tokens.size()-1;
        mem->is_active = true;

        return mem;
    }


    // processtokens sends a message to one agent
    bool processtokens_inplace(uint8_t tgt_kv, Kv_mem *mem, bool iskey=false)
    {
        size_t i;
        int n_batch = 64;

        if( mem->m->tokens.size() == 0 ) {
            llama_quick_tokenize( mem->m->what, mem->m->tokens );
        }

        mem->first = seq_start[tgt_kv];
        mem->last = mem->first + mem->m->n_tokens - 1;
        mem->m->n_tokens = mem->m->tokens.size();

        LLAMA_LOG_INFO("%s: tgt_kv=%d from=%s first=%d tokens.size()=%u\nwhat=%s\n", __func__,
                       tgt_kv, mem->m->who.c_str(), mem->first, mem->m->n_tokens, mem->m->what.c_str());

        for( i=0; i < mem->m->tokens.size(); i += n_batch ) {
            size_t batch_end = std::min(i + n_batch, mem->m->tokens.size());
            std::vector<int> teabatch(mem->m->tokens.begin() + i, mem->m->tokens.begin() + batch_end);

            llama_batch batch = llama_batch_init(teabatch.size(), 0, 1);
            batch.n_tokens = teabatch.size();

            for (int32_t k = 0; k < batch.n_tokens; k++) {
                batch.token[k] = teabatch[k];
            }

            if( seq_start[tgt_kv] + batch.n_tokens >= kv_extent[tgt_kv] ) {
                useactor( kvuser[tgt_kv]->name, true );
            }
            current_context->sequential_start = current_context->seq_end = seq_start[tgt_kv];
            int res = llama_decode(current_context, batch);
            seq_start[tgt_kv] += batch.n_tokens;

            llama_batch_free(batch);
        }

        System_eidet *eid = (System_eidet*)pool_alloc(sizeof(System_eidet));
        new (eid) System_eidet;
        eid->prepare();

        // read from current_kv and build eidet
        eid->build(&(kv[tgt_kv]), mem->m->who, mem->m->what, mem->first, mem->m->n_tokens);
        // add to source
        mem->e = eid;
        mem->is_full = true;
        mem->m->release();
        pool_free( mem->m );
        mem->m = NULL;
        mem->is_active = true;

        LLAMA_LOG_INFO("%s(%s): done\n", __func__, quick_ts().c_str());
        return true;
    }

    System_actor *getactor( std::string who )
    {
        System_actor *a;

        if( players.count(who) <= 0 ) {
            // initialize actor
            LLAMA_LOG_INFO("%s: prepare actor1 %s\n", __func__, who.c_str());
            a = (System_actor*)pool_alloc(sizeof(System_actor));
            new (a) System_actor;
            a->prepare();
            a->name = who;
            a->loadfile();
            actors.push_back(a);
            players[who] = a;
            LLAMA_LOG_INFO("%s: actor prepared3\n", __func__);

            // link the actor into the ragwordmap
            // add history to ragwordmap
            std::string srch;
            std::vector<Kv_mem*>::iterator it;
            for( it = a->history.begin(); it != a->history.end(); it++ ) {
                Kv_mem *memlink = *it;
                System_memory *m = memlink->m;
                for( auto srch : m->keywords ) {
                    std::vector<System_memory *> *x;

                    if( ragwordmap.contains(srch) ) {
                        x = ragwordmap.at(srch);
                    } else {
                        x = (std::vector<System_memory *>*)pool_alloc(sizeof(std::vector<System_memory *>));
                        new (x) std::vector<System_memory *>;
                        ragwordmap[srch]=x;
                    }
                    x->push_back( m );
                }
            }
        } else {
            a = players[who];
        }

        return a;
    }
    System_memory *remember( std::string actor, std::string who, std::string what, std::string when )
    {
        System_actor *a = current_kb->getactor(actor);
        System_memory *m = (System_memory*)pool_alloc(sizeof(System_memory));
        new (m) System_memory;
        m->prepare();

        m->who = who;
        m->what = what;
        if( when.length() == 0 ) {
            m->when = llama_ts_now();
        } else {
            m->when = llama_string_to_ts(when);
        }
        a->addhist(m);

        return m;
    }

    bool verify_map( int extent, std::vector<Kv_mem*> *map )
    {
        std::vector<Kv_mem*>::iterator it;
        int tokens=0;
        for( it = map->begin(); it != map->end(); it++ ) {
            Kv_mem *eid = *it;
            tokens += (eid->last - eid->first)+1;
        }
        int reserve_space = (extent>1024)?(400):(200);
        return !( tokens > extent-reserve_space );
    }

};

int llama_poll_vocab( std::unordered_map< std::string, int > &searchspace, float *logits )
{
    int i, j;
    llama_vocab *vocab = &(current_model->vocab);
    std::unordered_map< int, float > scores;
    std::unordered_map< int, std::string > revMap;

    for( const auto &pair : searchspace ) {
        if( revMap.find(pair.second) != revMap.end() ) {
            continue;
        }
        revMap[ pair.second ] = pair.first;
    }

    for( i=0; i<32003; i++ ) {
        std::string *link = &(vocab->id_to_token[i].text);
        if( logits[i] <= 0.5 ) continue;

        for( const auto &pair : searchspace ) {
            bool found=true;
            for( j=0; j<link->length(); j++ ) {
                if( (*link)[j] != pair.first[j] ) {
                    found=false;
                    break;
                }
            }
            if( found )
                scores[pair.second] += logits[i];
        }

    }
    int choice=-1;
    float highestScore=0;
    for( const auto &pair : scores ) {
        LLAMA_LOG_INFO("%s: score for [%s]: [%f]\n", __func__, revMap[pair.first].c_str(), pair.second);
        if( pair.second > highestScore ) {
            highestScore = pair.second;
            choice = pair.first;
        }
    }

    return choice;
}

void llama_mark_rewind( )
{
    current_kb->mark_rewind();
}
void llama_rewind_to_mark( )
{
    current_kb->rewind_to_mark();
}
void llama_mark_generation( std::string author )
{
    current_kb->mark_generation(author);
}
void llama_rewind_generation( std::string message, std::vector<int> &tokens )
{
    current_kb->rewind_generation(message, tokens);
}
void llama_query_actor_names(std::vector<std::string> &names)
{
    current_kb->query_actor_names(names);
}

Kv_mem *new_kv_mem( void )
{
    Kv_mem *m = current_kb->getmem();
    m->m = NULL;
    m->e = NULL;
    m->is_full = false;
    m->is_active = false;
    m->first = m->last = 0;
    return m;
}
Kv_mem *new_kv_mem( System_memory *memory )
{
    Kv_mem *m = current_kb->getmem();
    m->m = memory;
    m->e = NULL;
    m->is_full = false;
    m->is_active = false;
    m->first = m->last = 0;
    return m;
}
Kv_mem *new_kv_mem( System_eidet *memory )
{
    Kv_mem *m = current_kb->getmem();
    m->e = memory;
    m->m = NULL;
    m->is_full = true;
    m->first = m->last = 0;
    return m;
}

void llama_save_actors( void )
{
    current_kb->saveall();
}

void llama_unload_actor( std::string actor )
{
    current_kb->unload(actor);
}

void llama_record_memory( std::string actor, std::string who, std::string when, std::string what )
{
    System_memory *m = current_kb->remember(actor,who,what,when);
    // return m;
}


const char * llama_context_charname( llama_context *ctx )
{
    return ctx->charname.c_str();
}

const char * llama_model_impressions( llama_model *mdl )
{
    return mdl->impressionspath.c_str();
}

void llama_context_setname( llama_model *model, llama_context *ctx, std::string s )
{
    if( ctx->charname.compare(s) != 0 ) {
        ctx->charname = s;
        //llama_context_loadchar(model,ctx);
    }
}

void llama_context_save_vocab( llama_context *ctx )
{
    /*
    if( ctx->charname.length() <= 0 ) return;

    const llama_model *model = &ctx->model;

    std::string filename = model->impressionspath + "\\" + ctx->charname + "\\vocabs.dat";
    FILE *fp = fopen(filename.c_str(), "w");
    int i, sz;
    for( i = 0; i< ctx->write_token; i++ ) {
        sz = model->vocab.id_to_token[i+32002].text.length();
        fwrite( &sz, sizeof(int), 1, fp );
        fwrite( model->vocab.id_to_token[i+32002].text.c_str(), sizeof(char), sz, fp );
        fwrite( ctx->mvocab[i], sizeof(float), 4096, fp );
    }
    fclose(fp);*/
}
void llama_context_add_vocab( llama_context *ctx, char *vocab_name, float *embd )
{
    /*
    if( ctx->charname.length() <= 0 ) return;

    const llama_model *model = &ctx->model;

    return; // no more haha

    int n = ctx->write_token;
    ctx->write_token++;
    LLAMA_LOG_INFO("%s: write at token %d\n", __func__, n);

    std::string filename = model->impressionspath + "\\" + ctx->charname + "\\vocabs.dat";
    FILE *fp = fopen(filename.c_str(), "r");
    if( fp ) {
        fclose(fp);
        fp = fopen(filename.c_str(), "rb+");
        LLAMA_LOG_INFO("%s: seek eof\n", __func__);
        fseek(fp, 0, SEEK_END);
    } else {
        fp = fopen(filename.c_str(),  "wb");
    }

    LLAMA_LOG_INFO("%s: write to file %d\n", __func__, fp?1:0);

    bool new_name=false;
    if( !vocab_name ) {
        std::string vw = std::to_string(n);
        while( vw.length() < 5 ) {
            vw = "0" + vw;
        }
        vw = "!z" + vw;
        vocab_name = (char*)malloc(vw.length()+1);
        strcpy( vocab_name, vw.c_str() );
        new_name = true;
    }
    std::string vn(vocab_name);
    auto token_data = model->vocab.id_to_token.at(n+32002);
    token_data.text = std::move(vn);

    //LLAMA_LOG_INFO("%s: writing mvocab\n", __func__);
    memcpy( ctx->mvocab[n], embd, 4096*sizeof(float) );

    //model->vocab.token_to_id[(llama_vocab::token)vn] = n+32002;

    int tdrowsize=4096/32;
    block_q4_0 *td = (block_q4_0*)ctx->tok_embd->data;
    //LLAMA_LOG_INFO("%s: quantizing %s\n", __func__, vocab_name);
    quantize_row_q4_0( embd, td + ((32002+n)*tdrowsize), 4096 );

    //LLAMA_LOG_INFO("%s: fileseek\n", __func__);

    int sz = strlen(vocab_name);
    fwrite( &sz, sizeof(int), 1, fp );
    fwrite( vocab_name, sizeof(char), sz, fp );
    fwrite( embd, sizeof(float), 4096, fp );

    fflush( fp );

    //LLAMA_LOG_INFO("%s: fileclose\n", __func__);
    fclose( fp );

    if( new_name ) {
        free(vocab_name);
    }
    LLAMA_LOG_INFO("%s: added 0 past tokens\n", __func__);*/
}

void llama_toggle_full( llama_context *ctx, int value)
{
    ctx->record_all = value;
}

void llama_context_recall( llama_context *ctx )
{
    /*
    if( ctx->charname.length() <= 0 ) return;

    const llama_model *model = &ctx->model;

    if( ctx->write_token <= ctx->past_write_token ) {
        LLAMA_LOG_INFO("%s: adding 0 past tokens\n", __func__);
        return;
    }
    LLAMA_LOG_INFO("%s: adding %d,%d past tokens\n", __func__, ctx->write_token, ctx->past_write_token);

    llama_batch batch = llama_batch_init(ctx->write_token - ctx->past_write_token, 0, 1);

    batch.n_tokens = ctx->write_token - ctx->past_write_token;
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        batch.token   [i] = i+ctx->past_write_token + 32002;
        batch.pos     [i] = 4096+i+ctx->past_write_token;
        batch.n_seq_id[i] = 1;
        batch.seq_id  [i][0] = 0;
        //batch.logits  [i] = false;
    }

    // llama_decode will output logits only for the last token of the prompt
    //batch.logits[batch.n_tokens - 1] = true;

    LLAMA_LOG_INFO("%s: process past tokens\n", __func__);
    //int old_pos = ctx->
    int res = llama_decode(ctx, batch);
    LLAMA_LOG_INFO("%s: process done %d\n", __func__, res);
    llama_batch_free(batch);
    ctx->past_write_token = ctx->write_token;
    */
}

void llama_context_loadchar( llama_model *model, llama_context *ctx )
{
    /*
    std::filesystem::path ipath(model->impressionspath + "\\" + ctx->charname);
    std::string model_augment_path = model->impressionspath + "\\" + ctx->charname + "\\vocabs.dat";

    if( !std::filesystem::exists(ipath) ) {
        LLAMA_LOG_INFO("%s: creating dir %s\n", __func__, ipath.string().c_str());
        std::filesystem::create_directory(ipath);
    }

    return; // don't do it anymroe lol

    FILE *fp = NULL;
    LLAMA_LOG_INFO("%s: load augment %s\n", __func__, model_augment_path.c_str());

    int rowsize = 4096;
    int tdrowsize=rowsize/32;
    int sz, v;
    char buf[128];
    block_q4_0 *td = (block_q4_0*)ctx->tok_embd->data;
    //fp = fopen(model_augment_path.c_str(), "rb");

    if( fp ) {
        LLAMA_LOG_INFO("%s: reading vocabs databuffer\n", __func__);

        float *databuffer = (float*)malloc(sizeof(float)*rowsize);
        int n = 0;
        while( true ) {
            v = fread( &sz, sizeof(int), 1, fp );
            if( v == 0 ) {
                break;
            }
            fread( buf, sizeof(char), sz, fp );
            buf[sz] = 0;
            fread( databuffer, sizeof(float), rowsize, fp );

            LLAMA_LOG_INFO("%s: read key %s\n", __func__, buf);

            std::string vn(buf);
            auto token_data = model->vocab.id_to_token.at(n+32002);
            quantize_row_q4_0( databuffer, td + ((32002+n)*tdrowsize), rowsize );
            memcpy( ctx->mvocab[n], databuffer, sizeof(float)*rowsize );
            token_data.text = vn;
            n++;
        }
        free(databuffer);

        fclose(fp);
        ctx->write_token = n;
        llama_context_recall(ctx);
    }

    LLAMA_LOG_INFO("%s: done, %d tokens\n", __func__, ctx->write_token);
*/
}




//
// kv cache helpers
//

static bool llama_kv_cache_init(
             struct llama_kv_cache & cache,
                 const llama_model & model,
                         ggml_type   type_k,
                         ggml_type   type_v,
                          uint32_t   n_ctx,
                              bool   offload) {
    const struct llama_hparams & hparams = model.hparams;

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();
    const int64_t  n_layer      = hparams.n_layer;


    cache.size = n_ctx;

    cache.type_k = type_k;
    cache.type_v = type_v;

#ifdef GGML_USE_CLBLAST
    offload = false;
#endif

    // count used buffer types
    std::map<ggml_backend_buffer_type_t, int> buft_layer_count;
    if (offload) {
        for (int64_t i = 0; i < n_layer; ++i) {
            buft_layer_count[model.buft_layer[i].buft]++;
        }
    } else {
        buft_layer_count[llama_default_buffer_type_cpu(true)] = n_layer;
    }

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    for (auto & it : buft_layer_count) {
        int n_layers = it.second;
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2u*n_layers*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for kv cache\n", __func__);
            return false;
        }
        ctx_map[it.first] = ctx;
        cache.ctxs.push_back(ctx);
    }

    cache.k_l.reserve(n_layer);
    cache.v_l.reserve(n_layer);

    for (int i = 0; i < (int) n_layer; i++) {
        struct ggml_context * ctx = offload ? ctx_map.at(model.buft_layer[i].buft) : cache.ctxs.front();
        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*cache.size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*cache.size);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for kv cache\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        cache.bufs.push_back(buf);
    }

    return true;
}

//
// model loading and saving
//

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

static const char * llama_file_version_name(llama_fver version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1: return "GGUF V1 (support until nov 2023)";
        case GGUF_FILE_VERSION_V2: return "GGUF V2";
        case GGUF_FILE_VERSION_V3: return "GGUF V3 (latest)";
    }

    return "unknown";
}

static std::string llama_format_tensor_shape(const std::vector<int64_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, ne.at(i));
    }
    return buf;
}

static std::string llama_format_tensor_shape(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, t->ne[i]);
    }
    return buf;
}

static std::string llama_format_tensor_blocks(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5zu", t->nb[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5zu", t->nb[i]);
    }
    return buf;
}

namespace GGUFMeta {
    template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int)>
    struct GKV_Base_Type {
        static constexpr gguf_type gt = gt_;

        static T getter(const gguf_context * ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T> struct GKV_Base;

    template<> struct GKV_Base<bool        >: GKV_Base_Type<bool,         GGUF_TYPE_BOOL,    gguf_get_val_bool> {};
    template<> struct GKV_Base<uint8_t     >: GKV_Base_Type<uint8_t,      GGUF_TYPE_UINT8,   gguf_get_val_u8  > {};
    template<> struct GKV_Base<uint16_t    >: GKV_Base_Type<uint16_t,     GGUF_TYPE_UINT16,  gguf_get_val_u16 > {};
    template<> struct GKV_Base<uint32_t    >: GKV_Base_Type<uint32_t,     GGUF_TYPE_UINT32,  gguf_get_val_u32 > {};
    template<> struct GKV_Base<uint64_t    >: GKV_Base_Type<uint64_t,     GGUF_TYPE_UINT64,  gguf_get_val_u64 > {};
    template<> struct GKV_Base<int8_t      >: GKV_Base_Type<int8_t,       GGUF_TYPE_INT8,    gguf_get_val_i8  > {};
    template<> struct GKV_Base<int16_t     >: GKV_Base_Type<int16_t,      GGUF_TYPE_INT16,   gguf_get_val_i16 > {};
    template<> struct GKV_Base<int32_t     >: GKV_Base_Type<int32_t,      GGUF_TYPE_INT32,   gguf_get_val_i32 > {};
    template<> struct GKV_Base<int64_t     >: GKV_Base_Type<int64_t,      GGUF_TYPE_INT64,   gguf_get_val_i64 > {};
    template<> struct GKV_Base<float       >: GKV_Base_Type<float,        GGUF_TYPE_FLOAT32, gguf_get_val_f32 > {};
    template<> struct GKV_Base<double      >: GKV_Base_Type<double,       GGUF_TYPE_FLOAT64, gguf_get_val_f64 > {};
    template<> struct GKV_Base<const char *>: GKV_Base_Type<const char *, GGUF_TYPE_STRING,  gguf_get_val_str > {};

    template<> struct GKV_Base<std::string> {
        static constexpr gguf_type gt = GGUF_TYPE_STRING;

        static std::string getter(const gguf_context * ctx, const int kid) {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const gguf_type gt;
        const size_t length;
        const void * data;
    };

    template<> struct GKV_Base<ArrayInfo> {
        public:
        static constexpr gguf_type gt = GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const gguf_context *ctx, const int k) {
            return ArrayInfo {
                gguf_get_arr_type(ctx, k),
                size_t(gguf_get_arr_n(ctx, k)),
                gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T> {
        GKV() = delete;

        public:
        static T get_kv(const gguf_context * ctx, const int k) {
            const enum gguf_type kt = gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt) {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                    gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char * override_type_to_str(const llama_model_kv_override_type ty) {
            switch (ty) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  return "bool";
                case LLAMA_KV_OVERRIDE_TYPE_INT:   return "int";
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: return "float";
            }
            return "unknown";
        }

        static bool validate_override(const llama_model_kv_override_type expected_type, const struct llama_model_kv_override * ovrd) {
            if (!ovrd) { return false; }
            if (ovrd->tag == expected_type) {
                LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ",
                    __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag) {
                    case LLAMA_KV_OVERRIDE_TYPE_BOOL:  {
                        LLAMA_LOG_INFO("%s\n", ovrd->bool_value ? "true" : "false");
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_INT:   {
                        LLAMA_LOG_INFO("%" PRId64 "\n", ovrd->int_value);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                        LLAMA_LOG_INFO("%.6f\n", ovrd->float_value);
                    } break;
                    default:
                        // Shouldn't be possible to end up here, but just in case...
                        throw std::runtime_error(
                            format("Unsupported attempt to override %s type for metadata key %s\n",
                                override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
                target = ovrd->bool_value;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                target = ovrd->int_value;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
                target = ovrd->float_value;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            (void)target;
            (void)ovrd;
            if (!ovrd) { return false; }
            // Currently, we should never end up here so it would be a bug if we do.
            throw std::runtime_error(format("Unsupported attempt to override string type for metadata key %s\n",
                ovrd ? ovrd->key : "NULL"));
        }

        static bool set(const gguf_context * ctx, const int k, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            if (try_override<T>(target, ovrd)) {
                return true;
            }
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const gguf_context * ctx, const char * key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const gguf_context * ctx, const std::string & key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

struct llama_model_loader {
    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    int64_t n_elements = 0;
    size_t  n_bytes    = 0;

    bool use_mmap = false;

    llama_file  file;
    llama_ftype ftype;
    llama_fver  fver;

    std::unique_ptr<llama_mmap> mapping;
    std::unordered_map<std::string, struct llama_model_kv_override> kv_overrides;

    struct gguf_context * ctx_gguf = NULL;
    struct ggml_context * ctx_meta = NULL;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    llama_model_loader(const std::string & fname, bool use_mmap, const struct llama_model_kv_override * param_overrides_p) : file(fname.c_str(), "rb") {
        int trace = 0;
        if (getenv("LLAMA_TRACE")) {
            trace = atoi(getenv("LLAMA_TRACE"));
        }

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx_meta,
        };

        if (param_overrides_p != nullptr) {
            for (const struct llama_model_kv_override *p = param_overrides_p; p->key[0] != 0; p++) {
                kv_overrides.insert({std::string(p->key), *p});
            }
        }

        ctx_gguf = gguf_init_from_file(fname.c_str(), params);
        if (!ctx_gguf) {
            throw std::runtime_error(format("%s: failed loading model from %s\n", __func__, fname.c_str()));
        }

        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));

        n_kv      = gguf_get_n_kv(ctx_gguf);
        n_tensors = gguf_get_n_tensors(ctx_gguf);

        fver = (enum llama_fver ) gguf_get_version(ctx_gguf);

        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            n_elements += ggml_nelements(t);
            n_bytes    += ggml_nbytes(t);
        }

        LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
                __func__, n_kv, n_tensors, fname.c_str(), llama_file_version_name(fver));

        // determine file type based on the number of tensors for each quantization and print meta data
        // TODO: make optional
        {
            std::map<enum ggml_type, uint32_t> n_type;

            uint32_t n_type_max = 0;
            enum ggml_type type_max = GGML_TYPE_F32;

            for (int i = 0; i < n_tensors; i++) {
                enum ggml_type type = gguf_get_tensor_type(ctx_gguf, i);

                n_type[type]++;

                if (n_type_max < n_type[type]) {
                    n_type_max = n_type[type];
                    type_max   = type;
                }

                if (trace > 0) {
                    struct ggml_tensor * meta = ggml_get_tensor(ctx_meta, gguf_get_tensor_name(ctx_gguf, i));
                    LLAMA_LOG_INFO("%s: - tensor %4d: %32s %-8s ne [ %s ] nb [ %s ] %d\n", __func__, i, ggml_get_name(meta), ggml_type_name(type),
                                   llama_format_tensor_shape(meta).c_str(),
                                   llama_format_tensor_blocks(meta).c_str(),
                                   ggml_nbytes(meta));
                }
            }

            switch (type_max) {
                case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
                case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
                case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
                case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
                case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
                case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
                case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
                case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
                case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
                case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
                case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
                case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
                case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
                case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
                case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_S;   break;
                case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
                case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
                case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
                case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
                case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
                default:
                    {
                        LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                        ftype = LLAMA_FTYPE_ALL_F32;
                    } break;
            }

            // this is a way to mark that we have "guessed" the file type
            ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

            {
                const int kid = gguf_find_key(ctx_gguf, "general.file_type");
                if (kid >= 0) {
                    ftype = (llama_ftype) gguf_get_val_u32(ctx_gguf, kid);
                }
            }

            LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);
            for (int i = 0; i < n_kv; i++) {
                const char * name           = gguf_get_key(ctx_gguf, i);
                const enum gguf_type type   = gguf_get_kv_type(ctx_gguf, i);
                const std::string type_name =
                    type == GGUF_TYPE_ARRAY
                    ? format("%s[%s,%d]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(ctx_gguf, i)), gguf_get_arr_n(ctx_gguf, i))
                    : gguf_type_name(type);

                std::string value          = gguf_kv_to_str(ctx_gguf, i);
                const size_t MAX_VALUE_LEN = 40;
                if (value.size() > MAX_VALUE_LEN) {
                    value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
                }
                replace_all(value, "\n", "\\n");

                LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
            }

            // print type counts
            for (auto & kv : n_type) {
                if (kv.second == 0) {
                    continue;
                }

                LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
            }
        }

        if (!llama_mmap::SUPPORTED) {
            LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
            use_mmap = false;
        }

        this->use_mmap = use_mmap;
    }

    ~llama_model_loader() {
        if (ctx_gguf) {
            gguf_free(ctx_gguf);
        }
        if (ctx_meta) {
            ggml_free(ctx_meta);
        }
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, const bool required = true) {
        const int kid = gguf_find_key(ctx_gguf, key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(ctx_gguf, kid);


        result = arr_info.length;
        return true;
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const enum llm_kv kid, T & result, const bool required = true) {
        return get_arr_n(llm_kv(kid), result, required);
    }

    template<typename T>
    bool get_key(const std::string & key, T & result, const bool required = true) {
        auto it = kv_overrides.find(key);

        const struct llama_model_kv_override * override =
            it != kv_overrides.end() ? &it->second : nullptr;

        const bool found = GGUFMeta::GKV<T>::set(ctx_gguf, key, result, override);

        if (required && !found) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }

        return found;
    }

    template<typename T>
    bool get_key(const enum llm_kv kid, T & result, const bool required = true) {
        return get_key(llm_kv(kid), result, required);
    }

    std::string get_arch_name() const {
        return arch_name;
    }

    enum llm_arch get_arch() const {
        return llm_kv.arch;
    }

    const char * get_tensor_name(int i) const {
        return gguf_get_tensor_name(ctx_gguf, i);
    }

    struct ggml_tensor * get_tensor_meta(const char * name) const {
        return ggml_get_tensor(ctx_meta, name);
    }

    struct ggml_tensor * get_tensor_meta(int i) const {
        return get_tensor_meta(get_tensor_name(i));
    }

    struct ggml_tensor * create_tensor_for(struct ggml_context * ctx, struct ggml_tensor * meta) {
        struct ggml_tensor * tensor = ggml_dup_tensor(ctx, meta);
        ggml_set_name(tensor, ggml_get_name(meta));

        n_created++;

        return tensor;
    }

    struct ggml_tensor ** create_tensor_copies(struct ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne, int nc) {
        struct ggml_tensor **curs = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *)*nc);
        for( int i=0; i<nc; i++ ) {
            curs[i] = create_tensor(ctx,name,ne);
        }
        return curs;
    }

    struct ggml_tensor * create_tensor(struct ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne, bool required = true) {
        struct ggml_tensor * cur = ggml_get_tensor(ctx_meta, name.c_str());

        if (cur == NULL) {
            if (!required) {
                return NULL;
            }
            throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
        }

        {
            bool is_ok = true;
            for (size_t i = 0; i < ne.size(); ++i) {
                if (ne[i] != cur->ne[i]) {
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok) {
                throw std::runtime_error(
                        format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                            __func__, name.c_str(),
                            llama_format_tensor_shape(ne).c_str(),
                            llama_format_tensor_shape(cur).c_str()));
            }
        }

        return create_tensor_for(ctx, cur);
    }

    void done_getting_tensors() const {
        if (n_created != n_tensors) {
            throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
        }
    }

    size_t file_offset(const char * name) const {
        const int idx = gguf_find_tensor(ctx_gguf, name);

        if (idx < 0) {
            return gguf_get_data_offset(ctx_gguf);
        }

        return gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, idx);
    }

    void init_mapping(bool prefetch = true, llama_mlock * lmlock = nullptr) {
        // prefetch the whole file - all the data is needed anyway
        if (use_mmap) {
            mapping.reset(new llama_mmap(&file, prefetch ? -1 : 0, ggml_is_numa()));
        }

        // compute the total size of all tensors for progress reporting
        for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); i++) {
            struct ggml_tensor * cur = ggml_get_tensor(ctx_meta, gguf_get_tensor_name(ctx_gguf, i));
            size_data += ggml_nbytes(cur);
        }

        if (use_mmap && mapping) {
            if (lmlock) {
                lmlock->init(mapping->addr);
            }
            mmap_used_first = mapping->size;
        }
    }

    void get_mapping_range(size_t * first, size_t * last, ggml_context * ctx) const {
        GGML_ASSERT(mapping);

        *first = mapping->size;
        *last  = 0;
        for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            const size_t offs = file_offset(ggml_get_name(tensor));
            *first = std::min(*first, offs);
            *last  = std::max(*last,  offs + ggml_nbytes(tensor));
        }
    }

    // for backwards compatibility, does not support ggml-backend
    void load_data_for(struct ggml_tensor * cur) const {
        const size_t offs = file_offset(ggml_get_name(cur));

        if (use_mmap && mapping) {
            if (cur->data == nullptr) {
                cur->data = (uint8_t *)mapping->addr + offs;
            } else {
                memcpy(cur->data, (uint8_t *)mapping->addr + offs, ggml_nbytes(cur));
            }
        } else {
            GGML_ASSERT(cur->data != nullptr);
            file.seek(offs, SEEK_SET);
            file.read_raw(cur->data, ggml_nbytes(cur));
        }
    }

    size_t size_done = 0;
    size_t size_data = 0;
    size_t mmap_used_first = -1;
    size_t mmap_used_last  = 0;

    // Returns false if cancelled by progress_callback
    bool load_all_data(struct ggml_context * ctx, llama_progress_callback progress_callback, void * progress_callback_user_data, ggml_backend_buffer_t buf_mmap, llama_mlock * lmlock) {
        GGML_ASSERT(size_data != 0 && "call init_mapping() first");

        std::vector<no_init<uint8_t>> read_buf;


        LLAMA_LOG_INFO("%s: read context\n", __func__);

        for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
            //LLAMA_LOG_INFO("%s: read tensor %s\n", __func__, cur->name);
            if (progress_callback) {
                if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                    return false;
                }
            }

            const char *cname = ggml_get_name(cur);
            if( strcmp(cname, "output_embd") == 0 ) {
                continue;
            }
            const size_t offs = file_offset(cname);

            if (use_mmap && mapping) {
                if (buf_mmap && cur->data == nullptr) {
                    ggml_backend_tensor_alloc(buf_mmap, cur, (uint8_t *) mapping->addr + offs);
                    if (lmlock) {
                        lmlock->grow_to(offs + ggml_nbytes(cur));
                    }
                    mmap_used_first = std::min(mmap_used_first, offs);
                    mmap_used_last  = std::max(mmap_used_last,  offs + ggml_nbytes(cur));
                } else {
                    ggml_backend_tensor_set(cur, (uint8_t *) mapping->addr + offs, 0, ggml_nbytes(cur));
                }
            } else {
                if (ggml_backend_buffer_is_host(cur->buffer)) {
                    file.seek(offs, SEEK_SET);
                    file.read_raw(cur->data, ggml_nbytes(cur));
                } else {
                    read_buf.resize(ggml_nbytes(cur));
                    file.seek(offs, SEEK_SET);
                    file.read_raw(read_buf.data(), ggml_nbytes(cur));
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, ggml_nbytes(cur));
                }
            }

            size_done += ggml_nbytes(cur);
        }

        // check if this is the last call and do final cleanup
        if (size_done >= size_data) {
            // unmap offloaded tensors and metadata
            if (use_mmap && mapping) {
                mapping->unmap_fragment(0, mmap_used_first);
                if (mmap_used_last != 0) {
                    mapping->unmap_fragment(mmap_used_last, mapping->size);
                }
            }
            if (progress_callback) {
                // Even though the model is done loading, we still honor
                // cancellation since we need to free allocations.
                return progress_callback(1.0f, progress_callback_user_data);
            }
        }

        return true;
    }
};

template<>
bool llama_model_loader::get_key(const enum llm_kv kid, enum llama_pooling_type & result, const bool required) {
    uint32_t tmp;
    const bool found = get_key(kid, tmp, required);
    if (found) {
        result = (enum llama_pooling_type) tmp;
    } else {
        result = LLAMA_POOLING_TYPE_UNSPECIFIED;
    }
    return found;
}


//
// load LLaMA models
//

static const char * llama_model_arch_name(llm_arch arch) {
    auto it = LLM_ARCH_NAMES.find(arch);
    if (it == LLM_ARCH_NAMES.end()) {
        return "unknown";
    }
    return it->second;
}

static std::string llama_model_ftype_name(llama_ftype ftype) {
    if (ftype & LLAMA_FTYPE_GUESSED) {
        return llama_model_ftype_name((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
    }

    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:     return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:  return "F16";
        case LLAMA_FTYPE_MOSTLY_Q4_0: return "Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1: return "Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16:
                                      return "Q4_1, some F16";
        case LLAMA_FTYPE_MOSTLY_Q5_0: return "Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1: return "Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0: return "Q8_0";

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K:   return "Q2_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q2_K_S: return "Q2_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S: return "Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M: return "Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L: return "Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S: return "Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: return "Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S: return "Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M: return "Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:   return "Q6_K";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:return "IQ2_XXS - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS: return "IQ2_XS - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_S:  return "IQ2_S - 2.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M:  return "IQ2_M - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XS: return "IQ3_XS - 3.3 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:return "IQ3_XXS - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S  :return "IQ1_S - 1.5625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL: return "IQ4_NL - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS: return "IQ4_XS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_S:  return "IQ3_S - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_M:  return "IQ3_S mix - 3.66 bpw";

        default: return "unknown, may not work";
    }
}

static const char * llama_model_type_name(e_model type) {
    switch (type) {
        case MODEL_22M:    return "22M";
        case MODEL_33M:    return "33M";
        case MODEL_109M:   return "109M";
        case MODEL_137M:   return "137M";
        case MODEL_0_5B:   return "0.5B";
        case MODEL_1B:     return "1B";
        case MODEL_2B:     return "2B";
        case MODEL_3B:     return "3B";
        case MODEL_7B:     return "7B";
        case MODEL_8B:     return "8B";
        case MODEL_13B:    return "13B";
        case MODEL_14B:    return "14B";
        case MODEL_15B:    return "15B";
        case MODEL_20B:    return "20B";
        case MODEL_30B:    return "30B";
        case MODEL_34B:    return "34B";
        case MODEL_40B:    return "40B";
        case MODEL_65B:    return "65B";
        case MODEL_70B:    return "70B";
        case MODEL_SMALL:  return "0.1B";
        case MODEL_MEDIUM: return "0.4B";
        case MODEL_LARGE:  return "0.8B";
        case MODEL_XL:     return "1.5B";
        default:           return "?B";
    }
}

static const char * llama_model_vocab_type_name(enum llama_vocab_type type){
    switch (type) {
        case LLAMA_VOCAB_TYPE_SPM: return "SPM";
        case LLAMA_VOCAB_TYPE_BPE: return "BPE";
        case LLAMA_VOCAB_TYPE_WPM: return "WPM";
        default:                   return "unknown";
    }
}

static void llm_load_arch(llama_model_loader & ml, llama_model & model) {
    model.arch = ml.get_arch();
    if (model.arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}

static void llm_load_hparams(
        llama_model_loader & ml,
        llama_model & model) {
    auto & hparams = model.hparams;
    const gguf_context * ctx = ml.ctx_gguf;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        model.gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, model.name, false);

    // get hparams kv
    ml.get_arr_n(LLM_KV_TOKENIZER_LIST,       hparams.n_vocab);
    ml.get_key  (LLM_KV_CONTEXT_LENGTH,       hparams.n_ctx_train);
    ml.get_key  (LLM_KV_EMBEDDING_LENGTH,     hparams.n_embd);
    ml.get_key  (LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff);
    ml.get_key  (LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head);
    ml.get_key  (LLM_KV_BLOCK_COUNT,          hparams.n_layer);
    ml.get_key  (LLM_KV_EXPERT_COUNT,         hparams.n_expert,      false);
    ml.get_key  (LLM_KV_EXPERT_USED_COUNT,    hparams.n_expert_used, false);

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
    }

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv = hparams.n_head;
    ml.get_key(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_yarn_orig_ctx = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_yarn_orig_ctx, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    // sanity check for n_rot (optional)
    {
        hparams.n_rot = hparams.n_embd / hparams.n_head;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

        if (model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_FALCON) {
            if (hparams.n_rot != hparams.n_embd / hparams.n_head) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd / hparams.n_head));
            }
        }
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim
    }

    hparams.n_embd_head_k = hparams.n_embd / hparams.n_head;
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);

    hparams.n_embd_head_v = hparams.n_embd / hparams.n_head;
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);

    // arch-specific KVs
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 22: model.type = e_model::MODEL_1B; break;
                    case 26: model.type = e_model::MODEL_3B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    case 48: model.type = e_model::MODEL_34B; break;
                    case 60: model.type = e_model::MODEL_30B; break;
                    case 80: model.type = hparams.n_head == hparams.n_head_kv ? e_model::MODEL_65B : e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINICPM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_2B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_FALCON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 60: model.type = e_model::MODEL_40B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                if (model.type == e_model::MODEL_13B) {
                    // TODO: become GGUF KV parameter
                    hparams.f_max_alibi_bias = 8.0f;
                }
            } break;
        case LLM_ARCH_STARCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 36: model.type = e_model::MODEL_3B; break;
                    case 42: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PERSIMMON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 36: model.type = e_model::MODEL_8B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_REFACT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_1B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 3:
                        model.type = e_model::MODEL_17M; break; // bge-micro
                    case 6:
                        model.type = e_model::MODEL_22M; break; // MiniLM-L6
                    case 12:
                        switch (hparams.n_embd) {
                            case 384: model.type = e_model::MODEL_33M; break; // MiniLM-L12, bge-small
                            case 768: model.type = e_model::MODEL_109M; break; // bge-base
                        } break;
                    case 24:
                        model.type = e_model::MODEL_335M; break; // bge-large
                }
            } break;
        case LLM_ARCH_NOMIC_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type);

                if (hparams.n_layer == 12 && hparams.n_embd == 768) {
                    model.type = e_model::MODEL_137M;
                }
            } break;
        case LLM_ARCH_BLOOM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 30:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            case 4096: model.type = e_model::MODEL_7B; break;
                        } break;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_MPT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv, false);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_30B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STABLELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_QWEN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_5B : e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = hparams.n_head == 20 ? e_model::MODEL_4B : e_model::MODEL_13B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PHI2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PLAMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GPT2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 12: model.type = e_model::MODEL_SMALL; break;
                    case 24: model.type = e_model::MODEL_MEDIUM; break;
                    case 36: model.type = e_model::MODEL_LARGE; break;
                    case 48: model.type = e_model::MODEL_XL; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CODESHELL:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 42: model.type = e_model::MODEL_SMALL; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ORION:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_20B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GEMMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 18: model.type = e_model::MODEL_2B; break;
                    case 28: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_STARCODER2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 30: model.type = e_model::MODEL_3B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        default: (void)0;
    }

    model.ftype = ml.ftype;

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.need_kq_pos = true;
    }

    hparams.rope_type = llama_rope_type(&model);
}

// TODO: This should probably be in llama.h
static std::vector<llama_vocab::id> llama_tokenize_internal(const llama_vocab & vocab, std::string raw_text, bool bos, bool special = false);
static llama_token llama_byte_to_token(const llama_vocab & vocab, uint8_t ch);

static void llm_load_vocab(
        llama_model_loader & ml,
        llama_model & model) {
    auto & vocab = model.vocab;

    struct gguf_context * ctx = ml.ctx_gguf;

    const auto kv = LLM_KV(model.arch);

    const int token_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_LIST).c_str());
    if (token_idx == -1) {
        throw std::runtime_error("cannot find tokenizer vocab in model file\n");
    }

    const float * scores = nullptr;
    const int score_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_SCORES).c_str());
    if (score_idx != -1) {
        scores = (const float * ) gguf_get_arr_data(ctx, score_idx);
    }

    const int * toktypes = nullptr;
    const int toktype_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str());
    if (toktype_idx != -1) {
        toktypes = (const int * ) gguf_get_arr_data(ctx, toktype_idx);
    }

    // determine vocab type
    {
        std::string tokenizer_name;

        ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_name);

        if (tokenizer_name == "llama") {
            vocab.type = LLAMA_VOCAB_TYPE_SPM;

            // default special tokens
            vocab.special_bos_id = 1;
            vocab.special_eos_id = 2;
            vocab.special_unk_id = 0;
            vocab.special_sep_id = -1;
            vocab.special_pad_id = -1;

            const int add_space_prefix_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_ADD_PREFIX).c_str());
            if (add_space_prefix_keyidx != -1) {
                vocab.add_space_prefix = gguf_get_val_bool(ctx, add_space_prefix_keyidx);
            } // The default value of add_space_prefix is true.
        } else if (tokenizer_name == "gpt2") {
            vocab.type = LLAMA_VOCAB_TYPE_BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_MERGES).c_str());
            if (merges_keyidx == -1) {
                throw std::runtime_error("cannot find tokenizer merges in model file\n");
            }

            const int n_merges = gguf_get_arr_n(ctx, merges_keyidx);

            for (int i = 0; i < n_merges; i++) {
                const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);
                GGML_ASSERT(codepoints_from_utf8(word).size() > 0);

                std::string first;
                std::string second;

                const size_t pos = word.find(' ', 1);

                if (pos != std::string::npos) {
                    first  = word.substr(0, pos);
                    second = word.substr(pos + 1);
                }

                vocab.bpe_ranks.emplace(std::make_pair(first, second), i);
            }

            // default special tokens
            vocab.special_bos_id = 11;
            vocab.special_eos_id = 11;
            vocab.special_unk_id = -1;
            vocab.special_sep_id = -1;
            vocab.special_pad_id = -1;
        } else if (tokenizer_name == "bert") {
            vocab.type = LLAMA_VOCAB_TYPE_WPM;

            // default special tokens
            vocab.special_bos_id = 101;
            vocab.special_eos_id = 102;
            vocab.special_unk_id = 100;
            vocab.special_sep_id = -1;
            vocab.special_pad_id = -1;
            vocab.add_space_prefix = false;
        } else {
            LLAMA_LOG_WARN("%s: unknown tokenizer: '%s'", __func__, tokenizer_name.c_str());
            LLAMA_LOG_WARN("%s: using default tokenizer: 'llama'", __func__);

            vocab.type = LLAMA_VOCAB_TYPE_SPM;
        }
    }

    const uint32_t n_vocab = gguf_get_arr_n(ctx, token_idx);

    vocab.id_to_token.resize(n_vocab);

    for (uint32_t i = 0; i < n_vocab; i++) {
        std::string word = gguf_get_arr_str(ctx, token_idx, i);
        GGML_ASSERT(codepoints_from_utf8(word).size() > 0);

        vocab.token_to_id[word] = i;

        auto & token_data = vocab.id_to_token[i];
        token_data.text  = std::move(word);
        token_data.score = scores ? scores[i] : 0.0f;
        token_data.type  = toktypes ? (llama_token_type) toktypes[i] : LLAMA_TOKEN_TYPE_NORMAL;
    }
    /*
    for( int i = 0; i < 20000; i++ ) {
        std::string vw = std::to_string(i);
        while( vw.length() < 5 ) {
            vw = "0" + vw;
        }
        vw = "!z" + vw;

        vocab.token_to_id[vw] = i+n_vocab;

        auto & token_data = vocab.id_to_token[i+n_vocab];
        token_data.text = vw;
        token_data.score = 0.0f;
        token_data.type = LLAMA_TOKEN_TYPE_CONTROL;
    }
    */
    GGML_ASSERT(vocab.id_to_token.size() == vocab.token_to_id.size());

    // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if (vocab.type == LLAMA_VOCAB_TYPE_SPM) {
        try {
            vocab.linefeed_id = llama_byte_to_token(vocab, '\n');
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.", __func__, e.what());
            vocab.linefeed_id = vocab.special_pad_id;
        }
    } else if (vocab.type == LLAMA_VOCAB_TYPE_WPM) {
        vocab.linefeed_id = vocab.special_pad_id;
    } else {
        const std::vector<int> ids = llama_tokenize_internal(vocab, "\xC4\x8A", false); // U+010A
        GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        vocab.linefeed_id = ids[0];
    }

    // special tokens
    {
        const std::vector<std::pair<enum llm_kv, int32_t &>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID, vocab.special_bos_id },
            { LLM_KV_TOKENIZER_EOS_ID, vocab.special_eos_id },
            { LLM_KV_TOKENIZER_UNK_ID, vocab.special_unk_id },
            { LLM_KV_TOKENIZER_SEP_ID, vocab.special_sep_id },
            { LLM_KV_TOKENIZER_PAD_ID, vocab.special_pad_id },
        };
        for (const auto & it : special_token_types) {
            const std::string & key = kv(std::get<0>(it));
            int32_t & id = std::get<1>(it);

            uint32_t new_id;
            if (!ml.get_key(std::get<0>(it), new_id, false)) {
                continue;
            }
            if (new_id >= vocab.id_to_token.size()) {
                LLAMA_LOG_WARN("%s: bad special token: '%s' = %ud, using default id %d\n",
                    __func__, key.c_str(), new_id, id);
            } else {
                if( key.compare("tokenizer.ggml.bos_token_id") == 0 ) {
                    new_id = 32001; //SP
                }
                id = new_id;
                LLAMA_LOG_INFO("%s: record SP special token '%s' = %ud\n",
                    __func__, key.c_str(), new_id);
            }

        }

        // Handle add_bos_token and add_eos_token
        {
            bool temp = true;

            if (ml.get_key(LLM_KV_TOKENIZER_ADD_BOS, temp, false)) {
                vocab.special_add_bos = int(temp);
            }
            if (ml.get_key(LLM_KV_TOKENIZER_ADD_EOS, temp, false)) {
                vocab.special_add_eos = int(temp);
            }
        }
    }

    // build special tokens cache
    {
        // TODO: It is unclear (to me) at this point, whether special tokes are guaranteed to be of a deterministic type,
        //  and will always be correctly labeled in 'added_tokens.json' etc.
        // The assumption is, since special tokens aren't meant to be exposed to end user, they are designed
        //  to be unmatchable by the tokenizer, therefore tokens from the vocab, which are unmatchable by the tokenizer
        //  are special tokens.
        // From testing, this appears to correlate 1:1 with special tokens.
        //

        // Counting special tokens and verifying in only one direction
        //  is sufficient to detect difference in those two sets.
        //
        uint32_t special_tokens_count_by_type = 0;
        uint32_t special_tokens_count_from_verification = 0;

        bool special_tokens_definition_mismatch = false;

        for (const auto & t : vocab.token_to_id) {
            const auto & token = t.first;
            const auto & id    = t.second;

            if( id > 32002 ) continue;

            // Count all non-normal tokens in the vocab while iterating
            if (vocab.id_to_token[id].type != LLAMA_TOKEN_TYPE_NORMAL) {
                special_tokens_count_by_type++;
            }

            // Skip single character tokens
            if (token.length() > 1) {
                bool is_tokenizable = false;

                // Split token string representation in two, in all possible ways
                //  and check if both halves can be matched to a valid token
                for (unsigned i = 1; i < token.length();) {
                    const auto left  = token.substr(0, i);
                    const auto right = token.substr(i);

                    // check if we didnt partition in the middle of a utf sequence
                    auto utf = utf8_len(left.at(left.length() - 1));

                    if (utf == 1) {
                        if (vocab.token_to_id.find(left)  != vocab.token_to_id.end() &&
                            vocab.token_to_id.find(right) != vocab.token_to_id.end() ) {
                            is_tokenizable = true;
                            break;
                        }
                        i++;
                    } else {
                        // skip over the rest of multibyte utf sequence
                        i += utf - 1;
                    }
                }

                if (!is_tokenizable) {
                    // Some tokens are multibyte, but they are utf sequences with equivalent text length of 1
                    //  it's faster to re-filter them here, since there are way less candidates now

                    // Calculate a total "utf" length of a token string representation
                    size_t utf8_str_len = 0;
                    for (unsigned i = 0; i < token.length();) {
                        utf8_str_len++;
                        i += utf8_len(token.at(i));
                    }

                    // And skip the ones which are one character
                    if (utf8_str_len > 1) {
                        // At this point what we have left are special tokens only
                        vocab.special_tokens_cache[token] = id;

                        // Count manually found special tokens
                        special_tokens_count_from_verification++;

                        // If this manually found special token is not marked as such, flag a mismatch
                        if (vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_NORMAL) {
                            special_tokens_definition_mismatch = true;
                        }
                    }
                }
            }
        }

        if (special_tokens_definition_mismatch || special_tokens_count_from_verification != special_tokens_count_by_type) {
            LLAMA_LOG_WARN("%s: mismatch in special tokens definition ( %u/%zu vs %u/%zu ).\n",
                __func__,
                special_tokens_count_from_verification, vocab.id_to_token.size(),
                special_tokens_count_by_type, vocab.id_to_token.size()
            );
        } else {
            LLAMA_LOG_INFO("%s: special tokens definition check successful ( %u/%zu ).\n",
                __func__,
                special_tokens_count_from_verification, vocab.id_to_token.size()
            );
        }
    }
}

static void llm_load_print_meta(llama_model_loader & ml, llama_model & model) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const char * rope_scaling_type = LLAMA_ROPE_SCALING_TYPES.at(hparams.rope_scaling_type_train);

    // hparams
    LLAMA_LOG_INFO("%s: format           = %s\n",     __func__, llama_file_version_name(ml.fver));
    LLAMA_LOG_INFO("%s: arch             = %s\n",     __func__, LLM_ARCH_NAMES.at(model.arch));
    LLAMA_LOG_INFO("%s: vocab type       = %s\n",     __func__, llama_model_vocab_type_name(vocab.type));
    LLAMA_LOG_INFO("%s: n_vocab          = %u\n",     __func__, hparams.n_vocab);
    LLAMA_LOG_INFO("%s: n_merges         = %u\n",     __func__, (int) vocab.bpe_ranks.size());
    LLAMA_LOG_INFO("%s: n_ctx_train      = %u\n",     __func__, hparams.n_ctx_train);
    LLAMA_LOG_INFO("%s: n_embd           = %u\n",     __func__, hparams.n_embd);
    LLAMA_LOG_INFO("%s: n_head           = %u\n",     __func__, hparams.n_head);
    LLAMA_LOG_INFO("%s: n_head_kv        = %u\n",     __func__, hparams.n_head_kv);
    LLAMA_LOG_INFO("%s: n_layer          = %u\n",     __func__, hparams.n_layer);
    LLAMA_LOG_INFO("%s: n_rot            = %u\n",     __func__, hparams.n_rot);
    LLAMA_LOG_INFO("%s: n_embd_head_k    = %u\n",     __func__, hparams.n_embd_head_k);
    LLAMA_LOG_INFO("%s: n_embd_head_v    = %u\n",     __func__, hparams.n_embd_head_v);
    LLAMA_LOG_INFO("%s: n_gqa            = %u\n",     __func__, hparams.n_gqa());
    LLAMA_LOG_INFO("%s: n_embd_k_gqa     = %u\n",     __func__, hparams.n_embd_k_gqa());
    LLAMA_LOG_INFO("%s: n_embd_v_gqa     = %u\n",     __func__, hparams.n_embd_v_gqa());
    LLAMA_LOG_INFO("%s: f_norm_eps       = %.1e\n",   __func__, hparams.f_norm_eps);
    LLAMA_LOG_INFO("%s: f_norm_rms_eps   = %.1e\n",   __func__, hparams.f_norm_rms_eps);
    LLAMA_LOG_INFO("%s: f_clamp_kqv      = %.1e\n",   __func__, hparams.f_clamp_kqv);
    LLAMA_LOG_INFO("%s: f_max_alibi_bias = %.1e\n",   __func__, hparams.f_max_alibi_bias);
    LLAMA_LOG_INFO("%s: n_ff             = %u\n",     __func__, hparams.n_ff);
    LLAMA_LOG_INFO("%s: n_expert         = %u\n",     __func__, hparams.n_expert);
    LLAMA_LOG_INFO("%s: n_expert_used    = %u\n",     __func__, hparams.n_expert_used);
    LLAMA_LOG_INFO("%s: pooling type     = %d\n",     __func__, hparams.pooling_type);
    LLAMA_LOG_INFO("%s: rope type        = %d\n",     __func__, hparams.rope_type);
    LLAMA_LOG_INFO("%s: rope scaling     = %s\n",     __func__, rope_scaling_type);
    LLAMA_LOG_INFO("%s: freq_base_train  = %.1f\n",   __func__, hparams.rope_freq_base_train);
    LLAMA_LOG_INFO("%s: freq_scale_train = %g\n",     __func__, hparams.rope_freq_scale_train);
    LLAMA_LOG_INFO("%s: n_yarn_orig_ctx  = %u\n",     __func__, hparams.n_yarn_orig_ctx);
    LLAMA_LOG_INFO("%s: rope_finetuned   = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
    LLAMA_LOG_INFO("%s: model type       = %s\n",     __func__, llama_model_type_name(model.type));
    LLAMA_LOG_INFO("%s: model ftype      = %s\n",     __func__, llama_model_ftype_name(model.ftype).c_str());
    if (ml.n_elements >= 1e12) {
        LLAMA_LOG_INFO("%s: model params     = %.2f T\n", __func__, ml.n_elements*1e-12);
    } else if (ml.n_elements >= 1e9) {
        LLAMA_LOG_INFO("%s: model params     = %.2f B\n", __func__, ml.n_elements*1e-9);
    } else if (ml.n_elements >= 1e6) {
        LLAMA_LOG_INFO("%s: model params     = %.2f M\n", __func__, ml.n_elements*1e-6);
    } else {
        LLAMA_LOG_INFO("%s: model params     = %.2f K\n", __func__, ml.n_elements*1e-3);
    }
    if (ml.n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: model size       = %.2f MiB (%.2f BPW) \n", __func__, ml.n_bytes/1024.0/1024.0,        ml.n_bytes*8.0/ml.n_elements);
    } else {
        LLAMA_LOG_INFO("%s: model size       = %.2f GiB (%.2f BPW) \n", __func__, ml.n_bytes/1024.0/1024.0/1024.0, ml.n_bytes*8.0/ml.n_elements);
    }

    // general kv
    LLAMA_LOG_INFO("%s: general.name     = %s\n",    __func__, model.name.c_str());

    // special tokens
    if (vocab.special_bos_id != -1) { LLAMA_LOG_INFO( "%s: BOS token        = %d '%s'\n", __func__, vocab.special_bos_id, vocab.id_to_token[vocab.special_bos_id].text.c_str() ); }
    if (vocab.special_eos_id != -1) { LLAMA_LOG_INFO( "%s: EOS token        = %d '%s'\n", __func__, vocab.special_eos_id, vocab.id_to_token[vocab.special_eos_id].text.c_str() ); }
    if (vocab.special_unk_id != -1) { LLAMA_LOG_INFO( "%s: UNK token        = %d '%s'\n", __func__, vocab.special_unk_id, vocab.id_to_token[vocab.special_unk_id].text.c_str() ); }
    if (vocab.special_sep_id != -1) { LLAMA_LOG_INFO( "%s: SEP token        = %d '%s'\n", __func__, vocab.special_sep_id, vocab.id_to_token[vocab.special_sep_id].text.c_str() ); }
    if (vocab.special_pad_id != -1) { LLAMA_LOG_INFO( "%s: PAD token        = %d '%s'\n", __func__, vocab.special_pad_id, vocab.id_to_token[vocab.special_pad_id].text.c_str() ); }
    if (vocab.linefeed_id    != -1) { LLAMA_LOG_INFO( "%s: LF token         = %d '%s'\n", __func__, vocab.linefeed_id,    vocab.id_to_token[vocab.linefeed_id].text.c_str() );    }
}

// Returns false if cancelled by progress_callback
static bool llm_load_tensors(
        llama_model_loader & ml,
        llama_model & model,
        int n_gpu_layers,
        enum llama_split_mode split_mode,
        int main_gpu,
        const float * tensor_split,
        bool use_mlock,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    model.t_start_us = ggml_time_us();

    auto & hparams = model.hparams;

    model.split_mode   = split_mode;
    model.main_gpu     = main_gpu;
    model.n_gpu_layers = n_gpu_layers;

    const int64_t n_layer     = hparams.n_layer;
    const int64_t i_gpu_start = std::max((int64_t) hparams.n_layer - n_gpu_layers, (int64_t) 0);

    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    model.buft_input = llama_default_buffer_type_cpu(true);

    model.buft_layer.resize(n_layer);

    // assign cpu layers
    for (int64_t i = 0; i < i_gpu_start; ++i) {
        model.buft_layer[i] = llama_default_buffer_type_cpu(true);
    }

#ifndef GGML_USE_KOMPUTE
    if (split_mode == LLAMA_SPLIT_MODE_LAYER) {
        // calculate the split points
        int device_count = llama_get_device_count();
        bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + device_count, [](float x) { return x == 0.0f; });
        std::vector<float> splits(device_count);
        if (all_zero) {
            // default split, by free memory
            for (int i = 0; i < device_count; ++i) {
                splits[i] = llama_get_device_memory(i);
            }
        } else {
            std::copy(tensor_split, tensor_split + device_count, splits.begin());
        }

        // sum and normalize the splits to get the split points
        float split_sum = 0.0f;
        for (int i = 0; i < device_count; ++i) {
            split_sum += splits[i];
            splits[i] = split_sum;
        }
        for (int i = 0; i < device_count; ++i) {
            splits[i] /= split_sum;
        }

        // assign the repeating layers to the devices according to the splits
        int act_gpu_layers = std::min(n_gpu_layers, (int)n_layer + 1);
        for (int64_t i = i_gpu_start; i < n_layer; ++i) {
            int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + device_count, float(i - i_gpu_start)/act_gpu_layers) - splits.begin();
            model.buft_layer[i] = llama_default_buffer_type_offload(layer_gpu);
        }
        // assign the output layer
        if (n_gpu_layers > n_layer) {
            int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + device_count, float(act_gpu_layers - 1)/act_gpu_layers) - splits.begin();
            model.buft_output = llama_default_buffer_type_offload(layer_gpu);
        } else {
            model.buft_output = llama_default_buffer_type_cpu(true);
        }
    } else
#endif
    {
        ggml_backend_buffer_type_t split_buft;
        if (split_mode == LLAMA_SPLIT_MODE_ROW) {
            split_buft = llama_default_buffer_type_split(main_gpu, tensor_split);
        } else {
            // LLAMA_SPLIT_MODE_NONE or LLAMA_SPLIT_MODE_LAYER in backends where it is not supported
            split_buft = llama_default_buffer_type_offload(main_gpu);
        }
        // assign the repeating layers
        for (int64_t i = i_gpu_start; i < n_layer; ++i) {
            model.buft_layer[i] = {
                split_buft,
                llama_default_buffer_type_offload(main_gpu)
            };
        }
        // assign the output layer
        if (false && n_gpu_layers > n_layer) {
            model.buft_output = {
                split_buft,
                llama_default_buffer_type_offload(main_gpu)
            };
        } else {
            model.buft_output = llama_default_buffer_type_cpu(true);
        }
    }

    // count used buffer types
    std::map<ggml_backend_buffer_type_t, int> buft_layer_count;
    buft_layer_count[model.buft_input.buft]++;
    buft_layer_count[model.buft_input.buft_matrix]++;
    buft_layer_count[model.buft_output.buft]++;
    buft_layer_count[model.buft_output.buft_matrix]++;
    for (int64_t i = 0; i < n_layer; ++i) {
        buft_layer_count[model.buft_layer[i].buft]++;
        buft_layer_count[model.buft_layer[i].buft_matrix]++;
    }

    // create one context per buffer type
    size_t ctx_size = ggml_tensor_overhead()*(ml.n_tensors + 2); // +1 for models where tok_embd is duplicated as output
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    for (auto & it : buft_layer_count) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error(format("failed to create context"));
        }
        ctx_map[it.first] = ctx;
        model.ctxs.push_back(ctx);
    }

    LLAMA_LOG_INFO("%s: ggml x ctx size = %7.2f MiB\n", __func__, model.ctxs.size()*ctx_size/1024.0/1024.0);

    // create tensors for the weights
    {
        const int64_t n_embd       = hparams.n_embd;
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa();
        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa();
        const int64_t n_embd_gqa   = n_embd_v_gqa;
        const int64_t n_vocab      = hparams.n_vocab;
        const int64_t n_vocab_type = hparams.n_vocab_type;
        const int64_t n_ff         = hparams.n_ff;

        GGML_ASSERT(n_embd_gqa == n_embd_k_gqa);

        ggml_context * ctx_input        = ctx_map.at(model.buft_input.buft);
        ggml_context * ctx_output       = ctx_map.at(model.buft_output.buft);
        ggml_context * ctx_output_split = ctx_map.at(model.buft_output.buft_matrix);
        auto ctx_for_layer              = [&](int i) { return ctx_map.at(model.buft_layer[i].buft); };
        auto ctx_for_layer_split        = [&](int i) { return ctx_map.at(model.buft_layer[i].buft_matrix); };

        model.layers.resize(n_layer);

        const auto tn = LLM_TN(model.arch);
        switch (model.arch) {
            case LLM_ARCH_LLAMA:
            case LLM_ARCH_REFACT:
            case LLM_ARCH_MINICPM:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        const int64_t output_embdne[4] = {n_embd*n_layer,0,0,0};
                        model.output_norm = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_embd = ggml_new_tensor(ctx_output_split, model.output_norm->type, 1, output_embdne);
                        ggml_set_name( model.output_embd, "output_embd" );
                        if (model.arch != LLM_ARCH_MINICPM){
                            model.output = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                        }
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

                        // optional bias tensors
                        layer.bq = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     false);
                        layer.bk = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, false);
                        layer.bv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, false);
                        layer.bo = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     false);

                        layer.ffn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

                        layer.ffn_gate_inp = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd}, false);

                        if (layer.ffn_gate_inp == nullptr) {
                            GGML_ASSERT(hparams.n_expert      == 0);
                            GGML_ASSERT(hparams.n_expert_used == 0);

                            layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                            layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                            layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                        } else {
                            GGML_ASSERT(hparams.n_expert      > 0);
                            GGML_ASSERT(hparams.n_expert_used > 0);

                            // MoE branch
                            for (uint32_t x = 0; x < hparams.n_expert; ++x) {
                                layer.ffn_gate_exp[x] = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, x), {n_embd,   n_ff});
                                layer.ffn_down_exp[x] = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, x), {  n_ff, n_embd});
                                layer.ffn_up_exp[x]   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, x), {n_embd,   n_ff});
                            }
                        }
                    }
                } break;
            case LLM_ARCH_BAICHUAN:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
                    {
                        model.output_norm = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output      = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

                        layer.ffn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

                        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                    }
                } break;
            case LLM_ARCH_FALCON:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm   = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
                        if (gguf_find_tensor(ml.ctx_gguf, tn(LLM_TENSOR_OUTPUT, "weight").c_str()) >= 0) {
                            model.output = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,     "weight"), {n_embd, n_vocab});
                        } else {
                            model.output = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}); // needs to be on GPU
                            ml.n_created--; // artificial tensor
                            ml.size_data += ggml_nbytes(model.output);
                        }
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

                        if (gguf_find_tensor(ml.ctx_gguf, tn(LLM_TENSOR_ATTN_NORM_2, "weight", i).c_str()) >= 0) {
                            layer.attn_norm_2   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd});
                            layer.attn_norm_2_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd});
                        }

                        layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
                        layer.wo   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                    }
                } break;
            case LLM_ARCH_STARCODER:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
                    model.pos_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, hparams.n_ctx_train});

                    // output
                    {
                        model.output_norm   = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
                        model.output        = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

                        layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
                        layer.bqkv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

                        layer.wo   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
                        layer.bo   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

                        layer.ffn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

                        layer.ffn_down   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
                        layer.ffn_down_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

                        layer.ffn_up     = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i),   {n_embd, n_ff});
                        layer.ffn_up_b   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP, "bias", i),     {n_ff});
                    }
                } break;
            case LLM_ARCH_PERSIMMON:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"),  {n_embd, n_vocab});

                    {
                        model.output_norm    = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b  = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
                        model.output         = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm     = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd});
                        layer.attn_norm_b   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "bias",   i), {n_embd});

                        layer.wqkv          = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV,    "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
                        layer.bqkv          = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV,    "bias",   i), {n_embd + 2*n_embd_gqa});

                        layer.wo            = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT,    "weight", i), {n_embd, n_embd});
                        layer.bo            = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT,    "bias",   i), {n_embd});

                        layer.ffn_down      = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN,    "weight", i), {n_ff, n_embd});
                        layer.ffn_down_b    = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN,    "bias",   i), {n_embd});

                        layer.ffn_up        = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,      "weight", i), {n_embd, n_ff});
                        layer.ffn_up_b      = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,      "bias",   i), {n_ff});

                        layer.ffn_norm      = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM,    "weight", i), {n_embd});
                        layer.ffn_norm_b    = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM,    "bias",   i), {n_embd});

                        layer.attn_q_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {64});
                        layer.attn_q_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {64});

                        layer.attn_k_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {64});
                        layer.attn_k_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {64});
                    }
                } break;
            case LLM_ARCH_BERT:
            case LLM_ARCH_NOMIC_BERT:
                {
                    model.tok_embd     = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab});
                    model.type_embd    = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_vocab_type});
                    if (model.arch == LLM_ARCH_BERT) {
                        model.pos_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,    "weight"), {n_embd, hparams.n_ctx_train});
                    }

                    model.tok_norm   = ml.create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd});
                    model.tok_norm_b = ml.create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd});

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        if (model.arch == LLM_ARCH_BERT) {
                            layer.wq   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
                            layer.bq   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i),   {n_embd});

                            layer.wk   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
                            layer.bk   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i),   {n_embd_gqa});

                            layer.wv   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
                            layer.bv   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i),   {n_embd_gqa});
                        } else {
                            layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
                        }

                        layer.wo              = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {n_embd, n_embd});

                        layer.attn_out_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd});
                        layer.attn_out_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i),   {n_embd});

                        layer.ffn_up          = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd, n_ff});
                        layer.ffn_down        = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {n_ff, n_embd});

                        if (model.arch == LLM_ARCH_BERT) {
                            layer.bo         = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});
                            layer.ffn_up_b   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});

                            layer.ffn_down_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});
                        } else {
                            layer.ffn_gate   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                        }

                        layer.layer_out_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd});
                        layer.layer_out_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_LAYER_OUT_NORM, "bias", i),   {n_embd});
                    }
                } break;
            case LLM_ARCH_BLOOM:
                {
                    model.tok_embd   = ml.create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab});
                    model.tok_norm   = ml.create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd});
                    model.tok_norm_b = ml.create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd});

                    // output
                    {
                        model.output_norm   = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
                        model.output        = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

                        layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
                        layer.bqkv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

                        layer.wo   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
                        layer.bo   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

                        layer.ffn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

                        layer.ffn_down   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
                        layer.ffn_down_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

                        layer.ffn_up     = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
                        layer.ffn_up_b   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
                    }
                } break;
            case LLM_ARCH_MPT:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm   = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, false);

                        // same as tok_embd, duplicated to allow offloading
                        model.output        = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab});
                        ml.n_created--; // artificial tensor
                        ml.size_data += ggml_nbytes(model.output);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, false);

                        layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
                        layer.bqkv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, false);

                        layer.wo   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
                        layer.bo   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, false);

                        layer.ffn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, false);

                        layer.ffn_down   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
                        layer.ffn_down_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, false);

                        layer.ffn_up     = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                        layer.ffn_up_b   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, false);

                        // AWQ ScaleActivation layer
                        layer.ffn_act = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_ACT, "scales", i), {n_ff}, false);
                    }
                } break;
            case LLM_ARCH_STABLELM:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm_b = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
                        model.output_norm   = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output        = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm =   ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i), {n_embd});

                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

                        // optional bias tensors, present in Stable LM 2 1.6B
                        layer.bq = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     false);
                        layer.bk = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, false);
                        layer.bv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, false);

                        layer.ffn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

                        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                    }
                } break;
            case LLM_ARCH_QWEN:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output      = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

                        layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd*3});
                        layer.bqkv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd*3});
                        layer.wo   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

                        layer.ffn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

                        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff/2});
                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff/2, n_embd});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff/2});
                    }
                } break;
            case LLM_ARCH_QWEN2:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output      = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

                        // optional bias tensors
                        layer.bq = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd});
                        layer.bk = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa});
                        layer.bv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa});

                        layer.ffn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

                        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                    }
                } break;
            case LLM_ARCH_PHI2:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm   = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
                        model.output        = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                        model.output_b      = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

                        layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, false);
                        layer.bqkv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, false);

                        if (layer.wqkv == nullptr) {
                            layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd});
                            layer.bq = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q, "bias", i),   {n_embd});

                            layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa});
                            layer.bk = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K, "bias", i),   {n_embd_gqa});

                            layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa});
                            layer.bv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V, "bias", i),   {n_embd_gqa});
                        }

                        layer.wo   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
                        layer.bo   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

                        layer.ffn_down   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
                        layer.ffn_down_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

                        layer.ffn_up     = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
                        layer.ffn_up_b   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
                    }
                } break;
            case LLM_ARCH_PLAMO:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output      = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

                        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                    }
                } break;
            case LLM_ARCH_GPT2:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
                    model.pos_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"),   {n_embd, hparams.n_ctx_train});

                    // output
                    {
                        model.output_norm   = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
                        model.output        = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd});

                        layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
                        layer.bqkv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

                        layer.wo   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
                        layer.bo   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

                        layer.ffn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

                        layer.ffn_down   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
                        layer.ffn_down_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

                        layer.ffn_up     = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
                        layer.ffn_up_b   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
                    }
                } break;
            case LLM_ARCH_CODESHELL:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm   = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
                        model.output        = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

                        layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
                        layer.bqkv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

                        layer.wo   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
                        layer.bo   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

                        layer.ffn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

                        layer.ffn_down   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
                        layer.ffn_down_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

                        layer.ffn_up     = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i),   {n_embd, n_ff});
                        layer.ffn_up_b   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP, "bias", i),     {n_ff});
                    }
                } break;
            case LLM_ARCH_ORION:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
                    {
                        model.output_norm   = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
                        model.output        = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }
                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

                        layer.ffn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

                        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                    }
                } break;
            case LLM_ARCH_INTERNLM2:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output      = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        // layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});

                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
                        layer.ffn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                    }
                } break;
            case LLM_ARCH_GEMMA:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    model.output_norm = ml.create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                    model.output      = ml.create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}); // same as tok_embd, duplicated to allow offloading
                    ml.n_created--; // artificial tensor
                    ml.size_data += ggml_nbytes(model.output);

                    const int64_t n_ff          = hparams.n_ff;
                    const int64_t n_embd_head_k = hparams.n_embd_head_k;
                    const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa();
                    const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa();

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * hparams.n_head});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * hparams.n_head, n_embd});

                        layer.ffn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                    }
                } break;
            case LLM_ARCH_STARCODER2:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                    // output
                    {
                        model.output_norm   = ml.create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b = ml.create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});

                        model.output = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, false);
                        // if output is NULL, init from the input tok embed
                        if (model.output == NULL) {
                            model.output = ml.create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
                            ml.n_created--; // artificial tensor
                            ml.size_data += ggml_nbytes(model.output);
                        }

                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

                        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
                        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
                        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

                        // optional bias tensors
                        layer.bq = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd});
                        layer.bk = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa});
                        layer.bv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa});
                        layer.bo = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd});

                        layer.ffn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

                        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
                        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});

                        // optional bias tensors
                        layer.ffn_down_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd});
                        layer.ffn_up_b   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP ,  "bias", i), {  n_ff});
                    }
                } break;
            default:
                throw std::runtime_error("unknown architecture");
        }
    }

    ml.done_getting_tensors();

    ml.init_mapping(true, use_mlock ? &model.mlock_mmap : nullptr);

    LLAMA_LOG_INFO("%s: Building init output matrix ne [ %s ] nb [ %s ]\n", __func__,
                           llama_format_tensor_shape(model.output).c_str(),
                           llama_format_tensor_blocks(model.output).c_str());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, ggml_backend_buffer_t>> ctx_bufs;

    for (auto & it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = nullptr;

        // only the mmap region containing the tensors in the model is mapped to the backend buffer
        // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer, then we could just use metal for all layers
        // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
        if (ml.use_mmap && buft == llama_default_buffer_type_cpu(true)) {

            if( false ) {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            } else {
                size_t first, last;
                ml.get_mapping_range(&first, &last, ctx);
                buf = ggml_backend_cpu_buffer_from_ptr((char *) ml.mapping->addr + first, (last - first));
            }
        }
#ifdef GGML_USE_METAL
        else if (ml.use_mmap && buft == ggml_backend_metal_buffer_type()) {
            const size_t max_size = ggml_get_max_tensor_size(ctx);
            size_t first, last;
            ml.get_mapping_range(&first, &last, ctx);
            buf = ggml_backend_metal_buffer_from_ptr((char *) ml.mapping->addr + first, last - first, max_size);
        }
#endif
        else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf != nullptr && use_mlock && ggml_backend_buffer_is_host(buf)) {
                model.mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = model.mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
        }
        if (buf == nullptr) {
            throw std::runtime_error("failed to allocate buffer");
        }
        // indicate that this buffer contains weights
        // this is used by ggml_backend_sched to improve op scheduling -> ops that use a weight are preferably scheduled to the backend that contains the weight
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        model.bufs.push_back(buf);
        ctx_bufs.emplace_back(ctx, buf);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading non-repeating layers to GPU\n", __func__);
        }

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers       = hparams.n_layer + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements
    for (ggml_backend_buffer_t buf : model.bufs) {
        LLAMA_LOG_INFO("%s: %10s buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0);
    }

    // populate tensors_by_name
    for (ggml_context * ctx : model.ctxs) {
        for (auto * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
            model.tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // load tensor data
    for (auto & it : ctx_bufs) {
        ggml_context * ctx = it.first;
        ggml_backend_buffer_t buf = it.second;
        if (!ml.load_all_data(ctx, progress_callback, progress_callback_user_data, buf, use_mlock ? &model.mlock_mmap : NULL)) {
            return false;
        }
    }

    model.mapping = std::move(ml.mapping);
    model.t_load_us = ggml_time_us() - model.t_start_us;
    return true;
}

#ifdef GGML_USE_KOMPUTE
static const llm_arch LLM_KOMPUTE_SUPPORTED_ARCHES[] {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GPT2,
    // LLM_ARCH_MPT, -- needs GGML_OP_ALIBI
    LLM_ARCH_STARCODER,
    // LLM_ARCH_PERSIMMON, -- needs GGML_OP_CONCAT
    // LLM_ARCH_REFACT, -- needs GGML_OP_ALIBI
    LLM_ARCH_BERT,
    LLM_ARCH_NOMIC_BERT,
    // LLM_ARCH_BLOOM, -- needs GGML_OP_ALIBI
    LLM_ARCH_STABLELM,
    LLM_ARCH_QWEN,
    LLM_ARCH_QWEN2,
    LLM_ARCH_PHI2,
    // LLM_ARCH_PLAMO, -- unable to test
    LLM_ARCH_CODESHELL,
    LLM_ARCH_ORION,
    LLM_ARCH_INTERNLM2,
    LLM_ARCH_MINICPM,
    LLM_ARCH_GEMMA,
};
#endif

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static int llama_model_load(const std::string & fname, llama_model & model, llama_model_params & params) {
    try {
        llama_model_loader ml(fname, params.use_mmap, params.kv_overrides);

        model.hparams.vocab_only = params.vocab_only;

        try {
            llm_load_arch(ml, model);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            llm_load_hparams(ml, model);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            llm_load_vocab(ml, model);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        //model.hparams.n_vocab += 20000;

        llm_load_print_meta(ml, model);

        if (model.hparams.n_vocab != model.vocab.id_to_token.size()) {
            throw std::runtime_error("vocab size mismatch");
        }

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

#ifdef GGML_USE_KOMPUTE
        auto & kparch = LLM_KOMPUTE_SUPPORTED_ARCHES;
        if (params.n_gpu_layers > 0 && (
            std::find(kparch, std::end(kparch), model.arch) == std::end(kparch)
            || model.hparams.n_expert > 0
            || !(
                model.ftype == LLAMA_FTYPE_ALL_F32 ||
                model.ftype == LLAMA_FTYPE_MOSTLY_F16 ||
                model.ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ||
                model.ftype == LLAMA_FTYPE_MOSTLY_Q4_1
            )
        )) {
            // TODO(cebtenzzre): propagate this error outside of llama_load_model_from_file
            LLAMA_LOG_WARN("%s: disabling Kompute due to unsupported model arch or quantization\n", __func__);
            params.n_gpu_layers = 0;
        }
#endif
        model.impressionspath = fname.substr(0, fname.find_last_of('\\'));

        if (!llm_load_tensors(
            ml, model, params.n_gpu_layers, params.split_mode,  params.main_gpu, params.tensor_split, params.use_mlock,
            params.progress_callback, params.progress_callback_user_data
        )) {
            return -2;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

    return 0;
}

//
// llm_build
//

enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
};

static struct ggml_tensor * llm_build_inp_embd(
        struct ggml_context * ctx,
        const llama_hparams & hparams,
          const llama_batch & batch,
         struct ggml_tensor * tok_embd,
         struct ggml_tensor * inp_tokens,
         struct ggml_tensor * inp_embd,
         const llm_build_cb & cb) {
    const int64_t n_embd = hparams.n_embd;

    struct ggml_tensor * inpL;

    if (batch.token) {
        struct ggml_tensor * inp_tokens_v = ggml_view_1d(ctx, inp_tokens, batch.n_tokens, 0);
        cb(inp_tokens, "inp_tokens", -1);

        inpL = ggml_get_rows(ctx, tok_embd, inp_tokens_v);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        inpL = ggml_view_2d(ctx, inp_embd, n_embd, batch.n_tokens, inp_embd->nb[1], 0);
    }

    return inpL;
}

static void llm_build_kv_store(
        struct ggml_context * ctx,
        const llama_hparams & hparams,
       llama_kv_cache * kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * k_cur,
         struct ggml_tensor * v_cur,
                    int64_t   n_ctx,
                    int32_t   n_tokens,
                    int32_t   kv_head,
         const llm_build_cb & cb,
                    int64_t   il) {
    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa();

    // compute the transposed [n_tokens, n_embd] V matrix
    struct ggml_tensor * v_cur_t = ggml_transpose(ctx, ggml_reshape_2d(ctx, v_cur, n_embd_v_gqa, n_tokens));
    //struct ggml_tensor * v_cur_t = ggml_transpose(ctx, v_cur); // TODO: reshape above is likely not needed
    cb(v_cur_t, "v_cur_t", il);

    struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv->k_l[il], n_tokens*n_embd_k_gqa, // how much attention is on each token
            (ggml_row_size(kv->k_l[il]->type, n_embd_k_gqa))*kv_head);
    cb(k_cache_view, "k_cache_view", il);

    struct ggml_tensor * v_cache_view = ggml_view_2d(ctx, kv->v_l[il], n_tokens, n_embd_v_gqa, // how much each other token is related to it?
            (  n_ctx)*ggml_type_size(kv->v_l[il]->type),
            (kv_head)*ggml_type_size(kv->v_l[il]->type));
    cb(v_cache_view, "v_cache_view", il);

    //LLAMA_LOG_INFO("Building kv: building data\n");
    // important: storing RoPE-ed version of K in the KV cache!
    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur,   k_cache_view));
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur_t, v_cache_view));

    //LLAMA_LOG_INFO("Building kv: done\n");
}

static struct ggml_tensor * llm_build_norm(
        struct ggml_context * ctx,
         struct ggml_tensor * cur,
        const llama_hparams & hparams,
         struct ggml_tensor * mw,
         struct ggml_tensor * mb,
              llm_norm_type   type,
         const llm_build_cb & cb,
                        int   il) {
    switch (type) {
        case LLM_NORM:     cur = ggml_norm    (ctx, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS: cur = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps); break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx, cur, mb);
    }

    return cur;
}

static struct ggml_tensor * llm_build_ffn(
        struct ggml_context * ctx,
         struct ggml_tensor * cur,
         struct ggml_tensor * up,
         struct ggml_tensor * up_b,
         struct ggml_tensor * gate,
         struct ggml_tensor * gate_b,
         struct ggml_tensor * down,
         struct ggml_tensor * down_b,
         struct ggml_tensor * act_scales,
            llm_ffn_op_type   type_op,
          llm_ffn_gate_type   type_gate,
         const llm_build_cb & cb,
                        int   il) {
    struct ggml_tensor * tmp = ggml_mul_mat(ctx, up, cur);
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = ggml_mul_mat(ctx, gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = ggml_mul_mat(ctx, gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }
    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            {
                cur = ggml_silu(ctx, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            {
                cur = ggml_gelu(ctx, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            {
                cur = ggml_relu(ctx, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
    }

    if (type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    cur = ggml_mul_mat(ctx, down, cur);
    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx, cur, down_b);
    }

    return cur;
}

// if max_alibi_bias > 0 then apply ALiBi
static struct ggml_tensor * llm_build_kqv(
        struct ggml_context * ctx,
          const llama_model & model,
        const llama_hparams & hparams,
       const llama_kv_cache * kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * wo,
         struct ggml_tensor * wo_b,
         struct ggml_tensor * q_cur,
         struct ggml_tensor * kq_mask,
         struct ggml_tensor * kq_pos,
                    int64_t   n_ctx,
                    int32_t   n_tokens,
                    int32_t   n_kv,
                    float     kq_scale,
         const llm_build_cb & cb,
                    int       il) {
    const int64_t n_head        = hparams.n_head;
    const int64_t n_head_kv     = hparams.n_head_kv;
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa();
    const int64_t n_embd_head_v = hparams.n_embd_head_v;

    int32_t n_ctx_small = (int32_t)n_ctx;

    //LLAMA_LOG_INFO("%s: build, n_ctx=%ld, n_tokens=%ld, n_kv=%ld\n", __func__, n_ctx_small, n_tokens, n_kv);
    struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
    cb(q, "q", il);

    struct ggml_tensor * k =
        ggml_view_3d(ctx, kv->k_l[il],
                n_embd_head_k, // 128
                n_kv, // head
                n_head_kv, // 8
                ggml_row_size(kv->k_l[il]->type, n_embd_k_gqa), // 1024
                ggml_row_size(kv->k_l[il]->type, n_embd_head_k), // 128
                0);
    cb(k, "k", il);

    struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    cb(kq, "kq", il);

    if (model.arch == LLM_ARCH_PHI2) {
        // for this arch, we need to perform the KQ multiplication with F32 precision, otherwise we get NaNs
        // ref: https://github.com/ggerganov/llama.cpp/pull/4490#issuecomment-1859055847
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    }

    kq = ggml_soft_max_ext(ctx, kq, kq_mask, NULL/*kq_pos*/, kq_scale, hparams.f_max_alibi_bias);
    cb(kq, "kq_soft_max_ext", il);

    // split cached v into n_head heads
    struct ggml_tensor * v =
        ggml_view_3d(ctx, kv->v_l[il],
                n_kv, // head
                n_embd_head_v, // 128
                n_head_kv, // 8
                ggml_element_size(kv->v_l[il])*n_ctx,
                ggml_element_size(kv->v_l[il])*n_ctx*n_embd_head_v,
                0);
    cb(v, "v", il);

    struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
    cb(kqv, "kqv", il);

    struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    cb(kqv_merged, "kqv_merged", il);

    struct ggml_tensor * cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_k*n_head, n_tokens);
    cb(cur, "kqv_merged_cont", il);

    ggml_build_forward_expand(graph, cur);
    cur = ggml_mul_mat(ctx, wo, cur);
    if (wo_b) {
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx, cur, wo_b);
    }


    /*LLAMA_LOG_INFO("%s: Building kqv matrix ne [ %s ] nb [ %s ]\n", __func__,
                           llama_format_tensor_shape(cur).c_str(),
                           llama_format_tensor_blocks(cur).c_str());*/


    return cur;
}

static struct ggml_tensor * llm_build_kv(
        struct ggml_context * ctx,
          const llama_model & model,
        const llama_hparams & hparams,
        llama_kv_cache * kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * wo,
         struct ggml_tensor * wo_b,
         struct ggml_tensor * k_cur,
         struct ggml_tensor * v_cur,
         struct ggml_tensor * q_cur,
         struct ggml_tensor * kq_mask,
         struct ggml_tensor * kq_pos,
                    int64_t   n_ctx,
                    int32_t   n_tokens,
                    int32_t   kv_head,
                    int32_t   n_kv,
                    float     kq_scale,
         const llm_build_cb & cb,
                    int       il) {

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(graph, q_cur);
    ggml_build_forward_expand(graph, k_cur);
    ggml_build_forward_expand(graph, v_cur);

    llm_build_kv_store(ctx, hparams, kv, graph, k_cur, v_cur, n_ctx, n_tokens, kv_head, cb, il);

    struct ggml_tensor * cur;

    cur  = llm_build_kqv(ctx, model, hparams, kv, graph, wo, wo_b,
            q_cur, kq_mask, kq_pos, n_ctx, n_tokens, n_kv, kq_scale, cb, il);
    cb(cur, "kqv_out", il);

    return cur;
}


static void llama_graph_compute(
        const llama_context & lctx,
          ggml_cgraph * gf,
                  int   n_threads) {
#ifdef GGML_USE_MPI
    const int64_t n_layer = lctx.model.hparams.n_layer;
    ggml_mpi_graph_compute_pre(lctx.ctx_mpi, gf, n_layer);
#endif

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(lctx.backend_metal)) {
        ggml_backend_metal_set_n_cb(lctx.backend_metal, n_threads);
    }
#endif

    if (lctx.backend_cpu != nullptr) {
        //LLAMA_LOG_INFO("%s: using backend_cpu\n", __func__);
        ggml_backend_cpu_set_n_threads(lctx.backend_cpu, n_threads);
        ggml_backend_cpu_set_abort_callback(lctx.backend_cpu, lctx.abort_callback, lctx.abort_callback_data);
    }

    ggml_backend_sched_graph_compute(lctx.sched, gf);

    //fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(lctx.sched));

#ifdef GGML_USE_MPI
    ggml_mpi_graph_compute_post(lctx.ctx_mpi, gf, n_layer);
#endif
}

struct llm_build_context {
    const llama_model    & model;
    const llama_context  & lctx;
    const llama_hparams  & hparams;
    const llama_cparams  & cparams;
    const llama_batch    & batch;
          llama_kv_cache * kv_self;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_keys;
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int32_t n_tokens;
    const int32_t n_kv;     // size of KV cache to consider (n_kv <= n_ctx)
    const int32_t kv_head;  // index of where we store new KV data in the cache
    const int32_t n_orig_ctx;

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    const llm_build_cb & cb;

    std::vector<uint8_t> & buf_compute_meta;

    struct ggml_context * ctx0 = nullptr;

    llm_build_context(
        llama_context  & lctx,
    const llama_batch  & batch,
    const llm_build_cb & cb,
                  bool   worst_case) :
        model            (lctx.model),
        lctx             (lctx),
        hparams          (model.hparams),
        cparams          (lctx.cparams),
        batch            (batch),
        kv_self          (lctx.kv_self),
        n_embd           (hparams.n_embd),
        n_layer          (hparams.n_layer),
        n_rot            (hparams.n_rot),
        n_ctx            (cparams.n_ctx),
        n_keys           (kv_context*cparams.n_ctx),
        n_head           (hparams.n_head),
        n_head_kv        (hparams.n_head_kv),
        n_embd_head_k    (hparams.n_embd_head_k),
        n_embd_k_gqa     (hparams.n_embd_k_gqa()),
        n_embd_head_v    (hparams.n_embd_head_v),
        n_embd_v_gqa     (hparams.n_embd_v_gqa()),
        n_expert         (hparams.n_expert),
        n_expert_used    (hparams.n_expert_used),
        freq_base        (cparams.rope_freq_base),
        freq_scale       (cparams.rope_freq_scale),
        ext_factor       (cparams.yarn_ext_factor),
        attn_factor      (cparams.yarn_attn_factor),
        beta_fast        (cparams.yarn_beta_fast),
        beta_slow        (cparams.yarn_beta_slow),
        norm_eps         (hparams.f_norm_eps),
        norm_rms_eps     (hparams.f_norm_rms_eps),
        n_tokens         (batch.n_tokens),
        n_kv             (worst_case ? lctx.kv_self->size : (lctx.seq_end+n_tokens) ),
        kv_head          (worst_case ? (lctx.kv_self->size -n_tokens) : lctx.seq_end),
        n_orig_ctx       (cparams.n_yarn_orig_ctx),
        pooling_type     (cparams.pooling_type),
        rope_type        (hparams.rope_type),
        cb               (cb),
        buf_compute_meta (lctx.buf_compute_meta) {
            // all initializations should be done in init()
        }

    void init() {
        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_compute_meta.size(),
            /*.mem_buffer =*/ buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };

        ctx0 = ggml_init(params);
    }

    void free() {
        if (ctx0) {
            ggml_free(ctx0);
            ctx0 = nullptr;
        }
    }

    struct ggml_cgraph * build_k_shift() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * tmp =
                // we rotate only the first n_rot dimensions
                ggml_rope_custom_inplace(ctx0,
                        ggml_view_3d(ctx0, kv_self->k_l[il],
                            n_embd_head_k, n_head_kv, n_keys,
                            ggml_row_size(kv_self->k_l[il]->type, n_embd_head_k),
                            ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),
                            0),
                        lctx.inp_K_shift, n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            cb(tmp, "K_shifted", il);
            ggml_build_forward_expand(gf, tmp);
        }

        return gf;
    }

    struct ggml_cgraph * build_defrag(const std::vector<uint32_t> & ids) {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        for (uint32_t i = 0; i < ids.size(); ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == ids.size()) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < ids.size() && ids[i + nm] == id + nm) {
                nm++;
            }

            for (int il = 0; il < n_layer; ++il) {
                ggml_tensor * view_k_src = ggml_view_2d(ctx0, kv_self->k_l[il],
                        n_embd_k_gqa, nm,
                        ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),
                        ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*i));

                ggml_tensor * view_k_dst = ggml_view_2d(ctx0, kv_self->k_l[il],
                        n_embd_k_gqa, nm,
                        ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),
                        ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*id));

                ggml_tensor * view_v_src = ggml_view_2d(ctx0, kv_self->v_l[il],
                        nm, n_embd_v_gqa,
                        ggml_row_size(kv_self->v_l[il]->type, kv_self->size),
                        ggml_row_size(kv_self->v_l[il]->type, i));

                ggml_tensor * view_v_dst = ggml_view_2d(ctx0, kv_self->v_l[il],
                        nm, n_embd_v_gqa,
                        ggml_row_size(kv_self->v_l[il]->type, kv_self->size),
                        ggml_row_size(kv_self->v_l[il]->type, id));

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_k_src, view_k_dst));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_v_src, view_v_dst));
            }

            i += nm - 1;
        }

        //LLAMA_LOG_INFO("gf->n_nodes = %d\n", gf->n_nodes);

        return gf;
    }

    struct ggml_cgraph * build_llama() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        const int64_t n_vocab = hparams.n_vocab;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        //LLAMA_LOG_INFO("build_inp_embd\n");
        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        struct ggml_tensor *inpL_first=NULL;



        struct ggml_tensor * out_embd = NULL;
        if( lctx.record_all == 2 ) {
            out_embd = ggml_view_1d(ctx0, model.output_embd, 32*4096, 0);
            LLAMA_LOG_INFO("%s: out_embd type = %s, size = %d\n", __func__, ggml_type_name(out_embd->type), ggml_type_size(out_embd->type));
        } else {
            LLAMA_LOG_INFO("build_layers tokens:%d, head:%d, keys:%d\n", n_tokens, kv_head, n_keys);
        }

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, inp_pos, n_keys, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            //std::string obuf="";
            //for( int d = 0; d < 4; d++ ) {
            //    obuf += std::to_string(cur->ne[d]) + ",";
            //}
            //LLAMA_LOG_INFO("%s: cur dims %s\n", __func__, obuf.c_str());

            struct ggml_tensor * ffn_inp = NULL;
//            if( inpL_first == NULL && lctx->override ) {
//                ffn_inp = ggml_add(ctx0, cur, lctx->override;
//            }
            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            if( inpL_first == NULL ) inpL_first = ffn_inp;

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   NULL,
                        model.layers[il].ffn_gate, NULL,
                        model.layers[il].ffn_down, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);

                //ggml_backend_tensor_get_async(backend_res, res, logits_out.data() + il * ( n_vocab * sizeof(float) * n_tokens ), 0, n_vocab*n_tokens*sizeof(float));
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                ggml_tensor * logits = ggml_mul_mat(ctx0, model.layers[il].ffn_gate_inp, cur); // [n_tokens, num_experts]
                cb(logits, "ffn_moe_logits", il);

                ggml_tensor * probs = ggml_soft_max(ctx0, logits); // [n_tokens, num_experts]
                cb(probs, "ffn_moe_probs", il);

                // select experts
                ggml_tensor * selected_experts = ggml_top_k(ctx0, probs, n_expert_used); // [n_tokens, num_experts_per_tok]
                cb(selected_experts->src[0], "ffn_moe_argsort", il);

                ggml_tensor * weights = ggml_get_rows(ctx0,
                        ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected_experts);
                cb(weights, "ffn_moe_weights", il);

                weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens); // [n_tokens, num_experts_per_tok]

                ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
                cb(weights_sum, "ffn_moe_weights_sum", il);

                weights = ggml_div(ctx0, weights, weights_sum); // [n_tokens, num_experts_per_tok]
                cb(weights, "ffn_moe_weights_norm", il);

                // compute expert outputs
                ggml_tensor * moe_out = nullptr;

                for (int i = 0; i < n_expert_used; ++i) {
                    ggml_tensor * cur_expert;

                    ggml_tensor * cur_up = ggml_mul_mat_id(ctx0, model.layers[il].ffn_up_exp, n_expert, selected_experts, i, cur);
                    cb(cur_up, "ffn_moe_up", il);

                    ggml_tensor * cur_gate = ggml_mul_mat_id(ctx0, model.layers[il].ffn_gate_exp, n_expert, selected_experts, i, cur);
                    cb(cur_gate, "ffn_moe_gate", il);

                    cur_gate = ggml_silu(ctx0, cur_gate);
                    cb(cur_gate, "ffn_moe_silu", il);

                    cur_expert = ggml_mul(ctx0, cur_up, cur_gate); // [n_tokens, n_embd]
                    cb(cur_expert, "ffn_moe_gate_par", il);

                    cur_expert = ggml_mul_mat_id(ctx0, model.layers[il].ffn_down_exp, n_expert, selected_experts, i, cur_expert); // [n_tokens, n_embd]
                    cb(cur_expert, "ffn_moe_down", il);

                    cur_expert = ggml_mul(ctx0, cur_expert,
                            ggml_view_2d(ctx0, weights, 1, n_tokens, weights->nb[1], i*weights->nb[0]));
                    cb(cur_expert, "ffn_moe_weighted", il);

                    if (i == 0) {
                        moe_out = cur_expert;
                    } else {
                        moe_out = ggml_add(ctx0, moe_out, cur_expert);
                        cb(moe_out, "ffn_moe_out", il);
                    }
                }

                cur = moe_out;
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;

            if( lctx.record_all == 2 ) {
                cur = llm_build_norm(ctx0, cur, hparams,
                        model.output_norm, NULL,
                        LLM_NORM_RMS, cb, -1);

                LLAMA_LOG_INFO("%s: Building output norm to ne [ %s ] nb [ %s ]\n", __func__,
                                       llama_format_tensor_shape(cur).c_str(),
                                       llama_format_tensor_blocks(cur).c_str());

                cur = ggml_set_1d_inplace(ctx0, out_embd, cur, il*512*4096*ggml_type_size(out_embd->type) );
            }
        }

        cur = inpL;
        if( lctx.record_all == 2 ) {
            cb(out_embd, "result_embds", -1);
        }
        if( inpL_first ) {
            cb(inpL_first, "input_coding", -1);
            ggml_build_forward_expand(gf, inpL_first);
        }

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        /*LLAMA_LOG_INFO("%s: Building output matrix ne [ %s ] nb [ %s ]\n", __func__,
                               llama_format_tensor_shape(cur).c_str(),
                               llama_format_tensor_blocks(cur).c_str());*/

        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_baichuan() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        // positions of the tokens in the KV cache
        struct ggml_tensor * KQ_pos = ggml_view_1d(ctx0, lctx.inp_KQ_pos, n_kv, 0);
        cb(KQ_pos, "KQ_pos", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                switch (model.type) {
                    case MODEL_7B:
                        Qcur = ggml_rope_custom(
                            ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos,
                            n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                        );
                        Kcur = ggml_rope_custom(
                            ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                            n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                        );
                        break;
                    case MODEL_13B:
                        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd/n_head, n_head, n_tokens);
                        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd/n_head, n_head, n_tokens);
                        break;
                    default:
                        GGML_ASSERT(false);
                }
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);


                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask, KQ_pos, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   NULL,
                        model.layers[il].ffn_gate, NULL,
                        model.layers[il].ffn_down, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_falcon() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * attn_norm;

            attn_norm = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(attn_norm, "attn_norm", il);

            // self-attention
            {
                if (model.layers[il].attn_norm_2) {
                    // Falcon-40B
                    cur = llm_build_norm(ctx0, inpL, hparams,
                            model.layers[il].attn_norm_2,
                            model.layers[il].attn_norm_2_b,
                            LLM_NORM, cb, il);
                    cb(cur, "attn_norm_2", il);
                } else {
                    cur = attn_norm;
                }

                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_custom(
                    ctx0, Qcur, inp_pos, n_rot, rope_type, 0, n_orig_ctx,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, Kcur, inp_pos, n_rot, rope_type, 0, n_orig_ctx,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * ffn_inp = cur;

            // feed forward
            {
                cur = llm_build_ffn(ctx0, attn_norm, // !! use the attn norm, not the result
                        model.layers[il].ffn_up,   NULL,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            cur = ggml_add(ctx0, cur, inpL);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        // norm
        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_starcoder() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * pos;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_persimmon() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head   == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head/2 == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * residual = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self attention
            {
                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                // split qkv
                GGML_ASSERT(n_head_kv == n_head);

                struct ggml_tensor * tmpqkv = ggml_reshape_4d(ctx0, cur, n_embd_head, 3, n_head, n_tokens);
                cb(tmpqkv, "tmpqkv", il);

                struct ggml_tensor * tmpqkv_perm = ggml_cont(ctx0, ggml_permute(ctx0, tmpqkv, 0, 3, 1, 2));
                cb(tmpqkv_perm, "tmpqkv", il);

                struct ggml_tensor * tmpq = ggml_view_3d(
                        ctx0, tmpqkv_perm, n_embd_head, n_head, n_tokens,
                        ggml_element_size(tmpqkv_perm) * n_embd_head,
                        ggml_element_size(tmpqkv_perm) * n_embd_head * n_head,
                        0
                        );
                cb(tmpq, "tmpq", il);

                struct ggml_tensor * tmpk = ggml_view_3d(
                        ctx0, tmpqkv_perm, n_embd_head, n_head, n_tokens,
                        ggml_element_size(tmpqkv_perm) * n_embd_head,
                        ggml_element_size(tmpqkv_perm) * n_embd_head * n_head,
                        ggml_element_size(tmpqkv_perm) * n_embd_head * n_head * n_tokens
                        );
                cb(tmpk, "tmpk", il);

                // Q/K Layernorm
                tmpq = llm_build_norm(ctx0, tmpq, hparams,
                        model.layers[il].attn_q_norm,
                        model.layers[il].attn_q_norm_b,
                        LLM_NORM, cb, il);
                cb(tmpq, "tmpq", il);

                tmpk = llm_build_norm(ctx0, tmpk, hparams,
                        model.layers[il].attn_k_norm,
                        model.layers[il].attn_k_norm_b,
                        LLM_NORM, cb, il);
                cb(tmpk, "tmpk", il);

                // RoPE the first n_rot of q/k, pass the other half, and concat.
                struct ggml_tensor * qrot = ggml_view_3d(
                        ctx0, tmpq, n_rot, n_head, n_tokens,
                        ggml_element_size(tmpq) * n_embd_head,
                        ggml_element_size(tmpq) * n_embd_head * n_head,
                        0
                        );
                cb(qrot, "qrot", il);

                struct ggml_tensor * krot = ggml_view_3d(
                        ctx0, tmpk, n_rot, n_head, n_tokens,
                        ggml_element_size(tmpk) * n_embd_head,
                        ggml_element_size(tmpk) * n_embd_head * n_head,
                        0
                        );
                cb(krot, "krot", il);

                // get the second half of tmpq, e.g tmpq[n_rot:, :, :]
                struct ggml_tensor * qpass = ggml_view_3d(
                        ctx0, tmpq, n_rot, n_head, n_tokens,
                        ggml_element_size(tmpq) * n_embd_head,
                        ggml_element_size(tmpq) * n_embd_head * n_head,
                        ggml_element_size(tmpq) * n_rot
                        );
                cb(qpass, "qpass", il);

                struct ggml_tensor * kpass = ggml_view_3d(
                        ctx0, tmpk, n_rot, n_head, n_tokens,
                        ggml_element_size(tmpk) * n_embd_head,
                        ggml_element_size(tmpk) * n_embd_head * n_head,
                        ggml_element_size(tmpk) * n_rot
                        );
                cb(kpass, "kpass", il);

                struct ggml_tensor * qrotated = ggml_rope_custom(
                    ctx0, qrot, inp_pos, n_rot, rope_type, 0, n_orig_ctx,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(qrotated, "qrotated", il);

                struct ggml_tensor * krotated = ggml_rope_custom(
                    ctx0, krot, inp_pos, n_rot, rope_type, 0, n_orig_ctx,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(krotated, "krotated", il);

                // ggml currently only supports concatenation on dim=2
                // so we need to permute qrot, qpass, concat, then permute back.
                qrotated = ggml_cont(ctx0, ggml_permute(ctx0, qrotated, 2, 1, 0, 3));
                cb(qrotated, "qrotated", il);

                krotated = ggml_cont(ctx0, ggml_permute(ctx0, krotated, 2, 1, 0, 3));
                cb(krotated, "krotated", il);

                qpass = ggml_cont(ctx0, ggml_permute(ctx0, qpass, 2, 1, 0, 3));
                cb(qpass, "qpass", il);

                kpass = ggml_cont(ctx0, ggml_permute(ctx0, kpass, 2, 1, 0, 3));
                cb(kpass, "kpass", il);

                struct ggml_tensor * Qcur = ggml_concat(ctx0, qrotated, qpass);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_concat(ctx0, krotated, kpass);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Q = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 2, 1, 0, 3));
                cb(Q, "Q", il);

                Kcur = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 2, 1, 0, 3));
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = ggml_view_3d(
                        ctx0, tmpqkv_perm, n_embd_head, n_head, n_tokens,
                        ggml_element_size(tmpqkv_perm) * n_embd_head,
                        ggml_element_size(tmpqkv_perm) * n_embd_head * n_head,
                        ggml_element_size(tmpqkv_perm) * n_embd_head * n_head * n_tokens * 2
                        );
                cb(Vcur, "Vcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Q, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, residual, cur);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        NULL,
                        LLM_FFN_RELU_SQR, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_refact() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        // positions of the tokens in the KV cache
        struct ggml_tensor * KQ_pos = ggml_view_1d(ctx0, lctx.inp_KQ_pos, n_kv, 0);
        cb(KQ_pos, "KQ_pos", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                cb(Kcur, "Kcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                cb(Qcur, "Qcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask, KQ_pos, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   NULL,
                        model.layers[il].ffn_gate, NULL,
                        model.layers[il].ffn_down, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_bert() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        // get input vectors with right size
        const size_t stride1 = n_tokens * ggml_type_size(lctx.inp_tokens->type);

        struct ggml_tensor * inp_pos  = ggml_view_1d(ctx0, lctx.inp_pos,  n_tokens, 0);
        struct ggml_tensor * inp_mean = ggml_view_2d(ctx0, lctx.inp_mean, n_tokens, n_tokens, stride1, 0);
        struct ggml_tensor * inp_cls  = ggml_view_1d(ctx0, lctx.inp_cls,  n_tokens, 0);

        // construct input embeddings (token, type, position)
        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);

        // token types are hardcoded to zero ("Sentence A")
        struct ggml_tensor * type_row0 = ggml_view_1d(ctx0, model.type_embd, n_embd, 0);
        inpL = ggml_add(ctx0, inpL, type_row0);
        if (model.arch == LLM_ARCH_BERT) {
            inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
        }
        cb(inpL, "inp_embd", -1);

        // embed layer norm
        inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, model.tok_norm_b, LLM_NORM, cb, -1);
        cb(inpL, "inp_norm", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_cont(ctx0, ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_tokens, n_tokens, n_tokens*ggml_type_size(lctx.inp_KQ_mask->type), 0));
        cb(KQ_mask, "KQ_mask", -1); // [n_tokens, n_tokens]

        // iterate layers
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * cur = inpL;

            struct ggml_tensor * Qcur;
            struct ggml_tensor * Kcur;
            struct ggml_tensor * Vcur;

            // self-attention
            if (model.arch == LLM_ARCH_BERT) {
                Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wq, cur), model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wk, cur), model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wv, cur), model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            } else {
                // compute Q and K and RoPE them
                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);
            }

            struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            cb(kq, "kq", il);

            kq = ggml_soft_max_ext(ctx0, kq, KQ_mask, nullptr, 1.0f/sqrtf(float(n_embd_head)), hparams.f_max_alibi_bias);
            cb(kq, "kq_soft_max_ext", il);

            struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_tokens)));
            cb(v, "v", il);

            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_tokens, n_embd_head, n_head_kv), kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
            cb(cur, "kqv_merged_cont", il);

            ggml_build_forward_expand(gf, cur);

            cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);
            if (model.layers[il].bo) {
                cb(cur, "kqv_wo", il);
            }

            if (model.layers[il].bo) {
                cur = ggml_add(ctx0, cur, model.layers[il].bo);
            }
            cb(cur, "kqv_out", il);

            // re-add the layer input
            cur = ggml_add(ctx0, cur, inpL);

            // attention layer norm
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, cb, il);

            struct ggml_tensor * ffn_inp = cur;
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (model.arch == LLM_ARCH_BERT) {
                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            } else {
                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   NULL,
                        model.layers[il].ffn_gate, NULL,
                        model.layers[il].ffn_down, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            }
            cb(cur, "ffn_out", il);

            // attentions bypass the intermediate layer
            cur = ggml_add(ctx0, cur, ffn_inp);

            // output layer norm
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, cb, il);

            // input for next layer
            inpL = cur;
        }

        // final output
        cur = inpL;
        cb(cur, "result_embd", -1);

        // pooling layer
        switch (pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // nop
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
                {
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)), inp_mean);
                    cb(cur, "result_embd_pooled", -1);
                } break;
            case LLAMA_POOLING_TYPE_CLS:
                {
                    cur = ggml_get_rows(ctx0, cur, inp_cls);
                    cb(cur, "result_embd_pooled", -1);
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ASSERT(false && "Invalid pooling type");
                } break;
        }

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_bloom() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        // positions of the tokens in the KV cache
        struct ggml_tensor * KQ_pos = ggml_view_1d(ctx0, lctx.inp_KQ_pos, n_kv, 0);
        cb(KQ_pos, "KQ_pos", -1);

        inpL = llm_build_norm(ctx0, inpL, hparams,
                model.tok_norm,
                model.tok_norm_b,
                LLM_NORM, cb, -1);
        cb(inpL, "inp_norm", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, KQ_pos, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            // Add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_mpt() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        // positions of the tokens in the KV cache
        struct ggml_tensor * KQ_pos = ggml_view_1d(ctx0, lctx.inp_KQ_pos, n_kv, 0);
        cb(KQ_pos, "KQ_pos", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * attn_norm;

            attn_norm = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(attn_norm, "attn_norm", il);

            // self-attention
            {
                cur = attn_norm;

                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                if (model.layers[il].bqkv){
                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);
                }

                if (hparams.f_clamp_kqv > 0.0f) {
                    cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(cur, "wqkv_clamped", il);
                }

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, KQ_pos, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            // Add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // feed forward
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);
                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        model.layers[il].ffn_act,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_stablelm() {
        struct ggml_cgraph * gf = ggml_new_graph(ctx0);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   NULL,
                        model.layers[il].ffn_gate, NULL,
                        model.layers[il].ffn_down, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_qwen() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 2*sizeof(float)*(n_embd)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_custom(
                    ctx0, Qcur, inp_pos, n_rot, rope_type, 0, n_orig_ctx,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, Kcur, inp_pos, n_rot, rope_type, 0, n_orig_ctx,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward forward
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   NULL,
                        model.layers[il].ffn_gate, NULL,
                        model.layers[il].ffn_down, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_qwen2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                // these nodes are added to the graph together so that they are not reordered
                // by doing so, the number of splits in the graph is reduced
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);

                Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, cur,
                    model.layers[il].ffn_up,   NULL,
                    model.layers[il].ffn_gate, NULL,
                    model.layers[il].ffn_down, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_phi2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * attn_norm_output;
        struct ggml_tensor * ffn_output;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            attn_norm_output = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(attn_norm_output, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = nullptr;
                struct ggml_tensor * Kcur = nullptr;
                struct ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);

                    Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                    Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
                } else {
                    Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_custom(
                    ctx0, Qcur, inp_pos, n_rot, rope_type, 0, n_orig_ctx,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                // with phi2, we scale the Q to avoid precision issues
                // ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
                Qcur = ggml_scale(ctx0, Qcur, 1.0f/sqrtf(float(n_embd_head)));
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, Kcur, inp_pos, n_rot, rope_type, 0, n_orig_ctx,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f, cb, il);
                cb(cur, "kqv_out", il);
            }

            // FF
            {
                ffn_output = llm_build_ffn(ctx0, attn_norm_output,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(ffn_output, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_output);
            cb(cur, "l_out", il);

            cur = ggml_add(ctx0, cur, inpL);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output_no_bias", -1);

        cur = ggml_add(ctx0, cur, model.output_b);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_plamo() {
        struct ggml_cgraph * gf = ggml_new_graph(ctx0);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            struct ggml_tensor * attention_norm = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_custom(
                        ctx0, ggml_reshape_3d(ctx0, Qcur, n_rot, n_head,    n_tokens), inp_pos,
                        n_embd_head, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                        ctx0, ggml_reshape_3d(ctx0, Kcur, n_rot, n_head_kv, n_tokens), inp_pos,
                        n_embd_head, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }
            struct ggml_tensor * sa_out = cur;

            cur = attention_norm;

            // feed-forward network
            {
                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up, NULL,
                        model.layers[il].ffn_gate, NULL,
                        model.layers[il].ffn_down, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, sa_out);
            cb(cur, "l_out", il);

            cur = ggml_add(ctx0, cur, inpL);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_gpt2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * pos;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_codeshell() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * tmpq = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * tmpk = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(tmpq, "tmpq", il);
                cb(tmpk, "tmpk", il);
                cb(Vcur, "Vcur", il);

                struct ggml_tensor * Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head,    n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_orion() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                // if (model.layers[il].bq) {
                //     Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                //     cb(Qcur, "Qcur", il);
                // }

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                // if (model.layers[il].bk) {
                //     Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                //     cb(Kcur, "Kcur", il);
                // }

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                // if (model.layers[il].bv) {
                //     Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                //     cb(Vcur, "Vcur", il);
                // }

                Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, cur,
                    model.layers[il].ffn_up,   NULL,
                    model.layers[il].ffn_gate, NULL,
                    model.layers[il].ffn_down, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_internlm2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, cur,
                    model.layers[il].ffn_up,   NULL,
                    model.layers[il].ffn_gate, NULL,
                    model.layers[il].ffn_down, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    // ref: https://arxiv.org/abs/2203.03466
    //      https://github.com/ggerganov/llama.cpp/issues/5276#issuecomment-1925774738
    // based on the original build_llama() function
    struct ggml_cgraph * build_minicpm() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        const int64_t n_embd = hparams.n_embd;
        //TODO: if the model varies, these parameters need to be read from the model
        const int64_t n_embd_base = 256;
        const float scale_embd  = 12.0f;
        const float scale_depth = 1.4f;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // scale the input embeddings
        inpL = ggml_scale(ctx0, inpL, scale_embd);
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            // scale_res - scale the hidden states for residual connection
            const float scale_res = scale_depth/sqrtf(float(n_layer));
            cur = ggml_scale(ctx0, cur, scale_res);
            cb(cur, "hidden_scaled", -1);

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   NULL,
                        model.layers[il].ffn_gate, NULL,
                        model.layers[il].ffn_down, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            // scale the hidden states for residual connection
            cur = ggml_scale(ctx0, cur, scale_res);
            cb(cur, "hidden_scaled_ffn", -1);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head scaling
        const float scale_lmhead = float(n_embd_base)/float(n_embd);
        cur = ggml_scale(ctx0, cur, scale_lmhead);
        cb(cur, "lmhead_scaling", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.tok_embd, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_gemma() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head_k = hparams.n_embd_head_k;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_custom(
                        ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens), inp_pos,
                        n_embd_head_k, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head_k)));
                cb(Qcur, "Qcur_scaled", il);

                Kcur = ggml_rope_custom(
                        ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens), inp_pos,
                        n_embd_head_k, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f, cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
            cb(sa_out, "sa_out", il);

            cur = llm_build_norm(ctx0, sa_out, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up, NULL,
                        model.layers[il].ffn_gate, NULL,
                        model.layers[il].ffn_down, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, sa_out);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_starcoder2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
        cb(inpL, "inp_embd", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_view_1d(ctx0, lctx.inp_pos, n_tokens, 0);
        cb(inp_pos, "inp_pos", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_view_2d(ctx0, lctx.inp_KQ_mask, n_kv, n_tokens, n_kv*ggml_type_size(lctx.inp_KQ_mask->type), 0);
        cb(KQ_mask, "KQ_mask", -1);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                    LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_custom(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                    n_rot, rope_type, 0, n_orig_ctx, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, model, hparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, nullptr, n_ctx, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                cb(cur, "kqv_out", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network

            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,
                        NULL,                      NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);
            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, model.output_norm_b,
                LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }
};

static struct ggml_cgraph * llama_build_graph_k_shift(llama_context & lctx) {
    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * , const char * , int ) {



        //Scott.
    };

    struct llm_build_context llm(lctx, dummy, cb, false);

    llm.init();

    struct ggml_cgraph * result = llm.build_k_shift();

    llm.free();

    return result;
}

static struct ggml_cgraph * llama_build_graph(
         llama_context & lctx,
     const llama_batch & batch,
                  bool   worst_case) {
    const auto & model = lctx.model;

    // this callback allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
    llm_build_cb cb = [&](struct ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!lctx.cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_node_backend(lctx.sched, cur, lctx.backend_cpu);
            }
        }
    };

    //LLAMA_LOG_INFO("build_graph\n");

    struct ggml_cgraph * result = NULL;

    struct llm_build_context llm(lctx, batch, cb, worst_case);

    llm.init();

    switch (model.arch) {
        case LLM_ARCH_LLAMA:
            {
                result = llm.build_llama();
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                result = llm.build_baichuan();
            } break;
        case LLM_ARCH_FALCON:
            {
                result = llm.build_falcon();
            } break;
        case LLM_ARCH_STARCODER:
            {
                result = llm.build_starcoder();
            } break;
        case LLM_ARCH_PERSIMMON:
            {
                result = llm.build_persimmon();
            } break;
        case LLM_ARCH_REFACT:
            {
                result = llm.build_refact();
            } break;
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
            {
                result = llm.build_bert();
            } break;
        case LLM_ARCH_BLOOM:
            {
                result = llm.build_bloom();
            } break;
        case LLM_ARCH_MPT:
            {
                result = llm.build_mpt();
            } break;
         case LLM_ARCH_STABLELM:
            {
                result = llm.build_stablelm();
            } break;
        case LLM_ARCH_QWEN:
            {
                result = llm.build_qwen();
            } break;
        case LLM_ARCH_QWEN2:
            {
                result = llm.build_qwen2();
            } break;
        case LLM_ARCH_PHI2:
            {
                result = llm.build_phi2();
            } break;
        case LLM_ARCH_PLAMO:
            {
                result = llm.build_plamo();
            } break;
        case LLM_ARCH_GPT2:
            {
                result = llm.build_gpt2();
            } break;
        case LLM_ARCH_CODESHELL:
            {
                result = llm.build_codeshell();
            } break;
        case LLM_ARCH_ORION:
            {
                result = llm.build_orion();
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                result = llm.build_internlm2();
            } break;
        case LLM_ARCH_MINICPM:
            {
                result = llm.build_minicpm();
            } break;
        case LLM_ARCH_GEMMA:
            {
                result = llm.build_gemma();
            } break;
        case LLM_ARCH_STARCODER2:
            {
                result = llm.build_starcoder2();
            } break;
        default:
            GGML_ASSERT(false);
    }

    llm.free();

    return result;
}

static void llama_set_k_shift(llama_context & lctx) {
}

static void llama_set_inputs(llama_context & lctx, const llama_batch & batch) {
    //
    // set input data
    //

    const auto & hparams = lctx.model.hparams;
    const auto & cparams = lctx.cparams;
    const auto & kv_self = lctx.kv_self;

    llama_pos posn[128];

    if (batch.token) {
        const int64_t n_tokens = batch.n_tokens;

        //LLAMA_LOG_INFO("%s: prepare %d tokens\n", __func__, n_tokens);
        ggml_backend_tensor_set(lctx.inp_tokens, batch.token, 0, n_tokens*ggml_element_size(lctx.inp_tokens));
    }

    if (batch.embd) {
        const int64_t n_embd   = hparams.n_embd;
        const int64_t n_tokens = batch.n_tokens;

        LLAMA_LOG_INFO("%s: use embd %d\n", __func__, n_embd);
        ggml_backend_tensor_set(lctx.inp_embd, batch.embd, 0, n_tokens*n_embd*ggml_element_size(lctx.inp_embd));
    }

    for( int i=0; i<batch.n_tokens; i++ ) {
        posn[i] = lctx.seq_end + i;
    }
    ggml_backend_tensor_set(lctx.inp_pos, posn, 0, batch.n_tokens*sizeof(llama_pos) );

    const int64_t n_tokens = batch.n_tokens;
    const int64_t n_kv     = lctx.seq_end + n_tokens;

    assert(ggml_backend_buffer_is_host(lctx.inp_KQ_mask->buffer));

    float * data = (float *) lctx.inp_KQ_mask->data;
    float *pattern = (float*)pool_alloc( n_kv * sizeof(float) );

    int i;
    for( i=0; i<lctx.seq_end; i++ ) {
        pattern[i] = 0.0;
    }
    for( ; i<n_kv; i++ ) {
        pattern[i] = -INFINITY;
    }

    for (int j = 0; j < n_tokens; ++j) {
        memcpy( data + (j*n_kv), pattern, n_kv*sizeof(float) );
        memset( data + (j*n_kv)+lctx.seq_end, 0, (j+1)*sizeof(float) );
        /*
        for( int jv = 0; jv <= j; jv++ ) {
            data[j*n_kv+lctx.seq_end+jv] = 0.0;
        }*/
    }
    pool_free(pattern);
}


const char *llama_idle( llama_context *ctx, const char **keyptr, const char **fmtptr, int *max_gen )
{
    /*
    float max_weight=0;
    int i;

    for( i=0; strncmp(control_labels[i].query, "><", 2) != 0; i++ ) {
        max_weight += control_labels[i].weight;
        //LLAMA_LOG_INFO("max_weight: %f\n", max_weight);
    }
    srand(clock());
    float x = ((float)rand()/(float)RAND_MAX) * max_weight;
    LLAMA_LOG_INFO("w: %f, x: %f\n", max_weight, x);

    for( i=0; strncmp(control_labels[i].query, "><", 2) != 0; i++ ) {
        x -= control_labels[i].weight;
        if( x <= 0.0 ) {
            *keyptr = control_labels[i].key;
            *fmtptr = control_labels[i].fmt;
            *max_gen = control_labels[i].max_tokens;

            //LLAMA_LOG_INFO("query: %s\n", control_labels[i].query);
            return control_labels[i].query;
        }
    }
    LLAMA_LOG_INFO("NULL %f\n", max_weight);
    */
    return NULL;
}

ggml_backend_t get_backend(ggml_tensor *x)
{
    ggml_backend_t backend_res = ggml_backend_sched_get_node_backend(current_context->sched, x);
    return backend_res;
}

// decode a batch of tokens by evaluating the transformer
//
//   - lctx:      llama context
//   - batch:     batch to evaluate
//
// return 0 on success
// return positive int on warning
// return negative int on error
//
static int llama_decode_internal(
         llama_context & lctx,
           llama_batch   batch) {
    const uint32_t n_tokens = batch.n_tokens;

    if (n_tokens == 0) {
        return 0;
    }

    if( lctx.ctx_ready == false ) {
        LLAMA_LOG_ERROR("%s: context not ready!", __func__);
        return -1;
    }

/*    if( n_tokens > 1 ) {
        LLAMA_LOG_INFO("%s: decode_internal n_ctx=%d, n_tokens = %d, seq_start=%d, seq_end=%d, ctx_ready = %s\n", __func__, lctx.kv_self->size, n_tokens, lctx.sequential_start, lctx.seq_end, lctx.ctx_ready?"true":"false");
    } else {
        LLAMA_LOG_INFO("%s: decode_internal n_ctx=%d, token=%d (%s), seq_start=%d, seq_end=%d, ctx_ready = %s\n", __func__, lctx.kv_self->size, batch.token[0], llama_token_get_text(&lctx.model, batch.token[0]), lctx.sequential_start, lctx.seq_end, lctx.ctx_ready?"true":"false");
    }*/

    //LLAMA_LOG_INFO("%s: context %p\n", __func__, &lctx);

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;
    const auto n_batch = cparams.n_batch;

    GGML_ASSERT(n_tokens <= n_batch);
    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT

    int n_threads = n_tokens == 1 ? cparams.n_threads : cparams.n_threads_batch;
    const int64_t t_start_us = ggml_time_us();

#ifdef GGML_USE_MPI
    // TODO: needs fix after #3228
    GGML_ASSERT(false && "not implemented");
    //ggml_mpi_eval_init(lctx.ctx_mpi, &n_tokens, &n_past, &n_threads);
#endif

    GGML_ASSERT(n_threads > 0);

    auto & kv_self = lctx.kv_self;
    const int64_t n_embd  = hparams.n_embd;
    const int64_t n_vocab = hparams.n_vocab;

    LLAMA_LOG_INFO("%s: kv_self %p (%p)\n", __func__, &lctx, lctx.sched);
    ggml_backend_sched_reset(lctx.sched);
    ggml_backend_sched_set_eval_callback(lctx.sched, lctx.cparams.cb_eval, lctx.cparams.cb_eval_user_data);
    //LLAMA_LOG_INFO("%s: kv_self\n", __func__);

    //lctx.kv_self->use_tokens(lctx.seq_end, batch.n_tokens);

    //LLAMA_LOG_INFO("%s: build graph\n", __func__);
    ggml_cgraph * gf = llama_build_graph(lctx, batch, false);

    // the output is always the last tensor in the graph
    struct ggml_tensor * res  = gf->nodes[gf->n_nodes - 1];
    struct ggml_tensor * embd = gf->nodes[gf->n_nodes - 2];
    struct ggml_tensor * out_embds = NULL;

    if( lctx.record_all == 2 ) {
        out_embds = gf->nodes[gf->n_nodes - 3];
    }

    //LLAMA_LOG_INFO("%s: do attention\n", __func__);
    if (!hparams.causal_attn) {
        res = nullptr; // do not extract logits for embedding models such as BERT

        // token or sequence embeddings
        embd = gf->nodes[gf->n_nodes - 1];
        if( lctx.record_all == 2 ) {
            out_embds = gf->nodes[gf->n_nodes - 2];
        }

        GGML_ASSERT(strcmp(embd->name, "result_embd") == 0 || strcmp(embd->name, "result_embd_pooled") == 0);
    } else {
        if (strcmp(res->name, "result_output") == 0) {
            // the token embeddings could be the second to last tensor, or the third to last tensor
            if (strcmp(embd->name, "result_norm") != 0) {
                embd = gf->nodes[gf->n_nodes - 3];
                if( lctx.record_all == 2 ) {
                    out_embds = gf->nodes[gf->n_nodes - 4];
                }
                GGML_ASSERT(strcmp(embd->name, "result_norm") == 0);
            }
        } else {
            GGML_ASSERT(false && "missing result_output tensor");
        }
    }

    //LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    // TODO: this is mostly important for Apple Silicon where CBLAS is still performing very well
    //       we still need some threads to process all non-mul_mat ops, but not too much to avoid interfering
    //       with the BLAS calls. need a better solution
    // MoE Special Case: This logic applies when hparams.n_expert == 0, i.e. the model is NOT an MoE model. When an MoE is
    //                   being processed then Accelerate/BLAS will not be involved, so capping would limit performance.
    if (n_tokens >= 32 && hparams.n_expert == 0 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas()) {
        n_threads = std::min(4, n_threads);
    }
    //LLAMA_LOG_INFO("%s: set inputs (%u)\n", __func__, n_tokens);

    llama_set_inputs(lctx, batch);

    //LLAMA_LOG_INFO("%s: run compute\n", __func__);
    llama_graph_compute(lctx, gf, n_threads);

    LLAMA_LOG_INFO("%s: compute done\n", __func__);

    lctx.seq_end += batch.n_tokens;


#ifdef GGML_PERF
    // print timing information per ggml operation (for debugging purposes)
    // requires GGML_PERF to be defined
    ggml_graph_print(gf);
#endif

    // plot the computation graph in dot format (for debugging purposes)
    //if (n_past%100 == 0) {
    //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
    //}

    // extract logits
    if (res) {
        auto & logits_out = lctx.logits;

#ifndef NDEBUG
        auto & logits_valid = lctx.logits_valid;
        logits_valid.clear();
        logits_valid.resize(n_tokens);

        logits_out.clear();
#endif

        //LLAMA_LOG_INFO("%s: get backend\n", __func__);
        ggml_backend_t backend_res = ggml_backend_sched_get_node_backend(lctx.sched, res);
        GGML_ASSERT(backend_res != nullptr);

        if( lctx.record_all > 0 ) {
            //LLAMA_LOG_INFO("%s: record_logits(0..%d)\n", __func__, n_tokens);
            logits_out.resize(n_vocab * n_tokens);
            ggml_backend_tensor_get_async(backend_res, res, logits_out.data(), 0, n_vocab*n_tokens*sizeof(float));
        } else {
            //LLAMA_LOG_INFO("%s: record_logits([%d])\n", __func__, n_tokens);
            logits_out.resize(n_vocab);
            ggml_backend_tensor_get_async(backend_res, res, logits_out.data(), (n_vocab*(n_tokens-1))*sizeof(float), n_vocab*sizeof(float));
        }

#ifndef NDEBUG
//                logits_valid[i] = true;
#endif
        ggml_backend_synchronize(backend_res);
        //LLAMA_LOG_INFO("%s: backend synced\n", __func__);
    }

    // extract embeddings
    if (cparams.embeddings && embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_node_backend(lctx.sched, embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    auto & embd_out = lctx.embd;

                    if( lctx.record_all == 2 ) {
                        embd_out.resize(n_embd * 32);
                        ggml_backend_tensor_get_async(backend_embd, out_embds, embd_out.data(), 0, 32*n_embd*sizeof(float));
                    } else if( lctx.record_all == 1 ) {
                        embd_out.resize((n_embd*(n_tokens-1)));
                        ggml_backend_tensor_get_async(backend_embd, embd, embd_out.data(), (n_embd*(n_tokens-1))*sizeof(float), n_embd*sizeof(float));
                    } else {
                        embd_out.resize(n_embd);
                        ggml_backend_tensor_get_async(backend_embd, embd, embd_out.data(), (n_embd*(n_tokens-1))*sizeof(float), n_embd*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_MEAN:/*
                {
                    GGML_ASSERT(strcmp(embd->name, "result_embd_pooled") == 0);

                    // extract sequence embeddings
                    auto & embd_seq_out = lctx.embd_seq;
                    embd_seq_out.clear();

                    for (uint32_t i = 0; i < n_tokens; i++) {
                        const llama_seq_id seq_id = batch.seq_id[i][0];
                        if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                            continue;
                        }
                        embd_seq_out[seq_id].resize(n_embd);
                        ggml_backend_tensor_get_async(backend_embd, embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                    }
                } */
                break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ASSERT(false && "unknown pooling type");
                } break;
        }
        ggml_backend_synchronize(backend_embd);
        //LLAMA_LOG_INFO("%s: backend synced\n", __func__);
    }


    // measure the performance only for the single-token evals
    if (n_tokens == 1) {
        lctx.t_eval_us += ggml_time_us() - t_start_us;
        lctx.n_eval++;
    }
    else if (n_tokens > 1) {
        lctx.t_p_eval_us += ggml_time_us() - t_start_us;
        lctx.n_p_eval += n_tokens;
    }

    // get a more accurate load time, upon first eval
    // TODO: fix this
    if (!lctx.has_evaluated_once) {
        lctx.t_load_us = ggml_time_us() - lctx.t_start_us;
        lctx.has_evaluated_once = true;
    }

    return 0;
}

std::vector<ggml_backend_buffer_type_t> backend_buft;

void prepare_kv_cache(struct llama_context *ctx, int n_ctx, int n_batch)
{
    // build worst-case graph
    int n_tokens = (int)std::min(n_ctx, n_batch);
    int n_past = n_ctx - n_tokens;
    llama_token token = llama_token_bos(&ctx->model); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
    ggml_cgraph * gf = llama_build_graph(*ctx, llama_batch_get_one(&token, n_tokens, n_past, 0), true);

    // initialize scheduler with the worst-case graph
    if (!ggml_backend_sched_reserve(ctx->sched, gf)) {
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return;
    }

    for (size_t i = 0; i < ctx->backends.size(); i++) {
        ggml_backend *backend = ctx->backends[i];
        ggml_backend_buffer_type_t buft = backend_buft[i];
        size_t size = ggml_backend_sched_get_buffer_size(ctx->sched, backend);
        LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                ggml_backend_buft_name(buft),
                size / 1024.0 / 1024.0);
    }


    // note: the number of splits during measure is higher than during inference due to the kv shift
    int n_splits = ggml_backend_sched_get_n_splits(ctx->sched);
    LLAMA_LOG_INFO("%s: graph splits (measure): %d\n", __func__, n_splits);
}

// find holes from the beginning of the KV cache and fill them by moving data from the end of the cache
static void llama_kv_cache_defrag_internal(struct llama_context & lctx) {
    /*
    auto & kv_self = lctx.kv_self;

    const auto & hparams = lctx.model.hparams;

    const uint32_t n_layer = hparams.n_layer;

    const uint32_t n_kv   = llama_kv_cache_cell_max(kv_self);
    const uint32_t n_used = kv_self->used;

    assert(n_used <= n_kv);

    //const int64_t t_start = ggml_time_us();

    // number of cells moved
    uint32_t n_moves = 0;

    // determine which KV cells to move where
    //
    //  cell i moves to ids[i]
    //
    //  if ids[i] == i || ids[i] == n_kv, then cell i is not moved
    //
    std::vector<uint32_t> ids(n_kv, n_kv);

    for (uint32_t i0 = 0; i0 < n_used; ++i0) {
        const auto & cell0 = kv_self->cells[i0];

        if (!cell0.is_empty()) {
            ids[i0] = i0;

            continue;
        }

        // found a hole - fill it with data from the end of the cache

        uint32_t nh = 1;

        // determine the size of the hole
        while (i0 + nh < n_used && kv_self->cells[i0 + nh].is_empty()) {
            nh++;
        }

        // each move requires 6*n_layer tensors (see build_defrag)
        //   - source view, destination view, copy operation
        //   - x2 for keys and values
        //
        if (6*(n_moves + nh)*n_layer >= LLAMA_MAX_NODES) {
            // the graph is too big, we cannot move more cells
            break;
        }

        uint32_t nf = 0;
        uint32_t is = n_kv - 1;

        // starting from the end, find nh non-empty cells
        for (; is > i0; --is) {
            const auto & cell1 = kv_self->cells[is];

            if (cell1.is_empty() || ids[is] != n_kv) {
                continue;
            }

            // non-empty cell which is not yet moved
            nf++;

            if (nf == nh) {
                break;
            }
        }

        // this can only happen if `n_used` is not accurate, which would be a bug
        GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh");

        nf = 0;

        uint32_t i1 = is;

        // are we moving a continuous block of memory?
        bool cont = false;

        // go back and move the nf cells to the hole
        for (; i1 < n_kv; ++i1) {
            auto & cell1 = kv_self->cells[i1];

            if (cell1.is_empty() || ids[i1] != n_kv) {
                cont = false;
                continue;
            }

            // this cell goes to (i0 + nf)
            ids[i1] = i0 + nf;

            // move the cell meta data
            kv_self->cells[i0 + nf] = cell1;

            // clear the old cell and move the head there
            cell1 = llama_kv_cell();
            kv_self->head = n_used;

            if (!cont) {
                n_moves++;
                cont = true;
            }

            nf++;

            if (nf == nh) {
                break;
            }
        }

        //LLAMA_LOG_INFO("(tmp log) KV defrag: move [%u, %u) to [%u, %u)\n", is, i1 + 1, i0, i0 + nh);

        i0 += nh - 1;
    }

    if (n_moves == 0) {
        return;
    }

    */

    //LLAMA_LOG_INFO("(tmp log) KV defrag cell moves: %u\n", n_moves);

    //LLAMA_LOG_INFO("expected gf nodes: %u\n", 6*n_moves*n_layer);

#if 0
    // CPU defrag
    //
    // TODO: optimizations are possible:
    //       - multiple threads
    //       - avoid copying to the host memory when already there
    //
    // likely not worth the effort, as we have ggml_graph based defrag
    //

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();

    const uint32_t kv_size = kv_self->size;

    std::vector<uint8_t> buf_k;
    std::vector<uint8_t> buf_v;

    for (uint32_t il = 0; il < n_layer; ++il) {
        const size_t k_size_row = ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa);
        const size_t k_size     = ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*kv_size);

        const size_t v_size_el = ggml_type_size(kv_self->v_l[il]->type);
        const size_t v_size    = ggml_row_size (kv_self->v_l[il]->type, n_embd_v_gqa*kv_size);

        buf_k.resize(k_size);
        buf_v.resize(v_size);

        ggml_backend_tensor_get(kv_self->k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_get(kv_self->v_l[il], buf_v.data(), 0, buf_v.size());

        // batch move [i, i+nm) to [id, id+nm)
        // note: cells can move only to a lower index
        for (uint32_t i = 0; i < n_kv; ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == n_kv) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < n_kv && ids[i + nm] == id + nm) {
                nm++;
            }

            // move keys
            {
                const int64_t os =  i*k_size_row;
                const int64_t od = id*k_size_row;

                memcpy(buf_k.data() + od, buf_k.data() + os, nm*k_size_row);
            }

            // move values (note: they are transposed)
            {
                const int64_t os =  i;
                const int64_t od = id;

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    memcpy(buf_v.data() + (od + j*kv_size)*v_size_el, buf_v.data() + (os + j*kv_size)*v_size_el, nm*v_size_el);
                }
            }

            i += nm - 1;
        }

        ggml_backend_tensor_set(kv_self->k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_set(kv_self->v_l[il], buf_v.data(), 0, buf_v.size());
    }
#else
    // ggml_graph defrag

    /*
    ggml_cgraph * gf = llama_build_graph_defrag(lctx, ids);
    llama_graph_compute(lctx, gf, lctx.cparams.n_threads);
    */

#endif

    //const int64_t t_end = ggml_time_us();
    //LLAMA_LOG_INFO("(tmp log) KV defrag time: %.3f ms\n", (t_end - t_start)/1000.0);
}

static void llama_kv_cache_update_internal(struct llama_context & lctx) {
}

//
// tokenizer
//

static enum llama_vocab_type llama_vocab_get_type(const llama_vocab & vocab) {
    return vocab.type;
}

static bool llama_is_normal_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_NORMAL;
}

static bool llama_is_unknown_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_UNKNOWN;
}

static bool llama_is_control_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_CONTROL;
}

static bool llama_is_byte_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_BYTE;
}

static bool llama_is_user_defined_token(const llama_vocab& vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_USER_DEFINED;
}

static uint8_t llama_token_to_byte(const llama_vocab& vocab, llama_token id) {
    GGML_ASSERT(llama_is_byte_token(vocab, id));
    const auto& token_data = vocab.id_to_token.at(id);
    switch (llama_vocab_get_type(vocab)) {
        case LLAMA_VOCAB_TYPE_SPM: {
            auto buf = token_data.text.substr(3, 2);
            return strtol(buf.c_str(), NULL, 16);
        }
        case LLAMA_VOCAB_TYPE_BPE: {
            GGML_ASSERT(false);
            return unicode_to_bytes_bpe(token_data.text);
        }
        case LLAMA_VOCAB_TYPE_WPM: {
            GGML_ASSERT(false);
        }
        default:
            GGML_ASSERT(false);
    }
}

static llama_token llama_byte_to_token(const llama_vocab & vocab, uint8_t ch) {
    static const char * hex = "0123456789ABCDEF";
    switch (llama_vocab_get_type(vocab)) {
        case LLAMA_VOCAB_TYPE_SPM: {
            const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
            auto token = vocab.token_to_id.find(buf);
            if (token != vocab.token_to_id.end()) {
                return (*token).second;
            }
            // Try to fall back to just the byte as a string
            const char buf2[2] = { (char)ch, 0 };
            return vocab.token_to_id.at(buf2);
        }
        case LLAMA_VOCAB_TYPE_WPM:
        case LLAMA_VOCAB_TYPE_BPE: {
            return vocab.token_to_id.at(bytes_to_unicode_bpe(ch));
        }
        default:
            GGML_ASSERT(false);
    }
}

static void llama_escape_whitespace(std::string & text) {
    replace_all(text, " ", "\xe2\x96\x81");
}

static void llama_unescape_whitespace(std::string & word) {
    replace_all(word, "\xe2\x96\x81", " ");
}

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

// SPM tokenizer
// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};

struct llm_tokenizer_spm {
    llm_tokenizer_spm(const llama_vocab & vocab) : vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            size_t len = utf8_len(text[offs]);
            sym.text = text.c_str() + offs;
            sym.n = std::min(len, text.size() - offs);
            offs += sym.n;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
    }

private:
    void resegment(llm_symbol & symbol, std::vector<llama_vocab::id> & output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab.token_to_id.find(text);

        // Do we need to support is_unused?
        if (token != vocab.token_to_id.end()) {
            output.push_back((*token).second);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            output.reserve(output.size() + symbol.n);
            for (int j = 0; j < (int)symbol.n; ++j) {
                llama_vocab::id token_id = llama_byte_to_token(vocab, symbol.text[j]);
                output.push_back(token_id);
            }
            return;
        }

        resegment(symbols[p->second.first],  output);
        resegment(symbols[p->second.second], output);
    }

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.token_to_id.find(text);

        if (token == vocab.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab.id_to_token.size()) {
            return;
        }

        const auto & tok_data = vocab.id_to_token[(*token).second];

        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    const llama_vocab & vocab;

    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;

    std::map<std::string, std::pair<int, int>> rev_merge;
};

// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe {
    llm_tokenizer_bpe(const llama_vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        int final_prev_index = -1;
        auto word_collection = bpe_gpt2_preprocess(text);

        symbols_final.clear();

        for (auto & word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t) ::utf8_len(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (size_t i = 1; i < symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.top();
                work_queue.pop();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the fnished tokens to the final list keeping correct order for next and prev
            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto & symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.token_to_id.find(str);

                if (token == vocab.token_to_id.end()) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocab.token_to_id.find(byte_str);
                        if (token_multibyte == vocab.token_to_id.end()) {
                            throw std::runtime_error("ERROR: byte not found in vocab");
                        }
                        output.push_back((*token_multibyte).second);
                    }
                } else {
                    output.push_back((*token).second);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
    }

    std::vector<std::string> bpe_gpt2_preprocess(const std::string & text) {
        std::vector<std::string> bpe_words;
        std::vector<std::string> bpe_encoded_words;

        std::string token = "";
        // GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
        bool collecting_numeric = false;
        bool collecting_letter = false;
        bool collecting_special = false;
        bool collecting_whitespace_lookahead = false;
        bool collecting = false;

        std::vector<std::string> text_utf;
        text_utf.reserve(text.size());
        bpe_words.reserve(text.size());
        bpe_encoded_words.reserve(text.size());

        auto cps = codepoints_from_utf8(text);
        for (size_t i = 0; i < cps.size(); ++i)
            text_utf.emplace_back(codepoint_to_utf8(cps[i]));

        for (int i = 0; i < (int)text_utf.size(); i++) {
            const std::string & utf_char = text_utf[i];
            bool split_condition = false;
            int bytes_remain = text_utf.size() - i;
            // forward backward lookups
            const std::string & utf_char_next = (i + 1 < (int)text_utf.size()) ? text_utf[i + 1] : "";
            const std::string & utf_char_next_next = (i + 2 < (int)text_utf.size()) ? text_utf[i + 2] : "";

            // handling contractions
            if (!split_condition && bytes_remain >= 2) {
                // 's|'t|'m|'d
                if (utf_char == "\'" && (utf_char_next == "s" || utf_char_next == "t" || utf_char_next == "m" || utf_char_next == "d")) {
                    split_condition = true;
                }
                if (split_condition) {
                    if (token.size()) {
                        bpe_words.emplace_back(token); // push previous content as token
                    }
                    token = utf_char + utf_char_next;
                    bpe_words.emplace_back(token);
                    token = "";
                    i++;
                    continue;
                }
            }
            if (!split_condition && bytes_remain >= 3) {
                // 're|'ve|'ll
                if (utf_char == "\'" && (
                    (utf_char_next == "r" && utf_char_next_next == "e") ||
                    (utf_char_next == "v" && utf_char_next_next == "e") ||
                    (utf_char_next == "l" && utf_char_next_next == "l"))
                    ) {
                    split_condition = true;
                }
                if (split_condition) {
                    // current token + next token can be defined
                    if (token.size()) {
                        bpe_words.emplace_back(token); // push previous content as token
                    }
                    token = utf_char + utf_char_next + utf_char_next_next;
                    bpe_words.emplace_back(token); // the contraction
                    token = "";
                    i += 2;
                    continue;
                }
            }

            if (!split_condition && !collecting) {
                if (codepoint_type(utf_char) == CODEPOINT_TYPE_LETTER || (!token.size() && utf_char == " " && codepoint_type(utf_char_next) == CODEPOINT_TYPE_LETTER)) {
                    collecting_letter = true;
                    collecting = true;
                }
                else if (codepoint_type(utf_char) == CODEPOINT_TYPE_DIGIT || (!token.size() && utf_char == " " && codepoint_type(utf_char_next) == CODEPOINT_TYPE_DIGIT)) {
                    collecting_numeric = true;
                    collecting = true;
                }
                else if (
                    ((codepoint_type(utf_char) != CODEPOINT_TYPE_LETTER && codepoint_type(utf_char) != CODEPOINT_TYPE_DIGIT) && (codepoint_type(utf_char) != CODEPOINT_TYPE_WHITESPACE)) ||
                    (!token.size() && utf_char == " " && codepoint_type(utf_char_next) != CODEPOINT_TYPE_LETTER && codepoint_type(utf_char_next) != CODEPOINT_TYPE_DIGIT && codepoint_type(utf_char_next) != CODEPOINT_TYPE_WHITESPACE)
                    ) {
                    collecting_special = true;
                    collecting = true;
                }
                else if (codepoint_type(utf_char) == CODEPOINT_TYPE_WHITESPACE && codepoint_type(utf_char_next) == CODEPOINT_TYPE_WHITESPACE) {
                    collecting_whitespace_lookahead = true;
                    collecting = true;
                }
                else if (codepoint_type(utf_char) == CODEPOINT_TYPE_WHITESPACE) {
                    split_condition = true;
                }
            }
            else if (!split_condition && collecting) {
                if (collecting_letter && codepoint_type(utf_char) != CODEPOINT_TYPE_LETTER) {
                    split_condition = true;
                }
                else if (collecting_numeric && codepoint_type(utf_char) != CODEPOINT_TYPE_DIGIT) {
                    split_condition = true;
                }
                else if (collecting_special && (codepoint_type(utf_char) == CODEPOINT_TYPE_LETTER || codepoint_type(utf_char) == CODEPOINT_TYPE_DIGIT || codepoint_type(utf_char) == CODEPOINT_TYPE_WHITESPACE)) {
                    split_condition = true;
                }
                else if (collecting_whitespace_lookahead && (codepoint_type(utf_char_next) == CODEPOINT_TYPE_LETTER || codepoint_type(utf_char_next) == CODEPOINT_TYPE_DIGIT)) {
                    split_condition = true;
                }
            }

            if (utf_char_next == "") {
                split_condition = true; // final
                token += utf_char;
            }

            if (split_condition) {
                if (token.size()) {
                    bpe_words.emplace_back(token);
                }
                token = utf_char;
                collecting = false;
                collecting_letter = false;
                collecting_numeric = false;
                collecting_special = false;
                collecting_whitespace_lookahead = false;
            }
            else {
                token += utf_char;
            }
        }

        for (std::string & word : bpe_words) {
            std::string encoded_token = "";
            for (char & c : word) {
                encoded_token += bytes_to_unicode_bpe(c);
            }
            bpe_encoded_words.emplace_back(encoded_token);
        }

        return bpe_encoded_words;
    }

    const llama_vocab & vocab;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;

    llm_bigram_bpe::queue work_queue;
};

struct llm_tokenizer_wpm {
    llm_tokenizer_wpm(const llama_vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        auto * token_map = &vocab.token_to_id;

        // normalize and split by whitespace
        std::vector<std::string> words = preprocess(text);

        // bos token prepended already

        // find the longest tokens that form the words
        for (const std::string &word : words) {
            // skip empty words
            if (word.size() == 0) {
                continue;
            }

            // prepend phantom space
            std::string word1 = "\xe2\x96\x81" + word;
            int n = word1.size();

            // we're at the start of a new word
            int i = 0;
            bool match_any = false;

            // move through character position in word
            while (i < n) {
                // loop through possible match length
                bool match = false;
                for (int j = n; j > i; j--) {
                    auto it = token_map->find(word1.substr(i, j - i));
                    if (it != token_map->end()) {
                        output.push_back(it->second);
                        match = true;
                        match_any = true;
                        i = j;
                        break;
                    }
                }

                // must be an unknown character
                if (!match) {
                    i++;
                }
            }

            // we didn't find any matches for this word
            if (!match_any) {
                output.push_back(vocab.special_unk_id);
            }
        }

        // append eos token
        output.push_back(vocab.special_eos_id);
    }

    std::vector<std::string> preprocess(const std::string & text) {
        // normalalization form D
        std::vector<uint32_t> codepoints = codepoints_from_utf8(text);
        std::vector<uint32_t> nfd_codepoints;
        for (uint32_t code : codepoints) {
            auto it = nfd_map.equal_range(code);
            if (it.first != it.second) {
                for (auto jt = it.first; jt != it.second; jt++) {
                    nfd_codepoints.push_back(jt->second);
                }
            } else {
                nfd_codepoints.push_back(code);
            }
        }

        // strip accents, strip control, uniformize whitespace,
        // to lowercase, pad chinese characters, pad punctuation
        std::string new_str = "";
        for (uint32_t code : nfd_codepoints) {
            int type = codepoint_type(code);
            if (type == CODEPOINT_TYPE_ACCENT_MARK || type == CODEPOINT_TYPE_CONTROL) {
                continue;
            }
            code = unicode_tolower(code);
            if (type == CODEPOINT_TYPE_WHITESPACE) {
                code = ' ';
            }
            std::string s = codepoint_to_utf8(code);
            if (type == CODEPOINT_TYPE_PUNCTUATION || is_ascii_punct(code) || is_chinese_char(code)) {
                new_str += " ";
                new_str += s;
                new_str += " ";
            } else {
                new_str += s;
            }
        }

        // split by whitespace
        uint64_t l = 0;
        uint64_t r = 0;
        std::vector<std::string> words;
        while (r < new_str.size()) {
            // if is whitespace
            if (isspace(new_str[r], std::locale::classic())) {
                if (r > l) words.push_back(new_str.substr(l, (r - l)));
                l = r + 1;
                r = l;
            }
            else {
                r += 1;
            }
        }
        if (r > l) {
            words.push_back(new_str.substr(l, (r - l)));
        }
        return words;
    }

    bool is_ascii_punct(uint32_t code) {
        if (code > 0xFF) {
            return false;
        }
        auto c = char(static_cast<unsigned char>(code));
        return ispunct(c, std::locale::classic());
    }

    bool is_chinese_char(uint32_t codepoint) {
        if ((codepoint >= 0x4E00  && codepoint <= 0x9FFF)  ||
            (codepoint >= 0x3400  && codepoint <= 0x4DBF)  ||
            (codepoint >= 0x20000 && codepoint <= 0x2A6DF) ||
            (codepoint >= 0x2A700 && codepoint <= 0x2B73F) ||
            (codepoint >= 0x2B740 && codepoint <= 0x2B81F) ||
            (codepoint >= 0x2B920 && codepoint <= 0x2CEAF) || // this should be 0x2B820 but in hf rust code it is 0x2B920
            (codepoint >= 0xF900  && codepoint <= 0xFAFF)  ||
            (codepoint >= 0x2F800 && codepoint <= 0x2FA1F) ||
            (codepoint >= 0x3000  && codepoint <= 0x303F)  ||
            (codepoint >= 0xFF00  && codepoint <= 0xFFEF)) {
            return true; // NOLINT
        }
        return false;
    }

    const llama_vocab & vocab;
};

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(llama_vocab::id _token)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0) {}

    fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((llama_vocab::id) - 1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length){
            GGML_ASSERT(_offset >= 0);
            GGML_ASSERT(_length >= 1);
            GGML_ASSERT(offset + length <= raw_text.length());
        }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const llama_vocab::id token;
    const std::string _dummy;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};

// #define PRETOKENIZERDEBUG

static void tokenizer_st_partition(const llama_vocab & vocab, std::forward_list<fragment_buffer_variant> & buffer) {
    // for each special token
    for (const auto & st: vocab.special_tokens_cache) {
        const auto & special_token = st.first;
        const auto & special_id    = st.second;

        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                auto * raw_text = &(fragment.raw_text);

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true) {
                    // find the first occurrence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    auto match = raw_text->find(special_token, raw_text_base_offset);

                    // no occurrences found, stop processing this fragment for a given special token
                    if (match == std::string::npos) break;

                    // check if match is within bounds of offset <-> length
                    if (match + special_token.length() > raw_text_base_offset + raw_text_base_length) break;

#ifdef PRETOKENIZERDEBUG
                    LLAMA_LOG_WARN("FF: (%ld %ld %ld) '%s'\n", raw_text->length(), raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        const int64_t left_reminder_length = match - raw_text_base_offset;
                        buffer.emplace_after(it, (*raw_text), left_reminder_offset, left_reminder_length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length, raw_text->substr(left_reminder_offset, left_reminder_length).c_str());
#endif
                        it++;
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + special_token.length() < raw_text_base_offset + raw_text_base_length) {
                        const int64_t right_reminder_offset = match + special_token.length();
                        const int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + special_token.length());
                        buffer.emplace_after(it, (*raw_text), right_reminder_offset, right_reminder_length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length, raw_text->substr(right_reminder_offset, right_reminder_length).c_str());
#endif

                        it++;

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source-1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    } else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source-1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

void llama_quick_tokenize( std::string raw_text, std::vector<llama_vocab::id> &output )
{
    if (raw_text.empty()) {
        return;
    }
    const llama_vocab &vocab = current_model->vocab;

    std::forward_list<fragment_buffer_variant> fragment_buffer;
    fragment_buffer.emplace_front(raw_text, 0, raw_text.length());

    for (const auto & fragment : fragment_buffer) {
        if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
            // without adding this leading whitespace, we do not get the same results as the original tokenizer

            // TODO: It's likely possible to get rid of this string copy entirely
            //  by modifying llm_tokenizer_x to operate with string offsets like pre-tokenizer
            //  and passing 'add space prefix' as bool argument
            //
            auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

            llm_tokenizer_spm tokenizer(vocab);
            llama_escape_whitespace(raw_text);
            tokenizer.tokenize(raw_text, output);
        } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
            output.push_back(fragment.token);
        }
    }
}

static std::vector<llama_vocab::id> llama_tokenize_internal(const llama_vocab & vocab,
        std::string raw_text, bool bos, bool special) {
    std::vector<llama_vocab::id> output;

    // OG tokenizer behavior:
    //
    // tokenizer.encode('', add_bos=True)  returns [1]
    // tokenizer.encode('', add_bos=False) returns []

    if (bos && vocab.special_bos_id != -1) {
        output.push_back(vocab.special_bos_id);
    }

    if (raw_text.empty()) {
        return output;
    }

    std::forward_list<fragment_buffer_variant> fragment_buffer;
    fragment_buffer.emplace_front(raw_text, 0, raw_text.length());

    if (special) tokenizer_st_partition(vocab, fragment_buffer);

    switch (vocab.type) {
        case LLAMA_VOCAB_TYPE_SPM:
            {
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        // without adding this leading whitespace, we do not get the same results as the original tokenizer

                        // TODO: It's likely possible to get rid of this string copy entirely
                        //  by modifying llm_tokenizer_x to operate with string offsets like pre-tokenizer
                        //  and passing 'add space prefix' as bool argument
                        //
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);
#if 0
                        if (&fragment == &fragment_buffer.front()) {
                            if (vocab.add_space_prefix) {
                                raw_text = " " + raw_text; // prefix with space if the first token is not special
                            }
                        }
#endif

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        llm_tokenizer_spm tokenizer(vocab);
                        llama_escape_whitespace(raw_text);
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
        case LLAMA_VOCAB_TYPE_BPE:
            {
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        llm_tokenizer_bpe tokenizer(vocab);
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
        case LLAMA_VOCAB_TYPE_WPM:
            {
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        llm_tokenizer_wpm tokenizer(vocab);
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
    }

    return output;
}

//
// grammar - internal
//

struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar {
    const std::vector<std::vector<llama_grammar_element>>   rules;
    std::vector<std::vector<const llama_grammar_element *>> stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8                                      partial_utf8;
};

struct llama_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_partial_utf8   partial_utf8;
};

// Decodes a UTF-8 string which may end in an incomplete sequence. Adds a terminating 0 for use as
// pointer. If an invalid sequence is encountered, returns `llama_partial_utf8.n_remain == -1`.
static std::pair<std::vector<uint32_t>, llama_partial_utf8> decode_utf8(
        const std::string & src,
        llama_partial_utf8   partial_start) {
    static const int      lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    const char          * pos      = src.c_str();
    std::vector<uint32_t> code_points;
    // common english strings have the same number of codepoints and bytes. `+ 1` for the terminating 0.
    code_points.reserve(src.size() + 1);
    uint32_t              value    = partial_start.value;
    int                   n_remain = partial_start.n_remain;

    // continue previous decode, if applicable
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            // invalid sequence, abort
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{ 0, -1 });
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }

    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // decode any subsequent utf-8 sequences, which may end in an incomplete one
    while (*pos != 0) {
        uint8_t  first_byte = static_cast<uint8_t>(*pos);
        uint8_t  highbits   = first_byte >> 4;
                 n_remain   = lookup[highbits] - 1;

        if (n_remain < 0) {
            // invalid sequence, abort
            code_points.clear();
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{ 0, n_remain });
        }

        uint8_t  mask       = (1 << (7 - n_remain)) - 1;
                 value      = first_byte & mask;
        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }
    code_points.push_back(0);

    return std::make_pair(std::move(code_points), llama_partial_utf8{ value, n_remain });
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool llama_grammar_is_end_of_sequence(const llama_grammar_element * pos) {
    switch (pos->type) {
        case LLAMA_GRETYPE_END: return true;  // NOLINT
        case LLAMA_GRETYPE_ALT: return true;  // NOLINT
        default:                return false;
    }
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static std::pair<bool, const llama_grammar_element *> llama_grammar_match_char(
        const llama_grammar_element * pos,
        const uint32_t                chr) {

    bool found            = false;
    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR;

    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT); // NOLINT

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return std::make_pair(found == is_positive_char, pos);
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static bool llama_grammar_match_partial_char(
        const llama_grammar_element * pos,
        const llama_partial_utf8      partial_utf8) {

    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR;
    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT);

    uint32_t partial_value = partial_utf8.value;
    int      n_remain      = partial_utf8.n_remain;

    // invalid sequence or 7-bit char split across 2 bytes (overlong)
    if (n_remain < 0 || (n_remain == 1 && partial_value < 2)) {
        return false;
    }

    // range of possible code points this partial UTF-8 sequence could complete to
    uint32_t low  = partial_value << (n_remain * 6);
    uint32_t high = low | ((1 << (n_remain * 6)) - 1);

    if (low == 0) {
        if (n_remain == 2) {
            low = 1 << 11;
        } else if (n_remain == 3) {
            low = 1 << 16;
        }
    }

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            if (pos->value <= high && low <= pos[1].value) {
                return is_positive_char;
            }
            pos += 2;
        } else {
            // exact char match, e.g. [a] or "a"
            if (low <= pos->value && pos->value <= high) {
                return is_positive_char;
            }
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return !is_positive_char;
}


// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static void llama_grammar_advance_stack(
        const std::vector<std::vector<llama_grammar_element>>   & rules,
        const std::vector<const llama_grammar_element *>        & stack,
        std::vector<std::vector<const llama_grammar_element *>> & new_stacks) {

    if (stack.empty()) {
        new_stacks.emplace_back(stack);
        return;
    }

    const llama_grammar_element * pos = stack.back();

    switch (pos->type) {
        case LLAMA_GRETYPE_RULE_REF: {
            const size_t                  rule_id = static_cast<size_t>(pos->value);
            const llama_grammar_element * subpos  = rules[rule_id].data();
            do {
                // init new stack without the top (pos)
                std::vector<const llama_grammar_element *> new_stack(stack.begin(), stack.end() - 1);
                if (!llama_grammar_is_end_of_sequence(pos + 1)) {
                    // if this rule ref is followed by another element, add that to stack
                    new_stack.push_back(pos + 1);
                }
                if (!llama_grammar_is_end_of_sequence(subpos)) {
                    // if alternate is nonempty, add to stack
                    new_stack.push_back(subpos);
                }
                llama_grammar_advance_stack(rules, new_stack, new_stacks);
                while (!llama_grammar_is_end_of_sequence(subpos)) {
                    // scan to end of alternate def
                    subpos++;
                }
                if (subpos->type == LLAMA_GRETYPE_ALT) {
                    // there's another alternate def of this rule to process
                    subpos++;
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case LLAMA_GRETYPE_CHAR:
        case LLAMA_GRETYPE_CHAR_NOT:
            new_stacks.emplace_back(stack);
            break;
        default:
            // end of alternate (LLAMA_GRETYPE_END, LLAMA_GRETYPE_ALT) or middle of char range
            // (LLAMA_GRETYPE_CHAR_ALT, LLAMA_GRETYPE_CHAR_RNG_UPPER); stack should never be left on
            // those
            GGML_ASSERT(false);
    }
}

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llama_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
static std::vector<std::vector<const llama_grammar_element *>> llama_grammar_accept(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const uint32_t                                                  chr) {

    std::vector<std::vector<const llama_grammar_element *>> new_stacks;

    for (const auto & stack : stacks) {
        if (stack.empty()) {
            continue;
        }

        auto match = llama_grammar_match_char(stack.back(), chr);
        if (match.first) {
            const llama_grammar_element * pos = match.second;

            // update top of stack to next element, if any
            std::vector<const llama_grammar_element *> new_stack(stack.begin(), stack.end() - 1);
            if (!llama_grammar_is_end_of_sequence(pos)) {
                new_stack.push_back(pos);
            }
            llama_grammar_advance_stack(rules, new_stack, new_stacks);
        }
    }

    return new_stacks;
}

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const std::vector<llama_grammar_candidate>                    & candidates);

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates_for_stack(
        const std::vector<std::vector<llama_grammar_element>> & rules,
        const std::vector<const llama_grammar_element *>      & stack,
        const std::vector<llama_grammar_candidate>            & candidates) {

    std::vector<llama_grammar_candidate> rejects;

    if (stack.empty()) {
        for (const auto & tok : candidates) {
            if (*tok.code_points != 0 || tok.partial_utf8.n_remain != 0) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }

    const llama_grammar_element * stack_pos = stack.back();

    std::vector<llama_grammar_candidate> next_candidates;
    for (const auto & tok : candidates) {
        if (*tok.code_points == 0) {
            // reached end of full codepoints in token, reject iff it ended in a partial sequence
            // that cannot satisfy this position in grammar
            if (tok.partial_utf8.n_remain != 0 &&
                    !llama_grammar_match_partial_char(stack_pos, tok.partial_utf8)) {
                rejects.push_back(tok);
            }
        } else if (llama_grammar_match_char(stack_pos, *tok.code_points).first) {
            next_candidates.push_back({ tok.index, tok.code_points + 1, tok.partial_utf8 });
        } else {
            rejects.push_back(tok);
        }
    }

    const auto * stack_pos_after = llama_grammar_match_char(stack_pos, 0).second;

    // update top of stack to next element, if any
    std::vector<const llama_grammar_element *> stack_after(stack.begin(), stack.end() - 1);
    if (!llama_grammar_is_end_of_sequence(stack_pos_after)) {
        stack_after.push_back(stack_pos_after);
    }
    std::vector<std::vector<const llama_grammar_element *>> next_stacks;
    llama_grammar_advance_stack(rules, stack_after, next_stacks);

    auto next_rejects = llama_grammar_reject_candidates(rules, next_stacks, next_candidates);
    for (const auto & tok : next_rejects) {
        rejects.push_back({ tok.index, tok.code_points - 1, tok.partial_utf8 });
    }

    return rejects;
}

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const std::vector<llama_grammar_candidate>                    & candidates) {
    GGML_ASSERT(!stacks.empty()); // REVIEW

    if (candidates.empty()) {
        return std::vector<llama_grammar_candidate>();
    }

    auto rejects = llama_grammar_reject_candidates_for_stack(rules, stacks.front(), candidates);

    for (size_t i = 1, size = stacks.size(); i < size; ++i) {
        rejects = llama_grammar_reject_candidates_for_stack(rules, stacks[i], rejects);
    }
    return rejects;
}

//
// grammar - external
//

struct llama_grammar * llama_grammar_init(
            const llama_grammar_element ** rules,
                                 size_t    n_rules,
                                 size_t    start_rule_index) {
    const llama_grammar_element * pos;

    // copy rule definitions into vectors
    std::vector<std::vector<llama_grammar_element>> vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
    }

    // loop over alternates of start rule to build initial stacks
    std::vector<std::vector<const llama_grammar_element *>> stacks;
    pos = rules[start_rule_index];
    do {
        std::vector<const llama_grammar_element *> stack;
        if (!llama_grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        llama_grammar_advance_stack(vec_rules, stack, stacks);
        while (!llama_grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == LLAMA_GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    return new llama_grammar{ std::move(vec_rules), std::move(stacks), {} };
}

void llama_grammar_free(struct llama_grammar * grammar) {
    delete grammar;
}

struct llama_grammar * llama_grammar_copy(const struct llama_grammar * grammar) {
    llama_grammar * result = new llama_grammar{ grammar->rules, grammar->stacks, grammar->partial_utf8 };

    // redirect elements in stacks to point to new rules
    for (size_t is = 0; is < result->stacks.size(); is++) {
        for (size_t ie = 0; ie < result->stacks[is].size(); ie++) {
            for (size_t ir0 = 0; ir0 < grammar->rules.size(); ir0++) {
                for (size_t ir1 = 0; ir1 < grammar->rules[ir0].size(); ir1++) {
                    if (grammar->stacks[is][ie] == &grammar->rules[ir0][ir1]) {
                         result->stacks[is][ie]  =  &result->rules[ir0][ir1];
                    }
                }
            }
        }
    }

    return result;
}

//
// sampling
//

void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = time(NULL);
    }
    ctx->rng.seed(seed);
}

void llama_sample_softmax(struct llama_context * ctx, llama_token_data_array * candidates) {
    GGML_ASSERT(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_time_us();

    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_k(struct llama_context * ctx, llama_token_data_array * candidates, int32_t k, size_t min_keep) {
    // TODO: move bucket sort to separate function so that top_p/tail_free/typical/softmax first is equally fast
    // if (k >= (int32_t)candidates->size) {
    //     return;
    // }

    const int64_t t_start_sample_us = ggml_time_us();

    if (k <= 0) {
        k = candidates->size;
    }

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        };
        if (k <= 128) {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        } else {
            constexpr int   nbuckets     = 128;
            constexpr float bucket_low   = -10.0f;
            constexpr float bucket_high  =  10.0f;
            constexpr float bucket_scale = nbuckets/(bucket_high - bucket_low);
            constexpr float bucker_inter = -bucket_low * bucket_scale;

            std::vector<int> bucket_idx(candidates->size);
            std::vector<int> histo(nbuckets, 0);

            for (int i = 0; i < (int)candidates->size; ++i) {
                const float val = candidates->data[i].logit;
                int ib = int(bucket_scale * val + bucker_inter); //nbuckets * (val - bucket_low) / (bucket_high - bucket_low);
                ib = std::max(0, std::min(nbuckets-1, ib));
                bucket_idx[i] = ib;
                ++histo[ib];
            }
            int nhave = 0;
            int ib = nbuckets - 1;
            for ( ; ib >= 0; --ib) {
                nhave += histo[ib];
                if (nhave >= k) break;
            }
            std::vector<llama_token_data> tmp_tokens(nhave);
            auto ptr = tmp_tokens.data();
            std::vector<llama_token_data*> bucket_ptrs;
            bucket_ptrs.reserve(nbuckets - ib);
            for (int j = nbuckets - 1; j >= ib; --j) {
                bucket_ptrs.push_back(ptr);
                ptr += histo[j];
            }
            for (int i = 0; i < (int)candidates->size; ++i) {
                int j = bucket_idx[i];
                if (j >= ib) {
                    *bucket_ptrs[nbuckets-1-j]++ = candidates->data[i];
                }
            }

            ptr = tmp_tokens.data();
            int ndone = 0;
            for (int j = nbuckets-1; j > ib; --j) {
                std::sort(ptr, ptr + histo[j], comp);
                ptr += histo[j];
                ndone += histo[j];
            }
            std::partial_sort(ptr, ptr + k - ndone, ptr + histo[ib], comp);

            std::memcpy(candidates->data, tmp_tokens.data(), k*sizeof(llama_token_data));

        }
        candidates->sorted = true;
    }
    candidates->size = k;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    llama_sample_softmax(ctx, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_min_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p <= 0.0f || !candidates->size) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    bool min_p_applied = false;

    // if the candidates aren't sorted, try the unsorted implementation first
    if (!candidates->sorted) {
        std::vector<llama_token_data> filtered_tokens;

        float max_logit = -FLT_MAX;
        for (size_t i = 0; i < candidates->size; ++i) {
            max_logit = std::max(max_logit, candidates->data[i].logit);
        }
        const float min_logit = max_logit + logf(p); // min logit for p_i >= p * p_max

        for (size_t i = 0; i < candidates->size; ++i) {
            if (candidates->data[i].logit >= min_logit) {
                filtered_tokens.push_back(candidates->data[i]);
            }
        }

        // if we have enough values the operation was a success
        if (filtered_tokens.size() >= min_keep) {
            memcpy(candidates->data, filtered_tokens.data(), filtered_tokens.size()*sizeof(llama_token_data));
            candidates->size = filtered_tokens.size();
            min_p_applied = true;
        }
    }

    // if the candidates are sorted or the unsorted implementation failed, use this implementation
    if (!min_p_applied) {
        // Sort the logits in descending order
        if (!candidates->sorted) {
            std::sort(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
                return a.logit > b.logit;
            });
            candidates->sorted = true;
        }

        const float min_logit = candidates->data[0].logit + logf(p); // min logit for p_i >= p * p_max
        size_t i = 1; // first token always matches

        for (; i < candidates->size; ++i) {
            if (candidates->data[i].logit < min_logit && i >= min_keep) {
                break; // prob too small
            }
        }

        // Resize the output vector to keep only the matching tokens
        candidates->size = i;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_tail_free(struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    llama_sample_softmax(nullptr, candidates);
    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = std::abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    {
        const float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);

        if (second_derivatives_sum > 1e-6f) {
            for (float & value : second_derivatives) {
                value /= second_derivatives_sum;
            }
        } else {
            for (float & value : second_derivatives) {
                value = 1.0f / second_derivatives.size();
            }
        }
    }

    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_typical(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_sample_softmax(nullptr, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < candidates->size; ++i) {
        float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_token_data> new_candidates;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates.push_back(candidates->data[idx]);
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();
    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_entropy(struct llama_context * ctx, llama_token_data_array * candidates_p, float min_temp, float max_temp, float exponent_val) {
    const int64_t t_start_sample_us = ggml_time_us();

    // no need to do anything if there is only one (or zero) candidates
    if(candidates_p->size <= 1) {
        return;
    }

    // Calculate maximum possible entropy
    float max_entropy = -logf(1.0f / candidates_p->size);

    llama_sample_softmax(nullptr, candidates_p);

    // Calculate entropy of the softmax probabilities
    float entropy = 0.0f;
    for (size_t i = 0; i < candidates_p->size; ++i) {
        float prob = candidates_p->data[i].p;
        if (prob > 0.0f) { // Ensure no log(0)
            entropy -= prob * logf(prob);
        }
    }

    // Normalize the entropy (max_entropy cannot be 0 here because we checked candidates_p->size != 1 above)
    float normalized_entropy = entropy / max_entropy;

    // Map the normalized entropy to the desired temperature range using the power function
    float dyn_temp = min_temp + (max_temp - min_temp) * powf(normalized_entropy, exponent_val);

#ifdef DEBUG
    LLAMA_LOG_INFO("Your text maxtemp value is: %f\n", max_temp);
    LLAMA_LOG_INFO("Entropy: %f\n", entropy);
    LLAMA_LOG_INFO("Max Possible Entropy: %f\n", max_entropy);
    LLAMA_LOG_INFO("Normalized Entropy: %f\n", normalized_entropy);
    LLAMA_LOG_INFO("Exponent: %f\n", exponent_val);
    LLAMA_LOG_INFO("Dynamic Temperature (dyn_temp): %f\n", dyn_temp);
#endif

    // Apply the dynamically calculated temperature scaling
    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= dyn_temp;
    }

    // Re-compute softmax probabilities after scaling logits with dynamic temperature
    double max_l_double = candidates_p->data[0].logit;
    double cum_sum_double = 0.0;
    for (size_t i = 0; i < candidates_p->size; ++i) {
        double p = exp(candidates_p->data[i].logit - max_l_double);
        candidates_p->data[i].p = p; // Store the scaled probability
        cum_sum_double += p;
    }
    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].p /= cum_sum_double; // Re-normalize the probabilities
    }

#ifdef DEBUG
    // Print the updated top 25 probabilities after temperature scaling
    LLAMA_LOG_INFO("\nUpdated Top 25 Probabilities After Dynamic Temperature Scaling (in percentages):\n");
    for (size_t i = 0; i < 25 && i < candidates_p->size; ++i) {
        LLAMA_LOG_INFO("Token %zu: %f%%\n", i + 1, candidates_p->data[i].p * 100.0f);
    }
#endif

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_temp(struct llama_context * ctx, llama_token_data_array * candidates_p, float temp) {
    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= temp;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_repetition_penalties(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
               const llama_token * last_tokens,
                          size_t   penalty_last_n,
                           float   penalty_repeat,
                           float   penalty_freq,
                           float   penalty_present) {
    if (penalty_last_n == 0 || (penalty_repeat == 1.0f && penalty_freq == 0.0f && penalty_present == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<llama_token, int> token_count;
    for (size_t i = 0; i < penalty_last_n; ++i) {
        token_count[last_tokens[i]]++;
    }

    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        const auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty_repeat;
        } else {
            candidates->data[i].logit /= penalty_repeat;
        }

        candidates->data[i].logit -= float(count) * penalty_freq + float(count > 0) * penalty_present;
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_grammar(struct llama_context * ctx, llama_token_data_array * candidates, const struct llama_grammar * grammar) {
    GGML_ASSERT(ctx);
    const int64_t t_start_sample_us = ggml_time_us();

    bool allow_eos = false;
    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            allow_eos = true;
            break;
        }
    }

    const llama_token eos = llama_token_eos(&ctx->model);

    std::vector<std::pair<std::vector<uint32_t>, llama_partial_utf8>> candidates_decoded;
    candidates_decoded.reserve(candidates->size);
    std::vector<llama_grammar_candidate>                              candidates_grammar;
    candidates_grammar.reserve(candidates->size);

    for (size_t i = 0; i < candidates->size; ++i) {
        const llama_token id    = candidates->data[i].id;
        const std::string piece = llama_token_to_piece(ctx, id);
        if (id == eos) {
            if (!allow_eos) {
                candidates->data[i].logit = -INFINITY;
            }
        } else if (piece.empty() || piece[0] == 0) {
            candidates->data[i].logit = -INFINITY;
        } else {
            candidates_decoded.push_back(decode_utf8(piece, grammar->partial_utf8));
            candidates_grammar.push_back({ i, candidates_decoded.back().first.data(), candidates_decoded.back().second });
        }
    }

    const auto rejects = llama_grammar_reject_candidates(grammar->rules, grammar->stacks, candidates_grammar);
    for (const auto & reject : rejects) {
        candidates->data[reject.index].logit = -INFINITY;
    }

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
}

static void llama_log_softmax(float * array, size_t size) {
    float max_l = *std::max_element(array, array + size);
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float p = expf(array[i] - max_l);
        sum += p;
        array[i] = p;
    }

    for (size_t i = 0; i < size; ++i) {
        array[i] = logf(array[i] / sum);
    }
}

void llama_sample_apply_guidance(
          struct llama_context * ctx,
                         float * logits,
                         float * logits_guidance,
                         float   scale) {
    GGML_ASSERT(ctx);

    const auto t_start_sample_us = ggml_time_us();
    const auto n_vocab = llama_n_vocab(llama_get_model(ctx));

    llama_log_softmax(logits, n_vocab);
    llama_log_softmax(logits_guidance, n_vocab);

    for (int i = 0; i < n_vocab; ++i) {
              auto & l = logits[i];
        const auto & g = logits_guidance[i];

        l = scale * (l - g) + g;
    }

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
}

llama_token llama_sample_token_mirostat(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int32_t m, float * mu) {
    GGML_ASSERT(ctx);

    auto N = float(llama_n_vocab(llama_get_model(ctx)));
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax(nullptr, candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    llama_sample_top_k(nullptr, candidates, int(k), 1);
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    llama_token X = llama_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_mirostat_v2(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu) {
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax(ctx, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    if (candidates->size == 0) {
        candidates->size = 1;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }

    // Normalize the probabilities of the remaining words
    llama_sample_softmax(ctx, candidates);

    // Sample the next word X from the remaining words
    llama_token X = llama_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_greedy(struct llama_context * ctx, llama_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_time_us();

    // Find max element
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit < b.logit;
    });

    llama_token result = max_iter->id;
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return result;
}

llama_token llama_sample_token(struct llama_context * ctx, llama_token_data_array * candidates) {
    GGML_ASSERT(ctx);

    const int64_t t_start_sample_us = ggml_time_us();
    llama_sample_softmax(nullptr, candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    auto & rng = ctx->rng;
    int idx = dist(rng);

    llama_token result = candidates->data[idx].id;

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
    return result;
}

void llama_grammar_accept_token(struct llama_context * ctx, struct llama_grammar * grammar, llama_token token) {
    const int64_t t_start_sample_us = ggml_time_us();

    if (token == llama_token_eos(&ctx->model)) {
        for (const auto & stack : grammar->stacks) {
            if (stack.empty()) {
                return;
            }
        }
        GGML_ASSERT(false);
    }

    const std::string piece = llama_token_to_piece(ctx, token);

    // Note terminating 0 in decoded string
    const auto   decoded     = decode_utf8(piece, grammar->partial_utf8);
    const auto & code_points = decoded.first;
    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        grammar->stacks = llama_grammar_accept(grammar->rules, grammar->stacks, *it);
    }
    grammar->partial_utf8 = decoded.second;
    GGML_ASSERT(!grammar->stacks.empty());

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
}

//
// Beam search
//

struct llama_beam {
    std::vector<llama_token> tokens;
    float p;  // Cumulative beam probability (renormalized relative to all beams)
    bool eob; // Initialize end-of-beam to false. Callback sets this to true.
    // Sort beams by probability. In case of ties, prefer beams at eob.
    bool operator<(const llama_beam & rhs) const {
        return std::make_pair(p, eob) < std::make_pair(rhs.p, rhs.eob);
    }
    // Shift off first n tokens and discard them.
    void shift_tokens(const size_t n) {
        if (n) {
            std::copy(tokens.begin() + n, tokens.end(), tokens.begin());
            tokens.resize(tokens.size() - n);
        }
    }
    llama_beam_view view() const { return {tokens.data(), tokens.size(), p, eob}; }
};

// A struct for calculating logit-related info.
struct llama_logit_info {
    const float * const logits;
    const int n_vocab;
    const float max_l;
    const float normalizer;
    struct sum_exp {
        float max_l;
        float operator()(float sum, float l) const { return sum + std::exp(l - max_l); }
    };
    llama_logit_info(llama_context * ctx)
      : logits(llama_get_logits(ctx))
      , n_vocab(llama_n_vocab(llama_get_model(ctx)))
      , max_l(*std::max_element(logits, logits + n_vocab))
      , normalizer(1.0f / std::accumulate(logits, logits + n_vocab, 0.0f, sum_exp{max_l}))
      { }
    llama_token_data get_token_data(const llama_token token_id) const {
        constexpr auto p = std::numeric_limits<float>::quiet_NaN();  // never used
        return {token_id, logits[token_id], p};
    }
    // Return top k token_data by logit.
    std::vector<llama_token_data> top_k(size_t k) {
        std::vector<llama_token_data> min_heap;  // min-heap by logit
        const llama_token k_min = std::min(static_cast<llama_token>(k), n_vocab);
        min_heap.reserve(k_min);
        for (llama_token token_id = 0 ; token_id < k_min ; ++token_id) {
            min_heap.push_back(get_token_data(token_id));
        }
        auto comp = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; };
        std::make_heap(min_heap.begin(), min_heap.end(), comp);
        for (llama_token token_id = k_min ; token_id < n_vocab ; ++token_id) {
            if (min_heap.front().logit < logits[token_id]) {
                std::pop_heap(min_heap.begin(), min_heap.end(), comp);
                min_heap.back().id = token_id;
                min_heap.back().logit = logits[token_id];
                std::push_heap(min_heap.begin(), min_heap.end(), comp);
            }
        }
        return min_heap;
    }
    float probability_from_logit(float logit) const {
        return normalizer * std::exp(logit - max_l);
    }
};

struct llama_beam_search_data {
    llama_context * ctx;
    size_t n_beams;
    int n_past;
    int n_predict;
    std::vector<llama_beam> beams;
    std::vector<llama_beam> next_beams;

    // Re-calculated on each loop iteration
    size_t common_prefix_length;

    // Used to communicate to/from callback on beams state.
    std::vector<llama_beam_view> beam_views;

    llama_beam_search_data(llama_context * ctx, size_t n_beams, int n_past, int n_predict)
      : ctx(ctx)
      , n_beams(n_beams)
      , n_past(n_past)
      , n_predict(n_predict)
      , beam_views(n_beams) {
        beams.reserve(n_beams);
        next_beams.reserve(n_beams);
    }

    // Collapse beams to a single beam given by index.
    void collapse_beams(const size_t beam_idx) {
        if (0u < beam_idx) {
            std::swap(beams[0], beams[beam_idx]);
        }
        beams.resize(1);
    }

    // Min-heaps are used to efficiently collect the top-k elements (k=n_beams).
    // The repetitive patterns below reflect the 2 stages of heaps:
    //  * Gather elements until the vector is full, then call std::make_heap() on it.
    //  * If the heap is full and a new element is found that should be included, pop the
    //    least element to the back(), replace it with the new, then push it into the heap.
    void fill_next_beams_by_top_probabilities(llama_beam & beam) {
        // Min-heaps use a greater-than comparator.
        const auto comp = [](const llama_beam & a, const llama_beam & b) { return a.p > b.p; };
        if (beam.eob) {
            // beam is at end-of-sentence, so just copy it to next_beams if its probability is high enough.
            if (next_beams.size() < n_beams) {
                next_beams.push_back(std::move(beam));
                if (next_beams.size() == n_beams) {
                    std::make_heap(next_beams.begin(), next_beams.end(), comp);
                }
            } else if (next_beams.front().p < beam.p) {
                std::pop_heap(next_beams.begin(), next_beams.end(), comp);
                next_beams.back() = std::move(beam);
                std::push_heap(next_beams.begin(), next_beams.end(), comp);
            }
        } else {
            // beam is not at end-of-sentence, so branch with next top_k tokens.
            if (!beam.tokens.empty()) {
                //llama_decode(ctx, llama_batch_get_one(beam.tokens.data(), beam.tokens.size(), n_past, 0));
            }
            llama_logit_info logit_info(ctx);
            std::vector<llama_token_data> next_tokens = logit_info.top_k(n_beams);
            size_t i=0;
            if (next_beams.size() < n_beams) {
                for (; next_beams.size() < n_beams ; ++i) {
                    llama_beam next_beam = beam;
                    next_beam.tokens.push_back(next_tokens[i].id);
                    next_beam.p *= logit_info.probability_from_logit(next_tokens[i].logit);
                    next_beams.push_back(std::move(next_beam));
                }
                std::make_heap(next_beams.begin(), next_beams.end(), comp);
            } else {
                for (; next_beams.front().p == 0.0f ; ++i) {
                    std::pop_heap(next_beams.begin(), next_beams.end(), comp);
                    next_beams.back() = beam;
                    next_beams.back().tokens.push_back(next_tokens[i].id);
                    next_beams.back().p *= logit_info.probability_from_logit(next_tokens[i].logit);
                    std::push_heap(next_beams.begin(), next_beams.end(), comp);
                }
            }
            for (; i < n_beams ; ++i) {
                const float next_p = beam.p * logit_info.probability_from_logit(next_tokens[i].logit);
                if (next_beams.front().p < next_p) {
                    std::pop_heap(next_beams.begin(), next_beams.end(), comp);
                    next_beams.back() = beam;
                    next_beams.back().tokens.push_back(next_tokens[i].id);
                    next_beams.back().p = next_p;
                    std::push_heap(next_beams.begin(), next_beams.end(), comp);
                }
            }
        }
    }

    // Find common_prefix_length based on beams.
    // Requires beams is not empty.
    size_t find_common_prefix_length() {
        size_t common_prefix_length = beams[0].tokens.size();
        for (size_t i = 1 ; i < beams.size() ; ++i) {
            common_prefix_length = std::min(common_prefix_length, beams[i].tokens.size());
            for (size_t j = 0 ; j < common_prefix_length ; ++j) {
                if (beams[0].tokens[j] != beams[i].tokens[j]) {
                    common_prefix_length = j;
                    break;
                }
            }
        }
        return common_prefix_length;
    }

    // Construct beams_state to send back to caller via the callback function.
    // Side effect: set common_prefix_length = find_common_prefix_length();
    llama_beams_state get_beams_state(const bool last_call) {
        for (size_t i = 0 ; i < beams.size() ; ++i) {
            beam_views[i] = beams[i].view();
        }
        common_prefix_length = find_common_prefix_length();
        return {beam_views.data(), beams.size(), common_prefix_length, last_call};
    }

    // Loop:
    //  * while i < n_predict, AND
    //  * any of the beams have not yet reached end-of-beam (eob), AND
    //  * the highest probability beam(s) (plural in case of ties) are not at end-of-sentence
    //    (since all other beam probabilities can only decrease)
    void loop(const llama_beam_search_callback_fn_t callback, void * const callback_data) {
        beams.push_back({{}, 1.0f, false});  // Start with one empty beam w/ probability = 1.0 and !eob.
        const auto not_eob = [](const llama_beam & beam) { return !beam.eob; };
        for (int i = 0 ; i < n_predict && std::any_of(beams.begin(),beams.end(),not_eob) &&
                       !beams[top_beam_index()].eob ; ++i) {
            callback(callback_data, get_beams_state(false));  // Sets common_prefix_length
            update_beams_from_beam_views();   // Update values (p,eob) that callback may have changed.
            if (common_prefix_length) {
                //llama_decode(ctx, llama_batch_get_one(beams[0].tokens.data(), common_prefix_length, n_past, 0));
                n_past += common_prefix_length;
            }
            // Zero-out next_beam probabilities to place them last in following min-heap.
            std::for_each(next_beams.begin(), next_beams.end(), [](llama_beam & beam) { beam.p = 0.0f; });
            for (llama_beam & beam : beams) {
                beam.shift_tokens(common_prefix_length);
                fill_next_beams_by_top_probabilities(beam);
            }
            // next_beams become the beams of next/final iteration. Swap them to re-use memory.
            beams.swap(next_beams);
            renormalize_beam_probabilities(beams);
        }
        collapse_beams(top_beam_index());
        callback(callback_data, get_beams_state(true));
    }

    // As beams grow, the cumulative probabilities decrease.
    // Renormalize them to avoid floating point underflow.
    static void renormalize_beam_probabilities(std::vector<llama_beam> & beams) {
        const auto sum_p = [](float sum, llama_beam & beam) { return sum + beam.p; };
        const float inv_sum = 1.0f / std::accumulate(beams.begin(), beams.end(), 0.0f, sum_p);
        std::for_each(beams.begin(), beams.end(), [=](llama_beam & beam) { beam.p *= inv_sum; });
    }

    // Assumes beams is non-empty.  Uses llama_beam::operator<() for ordering.
    size_t top_beam_index() {
        return std::max_element(beams.begin(), beams.end()) - beams.begin();
    }

    // Copy (p,eob) for each beam which may have been changed by the callback.
    void update_beams_from_beam_views() {
        for (size_t i = 0 ; i < beams.size() ; ++i) {
            beams[i].p = beam_views[i].p;
            beams[i].eob = beam_views[i].eob;
        }
    }
};

void llama_beam_search(llama_context * ctx,
                       llama_beam_search_callback_fn_t callback, void * callback_data,
                       size_t n_beams, int n_past, int n_predict) {
    assert(ctx);
    const int64_t t_start_sample_us = ggml_time_us();

    llama_beam_search_data beam_search_data(ctx, n_beams, n_past, n_predict);

    beam_search_data.loop(callback, callback_data);

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
}

//
// quantization
//

struct quantize_state_internal {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv    = 0;
    int n_ffn_down        = 0;
    int n_ffn_gate        = 0;
    int n_ffn_up          = 0;
    int i_attention_wv    = 0;
    int i_ffn_down        = 0;
    int i_ffn_gate        = 0;
    int i_ffn_up          = 0;

    int n_k_quantized     = 0;
    int n_fallback        = 0;

    bool has_imatrix      = false;

    quantize_state_internal(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
        {}
};


static void llama_tensor_dequantize_internal2(
    struct ggml_tensor * tensor, float *f32_output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    ggml_type_traits_t qtype;
    if (ggml_is_quantized(tensor->type)) {
        qtype = ggml_internal_get_type_traits(tensor->type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype.to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ASSERT(false); // unreachable
        }
        return;
    }

    size_t block_size = tensor->type == GGML_TYPE_F16 ? 1 : (size_t)ggml_blck_size(tensor->type);
    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else {
                qtype.to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

void llama_tensor_dequantize( struct ggml_tensor *tensor, float *output, std::vector<std::thread> & workers,
                              const size_t nelements, const int nthread )
{
    llama_tensor_dequantize_internal2( tensor, output, workers, nelements, nthread );
}

static void llama_tensor_dequantize_internal(
    struct ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    llama_tensor_dequantize_internal2(tensor, f32_output, workers, nelements, nthread);
}


static ggml_type get_k_quant_type(quantize_state_internal & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;
    const auto       tn = LLM_TN(arch);

    auto use_more_bits = [](int i_layer, int num_layers) -> bool {
        return i_layer < num_layers/8 || i_layer >= 7*num_layers/8 || (i_layer - num_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but iccasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            n_layer /= n_expert;
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (name == tn(LLM_TENSOR_OUTPUT, "weight") ||
        (LLM_TENSOR_NAMES.at(arch).find(LLM_TENSOR_OUTPUT) == LLM_TENSOR_NAMES.at(arch).end() && name == "token_embd.weight")) {
        int nx = tensor->ne[0];
        if (arch == LLM_ARCH_FALCON || nx % QK_K != 0) {
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                 ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (new_type != GGML_TYPE_Q8_0) {
            new_type = GGML_TYPE_Q6_K;
        }
    } else if (name == "token_embd.weight") {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
            ftype == LLAMA_FTYPE_MOSTLY_IQ1_S) {
            new_type = GGML_TYPE_Q2_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
            new_type = GGML_TYPE_IQ3_S;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ3_S;
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
        if (name.find("attn_v.weight") != std::string::npos) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && name.find("attn_k.weight") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (name.find("ffn_down") != std::string::npos) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (name.find("attn_output.weight") != std::string::npos) {
            if (qs.model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (name.find("attn_v.weight") != std::string::npos) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        else if (QK_K == 64 && (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S) &&
                (qs.i_attention_wv < qs.n_attention_wv/8 || qs.i_attention_wv >= 7*qs.n_attention_wv/8)) new_type = GGML_TYPE_Q6_K;
        if (qs.model.type == MODEL_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
    } else if (name.find("attn_k.weight") != std::string::npos) {
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (name.find("attn_q.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (name.find("ffn_down") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (name.find("attn_output.weight") != std::string::npos) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (name.find("attn_qkv.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (name.find("ffn_gate") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_gate;
    }
    else if (name.find("ffn_up") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_up;
    }

    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
    //}
    // IK: let's remove this, else Q2_K is almost the same as Q3_K_S
    //else if (name.find("ffn_gate") != std::string::npos || name.find("ffn_up") != std::string::npos) {
    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
    //}
    // This can be used to reduce the size of the Q5_K_S model.
    // The associated PPL increase is fully in line with the size reduction
    //else {
    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_S) new_type = GGML_TYPE_Q4_K;
    //}
    bool convert_incompatible_tensor = false;
    if (new_type == GGML_TYPE_Q2_K || new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K ||
        new_type == GGML_TYPE_Q5_K || new_type == GGML_TYPE_Q6_K || new_type == GGML_TYPE_IQ4_XS ||
        new_type == GGML_TYPE_IQ2_XS || new_type == GGML_TYPE_IQ2_XXS || new_type == GGML_TYPE_IQ2_S ||
        new_type == GGML_TYPE_IQ3_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || new_type == GGML_TYPE_IQ3_S) {
        int nx = tensor->ne[0];
        int ny = tensor->ne[1];
        if (nx % QK_K != 0) {
            LLAMA_LOG_WARN("\n\n%s : tensor cols %d x %d are not divisible by %d, required for %s", __func__, nx, ny, QK_K, ggml_type_name(new_type));
            convert_incompatible_tensor = true;
        } else {
            ++qs.n_k_quantized;
        }
    }
    if (convert_incompatible_tensor) {
        switch (new_type) {
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_IQ4_XS: new_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q4_K:   new_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:   new_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:   new_type = GGML_TYPE_Q8_0;   break;
            default: throw std::runtime_error("\nUnsupported tensor size encountered\n");
        }
        LLAMA_LOG_WARN(" - using fallback quantization %s\n", ggml_type_name(new_type));
        ++qs.n_fallback;
    }

    return new_type;
}

static int32_t llama_tensor_quantize_internal(enum ggml_type new_type, const float * f32_data, void * new_data, const int chunk_size, int nrows, int n_per_row, int64_t * hist_cur, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    std::mutex mutex;
    int counter = 0;
    size_t new_size = 0;
    if (nthread < 2) {
        // single-thread
        return ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, hist_cur, imatrix);
    }
    auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, chunk_size,
            nrows, n_per_row, imatrix]() {
        std::array<int64_t, 1 << 4> local_hist = {};
        const int nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    for (int j=0; j<int(local_hist.size()); ++j) {
                        hist_cur[j] += local_hist[j];
                    }
                    new_size += local_size;
                }
                break;
            }
            lock.unlock();
            const int this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            local_size += ggml_quantize_chunk(new_type, f32_data, new_data,
                    first_row * n_per_row, this_nrow, n_per_row, local_hist.data(), imatrix);
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    return new_size;
}


static void llama_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type quantized_type;
    llama_ftype ftype = params->ftype;

    switch (params->ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: quantized_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: quantized_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: quantized_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: quantized_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: quantized_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_F16:  quantized_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_ALL_F32:     quantized_type = GGML_TYPE_F32;  break;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    quantized_type = GGML_TYPE_Q2_K;    break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  quantized_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  quantized_type = GGML_TYPE_Q3_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  quantized_type = GGML_TYPE_Q4_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  quantized_type = GGML_TYPE_Q5_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    quantized_type = GGML_TYPE_Q6_K;    break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: quantized_type = GGML_TYPE_IQ2_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  quantized_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   quantized_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   quantized_type = GGML_TYPE_IQ2_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: quantized_type = GGML_TYPE_IQ3_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   quantized_type = GGML_TYPE_IQ1_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  quantized_type = GGML_TYPE_IQ4_NL;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  quantized_type = GGML_TYPE_IQ4_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:   quantized_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   quantized_type = GGML_TYPE_IQ3_S;   break;

        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    // mmap consistently increases speed Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_loader ml(fname_inp, use_mmap, NULL);
    ml.init_mapping(false); // no prefetching?

    llama_model model;
    llm_load_arch(ml, model);
    llm_load_hparams(ml, model);

    struct quantize_state_internal qs(model, params);

    if (params->only_copy) {
        ftype = model.ftype;
    }
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            LLAMA_LOG_INFO("================================ Have weights data with %d entries\n",int(imatrix_data->size()));
            qs.has_imatrix = true;
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    struct gguf_context * ctx_out = gguf_init_empty();

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out, ml.ctx_gguf);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", ftype);

    for (int i = 0; i < ml.n_tensors; ++i) {
        struct ggml_tensor * meta = ml.get_tensor_meta(i);

        const std::string name = ggml_get_name(meta);

        // TODO: avoid hardcoded tensor names - use the TN_* constants
        if (name.find("attn_v.weight") != std::string::npos || name.find("attn_qkv.weight") != std::string::npos) {
            ++qs.n_attention_wv;
        }
        else if (name.find("ffn_down") != std::string::npos) {
            ++qs.n_ffn_down;
        }
        else if (name.find("ffn_gate") != std::string::npos) {
            ++qs.n_ffn_gate;
        }
        else if (name.find("ffn_up") != std::string::npos) {
            ++qs.n_ffn_up;
        }
    }
    if (qs.n_attention_wv != qs.n_ffn_down || (uint32_t)qs.n_attention_wv != model.hparams.n_layer) {
        LLAMA_LOG_WARN("%s ============ Strange model: n_attention_wv = %d, n_ffn_down = %d, hparams.n_layer = %d\n",
                __func__, qs.n_attention_wv, qs.n_ffn_down, model.hparams.n_layer);
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<int64_t> hist_all(1 << 4, 0);

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int idx = 0;

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    // populate the original tensors so we get an initial meta data
    for (int i = 0; i < ml.n_tensors; ++i) {
        struct ggml_tensor * meta = ml.get_tensor_meta(i);
        gguf_add_tensor(ctx_out, meta);
    }

    std::ofstream fout(fname_out, std::ios::binary);
    fout.exceptions(std::ofstream::failbit); // fail fast on write errors

    const size_t meta_size = gguf_get_meta_size(ctx_out);

    LLAMA_LOG_INFO("%s: meta size = %zu bytes\n", __func__, meta_size);

    // placeholder for the meta data
    ::zeros(fout, meta_size);

    for (int i = 0; i < ml.n_tensors; ++i) {
        struct ggml_tensor * tensor = ml.get_tensor_meta(i);

        const std::string name = ggml_get_name(tensor);

        if (!ml.use_mmap) {
            if (read_data.size() < ggml_nbytes(tensor)) {
                read_data.resize(ggml_nbytes(tensor));
            }
            tensor->data = read_data.data();
        }
        ml.load_data_for(tensor);

        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

        // quantize only 2D tensors
        quantize &= (ggml_n_dims(tensor) == 2);
        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= !params->only_copy;

        // do not quantize expert gating tensors
        // NOTE: can't use LLM_TN here because the layer number is not known
        quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

        // do not quantize positional embeddings and token types (BERT)
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_POS_EMBD,    "weight");
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;

        if (quantize) {
            new_type = quantized_type;
            if (!params->pure) {
                new_type = get_k_quant_type(qs, new_type, tensor, ftype);
            }

            // If we've decided to quantize to the same type the tensor is already
            // in then there's nothing to do.
            quantize = tensor->type != new_type;
        }
        if (!quantize) {
            new_type = tensor->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor);
            LLAMA_LOG_INFO("size = %8.3f MB\n", ggml_nbytes(tensor)/1024.0/1024.0);
        } else {
            const size_t nelements = ggml_nelements(tensor);

            const float * imatrix = nullptr;
            if (imatrix_data) {
                auto it = imatrix_data->find(tensor->name);
                if (it == imatrix_data->end()) {
                    LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                } else {
                    if (it->second.size() == (size_t)tensor->ne[0]) {
                        imatrix = it->second.data();
                    } else {
                        LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                int(it->second.size()), int(tensor->ne[0]), tensor->name);
                    }
                }
            }
            if ((new_type == GGML_TYPE_IQ2_XXS ||
                 new_type == GGML_TYPE_IQ2_XS  ||
                 new_type == GGML_TYPE_IQ2_S   ||
                 new_type == GGML_TYPE_IQ1_S   ||
                (new_type == GGML_TYPE_Q2_K && params->ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && strcmp(tensor->name, "token_embd.weight") != 0)) && !imatrix) {
                LLAMA_LOG_ERROR("\n\n============================================================\n");
                LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                LLAMA_LOG_ERROR("============================================================\n\n");
                throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
            }

            float * f32_data;

            if (tensor->type == GGML_TYPE_F32) {
                f32_data = (float *) tensor->data;
            } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            } else {
                llama_tensor_dequantize_internal(tensor, f32_conv_buf, workers, nelements, nthread);
                f32_data = (float *) f32_conv_buf.data();
            }

            LLAMA_LOG_INFO("quantizing to %s .. ", ggml_type_name(new_type));
            fflush(stdout);

            if (work.size() < nelements * 4) {
                work.resize(nelements * 4); // upper bound on size
            }
            new_data = work.data();
            std::array<int64_t, 1 << 4> hist_cur = {};

            const int n_per_row = tensor->ne[0];
            const int nrows = nelements / n_per_row;

            static const int min_chunk_size = 32 * 512;
            const int chunk_size = n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row);

            const int nchunk = (nelements + chunk_size - 1)/chunk_size;
            const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
            new_size = llama_tensor_quantize_internal(new_type, f32_data, new_data, chunk_size, nrows, n_per_row, hist_cur.data(), imatrix, workers, nthread_use);

            LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB", ggml_nbytes(tensor)/1024.0/1024.0, new_size/1024.0/1024.0);
            int64_t tot_count = 0;
            for (size_t i = 0; i < hist_cur.size(); i++) {
                hist_all[i] += hist_cur[i];
                tot_count += hist_cur[i];
            }

            if (tot_count > 0) {
                LLAMA_LOG_INFO(" | hist: ");
                for (size_t i = 0; i < hist_cur.size(); i++) {
                    LLAMA_LOG_INFO("%5.3f ", hist_cur[i] / float(nelements));
                }
            }
            LLAMA_LOG_INFO("\n");
        }
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;

        // update the gguf meta data as we go
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data, new_size);

        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }

    // go back to beginning of file and write the updated meta data
    {
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *) data.data(), data.size());
    }

    fout.close();

    gguf_free(ctx_out);

    LLAMA_LOG_INFO("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

    // print histogram for all tensors
    {
        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); i++) {
            sum_all += hist_all[i];
        }

        if (sum_all > 0) {
            LLAMA_LOG_INFO("%s: hist: ", __func__);
            for (size_t i = 0; i < hist_all.size(); i++) {
                LLAMA_LOG_INFO("%5.3f ", hist_all[i] / float(sum_all));
            }
            LLAMA_LOG_INFO("\n");
        }
    }

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) incompatible with k-quants and required fallback quantization\n",
                __func__, qs.n_fallback, qs.n_k_quantized + qs.n_fallback);
    }
}

static int llama_apply_lora_from_file_internal(
    const struct llama_model & model, const char * path_lora, float scale, const char * path_base_model, int n_threads
) {
    LLAMA_LOG_INFO("%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

    const int64_t t_start_lora_us = ggml_time_us();

    llama_file fin(path_lora, "rb");

    // verify magic and version
    {
        uint32_t magic = fin.read_u32();
        if (magic != LLAMA_FILE_MAGIC_GGLA) {
            LLAMA_LOG_ERROR("%s: bad file magic\n", __func__);
            return 1;
        }

        uint32_t format_version = fin.read_u32();
        if (format_version != 1) {
            LLAMA_LOG_ERROR("%s: unsupported file version\n", __func__ );
            return 1;
        }
    }

    int32_t lora_r = fin.read_u32();
    int32_t lora_alpha = fin.read_u32();
    float scaling = scale * (float)lora_alpha / (float)lora_r;

    LLAMA_LOG_INFO("%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);

    // load base model
    std::unique_ptr<llama_model_loader> ml;
    if (path_base_model) {
        LLAMA_LOG_INFO("%s: loading base model from '%s'\n", __func__, path_base_model);
        ml.reset(new llama_model_loader(path_base_model, /*use_mmap*/ true, /*kv_overrides*/ nullptr));
        ml->init_mapping(/*prefetch*/ false); // no prefetching
    }

    struct tensor_meta {
        std::string name;
        ggml_type type;
        int32_t ne[2];
        size_t offset;
    };
    std::map<std::string, tensor_meta> tensor_meta_map;

    // load all tensor meta
    while (true) {
        if (fin.tell() == fin.size) {
            // eof
            break;
        }

        int32_t n_dims;
        int32_t name_len;
        int32_t ftype;

        fin.read_raw(&n_dims, sizeof(n_dims));
        fin.read_raw(&name_len, sizeof(name_len));
        fin.read_raw(&ftype, sizeof(ftype));

        if (n_dims != 1 && n_dims != 2) {
            LLAMA_LOG_ERROR("%s: unsupported tensor dimension %d\n", __func__, n_dims);
            return 1;
        }

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read_raw(&ne[i], sizeof(ne[i]));
        }

        std::string name;
        {
            GGML_ASSERT(name_len < GGML_MAX_NAME);
            char buf[GGML_MAX_NAME];
            fin.read_raw(buf, name_len);
            name = std::string(buf, name_len);
        }

        // check for lora suffix
        std::string lora_suffix;
        if (name.length() > 6) {
            lora_suffix = name.substr(name.length() - 6);
        }
        if (lora_suffix != ".loraA" && lora_suffix != ".loraB") {
            LLAMA_LOG_ERROR("%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
            return 1;
        }

        // tensor type
        ggml_type wtype;
        switch (ftype) {
            case 0: wtype = GGML_TYPE_F32;  break;
            case 1: wtype = GGML_TYPE_F16;  break;
            default:
                    {
                        LLAMA_LOG_ERROR("%s: invalid tensor data type '%d'\n",
                                __func__, ftype);
                        return 1;
                    }
        }

        // data offset
        size_t offset = fin.tell();
        offset = (offset + 31) & -32;

        // skip tensor data
        fin.seek(offset + ggml_row_size(wtype, ne[0]) * ne[1], SEEK_SET);

        tensor_meta_map.emplace(name, tensor_meta{ name, wtype, { ne[0], ne[1] }, offset });
    }

    bool warned = false;
    int n_tensors = 0;

    // apply
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    if (backend_cpu == nullptr) {
        LLAMA_LOG_ERROR("%s: error: failed to initialize cpu backend\n", __func__);
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);

    std::vector<no_init<uint8_t>> read_buf;
    for (const auto & it : model.tensors_by_name) {
        const std::string & base_name = it.first;
        ggml_tensor * model_t = it.second;

        if (tensor_meta_map.find(base_name + ".loraA") == tensor_meta_map.end() ||
            tensor_meta_map.find(base_name + ".loraB") == tensor_meta_map.end()) {
            continue;
        }

        tensor_meta & metaA = tensor_meta_map.at(base_name + ".loraA");
        tensor_meta & metaB = tensor_meta_map.at(base_name + ".loraB");

        ggml_init_params lora_init_params = {
            /* .mem_size   */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_buffer */ nullptr,
            /* .no_alloc   */ true,
        };
        ggml_context * lora_ctx = ggml_init(lora_init_params);
        if (lora_ctx == nullptr) {
            LLAMA_LOG_ERROR("%s: error: failed to initialize lora context\n", __func__);
            ggml_backend_free(backend_cpu);
            return 1;
        }

        // create tensors
        ggml_tensor * loraA = ggml_new_tensor_2d(lora_ctx, metaA.type, metaA.ne[0], metaA.ne[1]);
        ggml_tensor * loraB = ggml_new_tensor_2d(lora_ctx, metaB.type, metaB.ne[0], metaB.ne[1]);
        ggml_set_name(loraA, metaA.name.c_str());
        ggml_set_name(loraB, metaB.name.c_str());

        ggml_tensor * base_t;
        if (ml) {
            if (gguf_find_tensor(ml->ctx_gguf, base_name.c_str()) < 0) {
                LLAMA_LOG_ERROR("%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
                return 1;
            }
            base_t = ggml_dup_tensor(lora_ctx, ml->get_tensor_meta(base_name.c_str()));
        } else {
            base_t = ggml_dup_tensor(lora_ctx, model_t);
        }
        ggml_set_name(base_t, base_name.c_str());

        // allocate in backend buffer
        ggml_backend_buffer_t lora_buf = ggml_backend_alloc_ctx_tensors_from_buft(lora_ctx, ggml_backend_cpu_buffer_type());
        if (lora_buf == nullptr) {
            LLAMA_LOG_ERROR("%s: error: failed to allocate lora tensors\n", __func__);
            return 1;
        }

        // load tensor data
        auto load_tensor = [&read_buf, &fin](const tensor_meta & tensor_meta, ggml_tensor * tensor) {
            read_buf.resize(ggml_nbytes(tensor));
            fin.seek(tensor_meta.offset, SEEK_SET);
            fin.read_raw(read_buf.data(), ggml_nbytes(tensor));
            ggml_backend_tensor_set(tensor, read_buf.data(), 0, read_buf.size());
        };
        load_tensor(metaA, loraA);
        load_tensor(metaB, loraB);

        // load base model tensor data
        if (ml) {
            ml->load_data_for(base_t);
        } else {
            ggml_backend_tensor_copy(model_t, base_t);
        }

        if (ggml_is_quantized(base_t->type) && !warned) {
            LLAMA_LOG_WARN("%s: warning: using a lora adapter with a quantized model may result in poor quality, "
                            "use a f16 or f32 base model with --lora-base\n", __func__);
            warned = true;
        }

        if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
            LLAMA_LOG_ERROR("%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64 ");"
                            " are you sure that this adapter is for this model?\n", __func__, base_t->ne[0], loraA->ne[1]);
            ggml_free(lora_ctx);
            ggml_backend_buffer_free(lora_buf);
            ggml_backend_free(backend_cpu);
            return 1;
        }

        auto build_lora_graph = [&]() {
            // w = w + BA*s
            ggml_tensor * BA = ggml_mul_mat(lora_ctx, loraA, loraB);
            ggml_set_name(BA, "BA");

            if (scaling != 1.0f) {
                BA = ggml_scale(lora_ctx, BA, scaling);
                ggml_set_name(BA, "BA_scaled");
            }

            ggml_tensor * r;
            r = ggml_add_inplace(lora_ctx, base_t, BA);
            ggml_set_name(r, "r_add");

            if (base_t->type != model_t->type) {
                // convert the result to the model type
                r = ggml_cast(lora_ctx, r, model_t->type);
                ggml_set_name(r, "r_cast");
            }

            return r;
        };

        ggml_cgraph * gf = ggml_new_graph(lora_ctx);
        ggml_tensor * r = build_lora_graph();
        ggml_build_forward_expand(gf, r);

        ggml_backend_buffer_t graph_buf = ggml_backend_alloc_ctx_tensors_from_buft(lora_ctx, ggml_backend_cpu_buffer_type());
        if (graph_buf == nullptr) {
            LLAMA_LOG_ERROR("%s: error: failed to allocate graph tensors\n", __func__);
            ggml_free(lora_ctx);
            ggml_backend_buffer_free(lora_buf);
            ggml_backend_free(backend_cpu);
            return 1;
        }

        ggml_backend_graph_compute(backend_cpu, gf);

        ggml_backend_tensor_set(model_t, r->data, 0, ggml_nbytes(r));

#if 0
        // TODO: use scheduler with fallback to CPU for less copies between CPU and GPU
        //ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), backends.size(), GGML_DEFAULT_GRAPH_SIZE);

        // sched compute
        ggml_build_forward_expand(gf, build_graph());
        ggml_backend_sched_init_measure(sched, gf);

        // create the graph again, since the previous one was destroyed by the measure
        ggml_graph_clear(gf);
        ggml_build_forward_expand(gf, build_graph());
        ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_sched_free(sched);
#endif

        ggml_backend_buffer_free(lora_buf);
        ggml_backend_buffer_free(graph_buf);
        ggml_free(lora_ctx);

        n_tensors++;
        if (n_tensors % 4 == 0) {
            LLAMA_LOG_INFO(".");
        }
    }

    ggml_backend_free(backend_cpu);

    const int64_t t_lora_us = ggml_time_us() - t_start_lora_us;
    LLAMA_LOG_INFO(" done (%.2f ms)\n", t_lora_us / 1000.0);

    return 0;
}

//
// interface implementation
//
struct llama_model_params llama_model_default_params() {
    struct llama_model_params result = {
        /*.n_gpu_layers                =*/ 0,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
    };

#ifdef GGML_USE_METAL
    // note: we usually have plenty of VRAM, so by default offload all layers to the GPU
    result.n_gpu_layers = 999;
#endif

    return result;
}

struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        /*.seed                        =*/ LLAMA_DEFAULT_SEED,
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 64,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ 1.0f,
        /*.yarn_beta_fast              =*/ 32.0f,
        /*.yarn_beta_slow              =*/ 1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.logits_all                  =*/ false,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
    };

    return result;
}

struct llama_model_quantize_params llama_model_quantize_default_params() {
    struct llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q5_1,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.imatrix                     =*/ nullptr,
    };

    return result;
}

size_t llama_max_devices(void) {
#if defined(GGML_USE_METAL)
    return 1;
#elif defined(GGML_USE_CUBLAS)
    return GGML_CUDA_MAX_DEVICES;
#elif defined(GGML_USE_SYCL)
    return GGML_SYCL_MAX_DEVICES;
#elif defined(GGML_USE_VULKAN)
    return GGML_VK_MAX_DEVICES;
#else
    return 1;
#endif
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST) || defined(GGML_USE_METAL) || defined(GGML_USE_VULKAN) || \
    defined(GGML_USE_SYCL)   || defined(GGML_USE_KOMPUTE)
    // Defined when llama.cpp is compiled with support for offloading model layers to GPU.
    return true;
#else
    return false;
#endif
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

#ifdef GGML_USE_MPI
    ggml_mpi_backend_init();
#endif
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        ggml_numa_init(numa);
    }
}

void llama_backend_free(void) {
#ifdef GGML_USE_MPI
    ggml_mpi_backend_free();
#endif
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

struct llama_model * llama_load_model_from_file_gpt4all(
        const char * path_model,
        struct llama_model_params * params_p) {
    auto & params = *params_p;

    ggml_time_init();

    llama_model * model = new llama_model;
    current_model = model;

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_INFO(".");
                if (percentage >= 100) {
                    LLAMA_LOG_INFO("\n");
                }
            }
            return true;
        };
    }

    int status = llama_model_load(path_model, *model, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }
        delete model;
        return nullptr;
    }

    return model;
}

struct llama_model * llama_load_model_from_file(const char * path_model, struct llama_model_params params) {
    return llama_load_model_from_file_gpt4all(path_model, &params);
}

void llama_free_model(struct llama_model * model) {
    delete model;
}

struct llama_context *llama_release_context(struct llama_model *model, uint8_t ctx_n)
{
    /*
    auto it = model->contexts.begin()+ctx_n;
    struct llama_context *lctx = *it;
    LLAMA_LOG_INFO("release_context(%p,%d)\n", lctx, ctx_n);
    model->contexts.erase(it);
    */
    return NULL; //lctx
}
void llama_release_context2(struct llama_model &model, struct llama_context *ctx)
{
    /*
    for( auto it = model.contexts.begin(); it != model.contexts.end(); ++it ) {
        if( *it == ctx ) {
            LLAMA_LOG_INFO("release_context(%p)\n", ctx);
            model.contexts.erase(it);
            return;
        }
    }
    */
}
/*
void llama_release_backend(struct llama_model &model, ggml_backend_t backend)
{
    ggml_backend_free(backend);
}
*/

void llama_pick_actor( std::string actorname )
{
    LLAMA_LOG_INFO("pickActor(%s)\n", actorname.c_str());
    current_kb->useactor( actorname );
    current_kb->active_actor = actorname; // decide who is responding with logits
}
int llama_process_tokens( std::string toname, std::string fromname, std::string input, std::vector<int> &tokens )
{
    return current_kb->process_tokens(toname, fromname, input, tokens);
}

struct llama_context *llama_select_context(struct llama_model *model, uint8_t ctx_n)
{
    return NULL;
}

uint8_t llama_view_context(struct llama_model *model, uint8_t ctx_n)
{
    return 0;
}

struct llama_context * llama_new_context_with_model(
                 struct llama_model * model,
        struct llama_context_params   params,
        llama_context *copyctx, bool partcopy, bool useparams ) {

    if (!model) {
        return nullptr;
    }
    if( copyctx && !partcopy ) {
        LLAMA_LOG_ERROR("%s: not implemented copyctx. copyctx is only used for partial (parameter) copies\n", __func__);
        return nullptr;
    }


    llama_context * ctx = new llama_context(*model);
    current_context = ctx;

    const auto & hparams = model->hparams;
    llama_cparams & cparams = ctx->cparams;
    llama_cparams *oparams;
    if( copyctx ) {
        LLAMA_LOG_INFO("%s: copyctx %s\n",     __func__, useparams ? "useparams" : "!useparams");
        oparams = &copyctx->cparams;
        if( !useparams ) {
            memcpy(&ctx->cparams, oparams, sizeof(llama_cparams));
            ctx->is_clone=false;
        } else {
            ctx->is_clone=true;
        }
        ctx->abort_callback      = copyctx->abort_callback;
        ctx->abort_callback_data = copyctx->abort_callback_data;
        ctx->rng                 = copyctx->rng; //std::mt19937(time(NULL));
        ctx->logits_all          = copyctx->logits_all;
    } else if( !useparams ) {
        LLAMA_LOG_INFO("%s: error does not compute\n", __func__);
        return NULL;
    } else {
        ctx->is_clone=false;
    }
    if( useparams ) {
        cparams.n_batch          = params.n_batch;
        cparams.n_threads        = params.n_threads;
        cparams.n_threads_batch  = params.n_threads_batch;
        cparams.yarn_ext_factor  = params.yarn_ext_factor;
        cparams.yarn_attn_factor = params.yarn_attn_factor;
        cparams.yarn_beta_fast   = params.yarn_beta_fast;
        cparams.yarn_beta_slow   = params.yarn_beta_slow;
        cparams.defrag_thold     = params.defrag_thold;
        cparams.embeddings       = params.embeddings;
        cparams.offload_kqv      = params.offload_kqv;
        cparams.pooling_type     = params.pooling_type;

        cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;

        cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
        cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

        cparams.n_yarn_orig_ctx  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                                   hparams.n_yarn_orig_ctx != 0 ? hparams.n_yarn_orig_ctx :
                                                                  hparams.n_ctx_train;

        cparams.cb_eval           = params.cb_eval;
        cparams.cb_eval_user_data = params.cb_eval_user_data;

        auto rope_scaling_type = params.rope_scaling_type;
        if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
            rope_scaling_type = hparams.rope_scaling_type_train;
        }

        if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
            cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
        }

        if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
            cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
        }

        if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
                cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
            } else {
                cparams.pooling_type = hparams.pooling_type;
            }
        }

        ctx->abort_callback      = params.abort_callback;
        ctx->abort_callback_data = params.abort_callback_data;

        if (params.seed == LLAMA_DEFAULT_SEED)
            params.seed = time(NULL);

        ctx->rng                 = std::mt19937(params.seed);
        ctx->logits_all          = params.logits_all;
    }

    LLAMA_LOG_INFO("%s: Batch: %d Threads: %d Threads_Batch: %d Ctx: %d\n", __func__, cparams.n_batch, cparams.n_threads,
                   cparams.n_threads_batch, cparams.n_ctx);

    ggml_type type_k, type_v;

    if( useparams ) {
        type_k = params.type_k;
        type_v = params.type_v;
    } else {
        type_k = GGML_TYPE_F16;
        type_v = GGML_TYPE_F16;
    }

    LLAMA_LOG_INFO("%s: n_ctx      = %u\n",     __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: freq_base  = %.1f\n",   __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale = %g\n",     __func__, cparams.rope_freq_scale);
    GGML_ASSERT(hparams.n_embd_head_k % ggml_blck_size(type_k) == 0);
    GGML_ASSERT(hparams.n_embd_head_v % ggml_blck_size(type_v) == 0);

    if (!hparams.vocab_only) {
        // initialize backends
#ifdef GGML_USE_METAL
        if (model->n_gpu_layers > 0) {
            ctx->backend_metal = ggml_backend_metal_init();
            if (ctx->backend_metal == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize Metal backend\n", __func__);
                llama_free(ctx);
                return nullptr;
            }
            ctx->backends.push_back(ctx->backend_metal);
        }
#elif defined(GGML_USE_CUBLAS)
        if (model->n_gpu_layers > 0) {
            // with split_mode LLAMA_SPLIT_MODE_NONE or LLAMA_SPLIT_MODE_ROW, only the main GPU backend is used
            if (model->split_mode == LLAMA_SPLIT_MODE_NONE || model->split_mode == LLAMA_SPLIT_MODE_ROW) {
                ggml_backend_t backend = ggml_backend_cuda_init(model->main_gpu);
                if (backend == nullptr) {
                    LLAMA_LOG_ERROR("%s: failed to initialize CUDA%d backend\n", __func__, model->main_gpu);
                    llama_free(ctx);
                    return nullptr;
                }
                ctx->backends.push_back(backend);
            } else {
                // LLAMA_SPLIT_MODE_LAYER requires a backend for each GPU
                for (int device = 0; device < ggml_backend_cuda_get_device_count(); ++device) {
                    ggml_backend_t backend = ggml_backend_cuda_init(device);
                    if (backend == nullptr) {
                        LLAMA_LOG_ERROR("%s: failed to initialize CUDA%d backend\n", __func__, device);
                        llama_free(ctx);
                        return nullptr;
                    }
                    ctx->backends.push_back(backend);
                }
            }
        }
#elif defined(GGML_USE_VULKAN)
        if (model->n_gpu_layers > 0) {
            for (int device = 0; device < ggml_backend_vk_get_device_count(); ++device) {
                ggml_backend_t backend = ggml_backend_vk_init(device);
                if (backend == nullptr) {
                    LLAMA_LOG_ERROR("%s: failed to initialize Vulkan%d backend\n", __func__, device);
                    llama_free(ctx);
                    return nullptr;
                }
                ctx->backends.push_back(backend);
            }
        }
#elif defined(GGML_USE_SYCL)
        if (model->n_gpu_layers > 0) {
            // with split_mode LLAMA_SPLIT_MODE_NONE or LLAMA_SPLIT_MODE_ROW, only the main GPU backend is used
            if (model->split_mode == LLAMA_SPLIT_MODE_NONE || model->split_mode == LLAMA_SPLIT_MODE_ROW) {
                int main_gpu_index = ggml_backend_sycl_get_device_index(model->main_gpu);
                ggml_backend_t backend = ggml_backend_sycl_init(main_gpu_index);
                if (backend == nullptr) {
                    LLAMA_LOG_ERROR("%s: failed to initialize SYCL%d (index %d)backend\n", __func__, model->main_gpu, main_gpu_index);
                    llama_free(ctx);
                    return nullptr;
                }
                ctx->backends.push_back(backend);
            } else {
                // LLAMA_SPLIT_LAYER requires a backend for each GPU
                int id_list[GGML_SYCL_MAX_DEVICES];
                ggml_sycl_get_gpu_list(id_list, GGML_SYCL_MAX_DEVICES);
                for (int i = 0; i < ggml_backend_sycl_get_device_count(); ++i) {
                    int device_id = id_list[i];
                    ggml_backend_t backend = ggml_backend_sycl_init(i);
                    if (backend == nullptr) {
                        LLAMA_LOG_ERROR("%s: failed to initialize SYCL%d (index %d)backend\n", __func__, device_id, i);
                        llama_free(ctx);
                        return nullptr;
                    }
                    ctx->backends.push_back(backend);
                }
            }
        }
#elif defined(GGML_USE_KOMPUTE)
        if( model->n_gpu_layers > 0 ) {
            if (!copyctx) {
                ggml_backend *backend = ggml_backend_kompute_init(model->main_gpu);

                if (backend == nullptr ) {
                    LLAMA_LOG_ERROR("%s: failed to initialize Kompute backend\n", __func__);
                    llama_free(ctx);
                    return nullptr;
                } else {
                    LLAMA_LOG_INFO("%s: prepared Kompute layers\n", __func__);
                }
                ctx->backends.push_back(backend);
            } else {
                ctx->backends.push_back( copyctx->backends[0] );
            }
        }
#endif
        if( !copyctx ) {
            ctx->backend_cpu = ggml_backend_cpu_init();
            if (ctx->backend_cpu == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize CPU backend\n", __func__);
                llama_free(ctx);
                return nullptr;
            }
        } else {
            ctx->backend_cpu = copyctx->backend_cpu;
        }
        ctx->backends.push_back(ctx->backend_cpu);

        /*
        if (!llama_kv_cache_init(ctx->kv_self, ctx->model, type_k, type_v, kv_context*cparams.n_ctx, cparams.offload_kqv)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }

        {
            size_t memory_size_k = 0;
            size_t memory_size_v = 0;

            for (auto & k : ctx->kv_self->k_l) {
                memory_size_k += ggml_nbytes(k);
            }

            for (auto & v : ctx->kv_self->v_l) {
                memory_size_v += ggml_nbytes(v);
            }

            LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
        }
        */

        // resized during inference, reserve maximum
        ctx->logits.reserve(hparams.n_vocab);

        if (params.embeddings)
            ctx->embd.reserve(hparams.n_embd);

        // graph inputs
        if( copyctx ) {
            ctx->ctx_input = copyctx->ctx_input;
            ctx->inp_tokens  = copyctx->inp_tokens ;
            ctx->inp_embd    = copyctx->inp_embd   ;
            ctx->inp_pos     = copyctx->inp_pos    ;
            ctx->inp_KQ_mask = copyctx->inp_KQ_mask;
            ctx->inp_KQ_pos  = copyctx->inp_KQ_pos ;
            ctx->inp_K_shift = copyctx->inp_K_shift;
            ctx->inp_mean    = copyctx->inp_mean   ;
            ctx->inp_cls     = copyctx->inp_cls    ;
            ctx->buf_input = copyctx->buf_input;
        } else {
            ggml_init_params init_params = {
                /* .mem_size   */ ggml_tensor_overhead()*9,
                /* .mem_buffer */ nullptr,
                /* .no_alloc   */ true,
            };
            ctx->ctx_input = ggml_init(init_params);
            LLAMA_LOG_INFO("%s: creating input context\n", __func__);

            //ctx->tok_embd = ggml_new_tensor_2d(ctx->ctx_input, GGML_TYPE_Q4_0, 4096, hparams.n_vocab);

            ctx->inp_tokens  = ggml_new_tensor_1d(ctx->ctx_input, GGML_TYPE_I32, cparams.n_batch);
            ctx->inp_embd    = ggml_new_tensor_2d(ctx->ctx_input, GGML_TYPE_F32, hparams.n_embd, cparams.n_batch);
            ctx->inp_pos     = ggml_new_tensor_1d(ctx->ctx_input, GGML_TYPE_I32, cparams.n_batch);
            ctx->inp_KQ_mask = ggml_new_tensor_2d(ctx->ctx_input, GGML_TYPE_F32, kv_context*cparams.n_ctx, cparams.n_batch);
            ctx->inp_KQ_pos  = ggml_new_tensor_1d(ctx->ctx_input, GGML_TYPE_F32, kv_context*cparams.n_ctx);
            ctx->inp_K_shift = ggml_new_tensor_1d(ctx->ctx_input, GGML_TYPE_I32, kv_context*cparams.n_ctx);
            ctx->inp_mean    = ggml_new_tensor_2d(ctx->ctx_input, GGML_TYPE_F32, cparams.n_batch, cparams.n_batch);
            ctx->inp_cls     = ggml_new_tensor_1d(ctx->ctx_input, GGML_TYPE_I32, cparams.n_batch);

            ggml_set_name(ctx->inp_tokens,  "inp_tokens");
            ggml_set_name(ctx->inp_embd,    "inp_embd");
            ggml_set_name(ctx->inp_pos,     "inp_pos");
            ggml_set_name(ctx->inp_KQ_mask, "inp_KQ_mask");
            ggml_set_name(ctx->inp_KQ_pos,  "inp_KQ_pos");
            ggml_set_name(ctx->inp_K_shift, "inp_K_shift");
            ggml_set_name(ctx->inp_mean,    "inp_mean");
            ggml_set_name(ctx->inp_cls,     "inp_cls");

            LLAMA_LOG_INFO("%s: alloc input context\n", __func__);

            ctx->buf_input = ggml_backend_alloc_ctx_tensors_from_buft(ctx->ctx_input, llama_default_buffer_type_cpu(true));

            /*
            LLAMA_LOG_INFO("%s: writing ctx->tok_embd\n", __func__);
            int rowsize = 4096;
            int tdrowsize = rowsize/32;
            float fbuf[4096];
            memset( fbuf, 0, sizeof(float)*rowsize );
            block_q4_0 *td = (block_q4_0*)ctx->tok_embd->data;
            memcpy( ctx->tok_embd->data, model->tok_embd->data, 18*tdrowsize*hparams.n_vocab );

            LLAMA_LOG_INFO("%s: writing ctx->tok_embd[20000]\n", __func__);
            for( int i=0; i<20000; i++ ) {
                quantize_row_q4_0( fbuf, td + ((hparams.n_vocab+i)*tdrowsize), rowsize );
            }
            */

        }
        LLAMA_LOG_INFO("%s: %10s input buffer size   = %8.2f MiB\n", __func__,
                ggml_backend_buffer_name(ctx->buf_input),
                ggml_backend_buffer_get_size(ctx->buf_input) / 1024.0 / 1024.0);

        // buffer types used for the compute buffer of each backend
        for (ggml_backend *backend : ctx->backends) {
            if (ggml_backend_is_cpu(backend)) {
                // use host buffers for the CPU backend compute buffer
                backend_buft.push_back(llama_default_buffer_type_cpu(true));
            } else {
                backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
            }
        }

        // scheduler and compute buffers
        // buffer used to store the computation graph and the tensor meta data
        ctx->buf_compute_meta.resize(ggml_tensor_overhead()*LLAMA_MAX_NODES+ggml_graph_overhead_custom(LLAMA_MAX_NODES, false));
        LLAMA_LOG_INFO("%s: resize buf_compute_meta to %d\n", __func__, ctx->buf_compute_meta.size());
        if( !copyctx ) {
            ctx->sched = ggml_backend_sched_new(ctx->backends.data(), backend_buft.data(), ctx->backends.size(), LLAMA_MAX_NODES);
            LLAMA_LOG_INFO("%s: created new scheduler %p for context %p\n", __func__, ctx->sched, ctx);


        } else {
            ctx->sched = copyctx->sched;
        }
    }

#ifdef GGML_USE_MPI
    ctx->ctx_mpi = ggml_mpi_init();

    if (ggml_mpi_rank(ctx->ctx_mpi) > 0) {
        // Enter a blocking eval loop with dummy input, letting rank=0 drive the process
        // TODO: needs fix after #3228
        GGML_ASSERT(false && "not implemented");
        //const std::vector<llama_token> tmp(ctx->model.hparams.n_ctx, llama_token_bos(ctx));
        //while (!llama_eval(ctx, tmp.data(), tmp.size(), 0, 0)) {};
        llama_backend_free();
        exit(1);
    }
#endif

    /*
    if( model->contexts.capacity() == 0 ) { // generate base context
        LLAMA_LOG_INFO("%s: build context zero as well\n", __func__);
        model->contexts.reserve(2);
        int save_ctx = params.n_ctx;
        params.n_ctx = 512;
        model->zero = llama_new_context_with_model(model, params, ctx, true, true);
        params.n_ctx = save_ctx;
    }
    model->contexts.push_back(ctx);
    */
    ctx->ctx_ready = true;
    current_kb = (System_kb*)pool_alloc(sizeof(System_kb));
    new (current_kb)  System_kb;
    current_kb->prepare();
    LLAMA_LOG_INFO("Initializing KB.\n");
    memcpy( &current_kb->hparams, &hparams, sizeof(llama_hparams) );
    current_kb->useactor("System");
    LLAMA_LOG_INFO("current_kb initialized\n");
    return ctx;
}

void llama_free(struct llama_context * ctx) {

    delete ctx;
}

const llama_model * llama_get_model(const struct llama_context * ctx) {
    return &ctx->model;
}

uint32_t llama_n_ctx(const struct llama_context * ctx) {
    return ctx->cparams.n_ctx;
}

uint32_t llama_n_batch(const struct llama_context * ctx) {
    return ctx->cparams.n_batch;
}

enum llama_vocab_type llama_vocab_type(const struct llama_model * model) {
    return model->vocab.type;
}

enum llama_rope_type llama_rope_type(const struct llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_PERSIMMON:
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_STARCODER2:
            return LLAMA_ROPE_TYPE_NEOX;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ASSERT(false && "unknown architecture");
            break;
    }

    return LLAMA_ROPE_TYPE_NONE;
}

int32_t llama_n_vocab(const struct llama_model * model) {
    return model->vocab.id_to_token.size();
}

int32_t llama_n_ctx_train(const struct llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_n_embd(const struct llama_model * model) {
    return model->hparams.n_embd;
}

float llama_rope_freq_scale_train(const struct llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const struct llama_model * model) {
    return (int)model->gguf_kv.size();
}

int32_t llama_model_meta_key_by_index(const struct llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s %s %s",
            llama_model_arch_name(model->arch),
            llama_model_type_name(model->type),
            llama_model_ftype_name(model->ftype).c_str());
}

uint64_t llama_model_size(const struct llama_model * model) {
    uint64_t size = 0;
    for (const auto & it : model->tensors_by_name) {
        size += ggml_nbytes(it.second);
    }
    return size;
}

uint64_t llama_model_n_params(const struct llama_model * model) {
    uint64_t nparams = 0;
    for (const auto & it : model->tensors_by_name) {
        nparams += ggml_nelements(it.second);
    }
    return nparams;
}

struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name) {
    auto it = std::find_if(model->tensors_by_name.begin(), model->tensors_by_name.end(),
            [name](const std::pair<std::string, struct ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == model->tensors_by_name.end()) {
        return nullptr;
    }
    return it->second;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_internal(fname_inp, fname_out, params);
        return 0;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }
}

int32_t llama_model_apply_lora_from_file(const struct llama_model * model, const char * path_lora, float scale, const char * path_base_model, int32_t n_threads) {
    try {
        return llama_apply_lora_from_file_internal(*model, path_lora, scale, path_base_model, n_threads);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

/*
void llama_kv_cache_view_update(const struct llama_context * ctx, struct llama_kv_cache_view * view) {
    if (uint32_t(view->n_cells) < ctx->kv_self->size || view->cells == nullptr) {
        view->n_cells = int32_t(ctx->kv_self->size);
        void * p = realloc(view->cells, sizeof(struct llama_kv_cache_view_cell) * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells");
        view->cells = (struct llama_kv_cache_view_cell *)p;
        p = realloc(view->cells_sequences, sizeof(llama_seq_id) * view->n_max_seq * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells sequences");
        view->cells_sequences = (llama_seq_id *)p;
    }

    const std::vector<llama_kv_cell> & kv_cells = ctx->kv_self->cells;
    llama_kv_cache_view_cell * c_curr = view->cells;
    llama_seq_id * cs_curr = view->cells_sequences;
    int32_t used_cells = 0;
    int32_t token_count = 0;
    int32_t curr_contig_idx = -1;
    uint32_t max_contig = 0;
    int32_t max_contig_idx = -1;

    for (int32_t i = 0; i < int32_t(ctx->kv_self->size); i++, c_curr++, cs_curr += view->n_max_seq) {
        const size_t curr_size = kv_cells[i].seq_id.size();
        token_count += curr_size;
        c_curr->pos = kv_cells[i].pos + kv_cells[i].delta;

        if (curr_size > 0) {
            if (curr_contig_idx >= 0 && uint32_t(i - curr_contig_idx) > max_contig) {
                max_contig = i - curr_contig_idx;
                max_contig_idx = curr_contig_idx;
            }
            curr_contig_idx = -1;
        } else if (curr_contig_idx < 0) {
            curr_contig_idx = i;
        }

        int seq_idx = 0;
        for (const llama_seq_id it : kv_cells[i].seq_id) {
            if (seq_idx >= view->n_max_seq) {
                break;
            }
            cs_curr[seq_idx] = it;
            seq_idx++;
        }
        if (seq_idx != 0) {
            used_cells++;
        }
        for (; seq_idx < view->n_max_seq; seq_idx++) {
            cs_curr[seq_idx] = -1;
        }
    }
    if (curr_contig_idx >= 0 && kv_cells.size() - curr_contig_idx > max_contig) {
        max_contig_idx = curr_contig_idx;
        max_contig = kv_cells.size() - curr_contig_idx;
    }
    view->max_contiguous = max_contig;
    view->max_contiguous_idx = max_contig_idx;
    view->token_count = token_count;
    view->used_cells = used_cells;
    if (uint32_t(used_cells) != ctx->kv_self->used) {
        LLAMA_LOG_ERROR("%s: used cells mismatch. kv_cache says %d but we calculated %d\n",
            __func__, ctx->kv_self->used, used_cells);
    }
}
*/


// Returns the *maximum* size of the state
size_t llama_get_state_size(const struct llama_context * ctx) {
    // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized state.
    // for reference, std::mt19937(1337) serializes to 6701 bytes.
    const size_t s_rng_size        = 2*sizeof(size_t);
    const size_t s_rng             = LLAMA_MAX_RNG_STATE;
    const size_t s_logits_size     = sizeof(size_t);
    // assume worst case for logits although only currently set ones are serialized
    const size_t s_logits          = ctx->logits.capacity() * sizeof(float);
    const size_t s_embedding_size  = sizeof(size_t);
    const size_t s_embedding       = ctx->embd.capacity() * sizeof(float);
    const size_t s_kv_buf_size     = sizeof(size_t);
    const size_t s_kv_used         = sizeof(uint32_t);
    const size_t s_kv              = ctx->kv_self->total_size();

    const size_t s_total = (
        + s_rng_size
        + s_rng
        + s_logits_size
        + s_logits
        + s_embedding_size
        + s_embedding
        + s_kv_buf_size
        + s_kv_used
        + s_kv
        + 512
    );

    return s_total;
}

// llama_context_data
#define DC_WRITE(x) write( &(x), sizeof(x) )
#define DC_READ(x) x = *( static_cast<decltype(x)*>((void*)inp) ); inp += sizeof(x)

struct llama_data_context {
    virtual void write(const void * src, size_t size) = 0;
    virtual size_t get_size_written() = 0;
    virtual ~llama_data_context() = default;
};

struct llama_data_buffer_context : llama_data_context {
    uint8_t * ptr;
    size_t size_written = 0;

    llama_data_buffer_context(uint8_t * p) : ptr(p) {}

    void write(const void * src, size_t size) override {
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_file_context : llama_data_context {
    llama_file * file;
    size_t size_written = 0;

    llama_data_file_context(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

/** copy state data into either a buffer or file depending on the passed in context
 *
 * file context:
 * llama_file file("/path", "wb");
 * llama_data_file_context data_ctx(&file);
 * llama_copy_state_data(ctx, &data_ctx);
 *
 * buffer context:
 * std::vector<uint8_t> buf(max_size, 0);
 * llama_data_buffer_context data_ctx(&buf.data());
 * llama_copy_state_data(ctx, &data_ctx);
 *
*/
static void llama_copy_state_data_internal(struct llama_context * ctx, llama_data_context * data_ctx) {
    // version info
    size_t version = 2123;
    char separator = (char)125;
    data_ctx->write(&version, sizeof(size_t));

    // copy rng
    {
        std::ostringstream rng_ss;
        rng_ss << ctx->rng;

        const std::string & rng_str = rng_ss.str();
        const size_t        rng_size = rng_str.size();

        GGML_ASSERT(rng_size <= LLAMA_MAX_RNG_STATE);

        data_ctx->write(&rng_size,      sizeof(rng_size));
        data_ctx->write(rng_str.data(), rng_size);
    }

    // copy logits
    {
        const size_t logits_size = ctx->logits.size();

        data_ctx->write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            data_ctx->write(ctx->logits.data(), logits_size * sizeof(float));
        }
    }

    // copy embeddings
    {
        const size_t embeddings_size = ctx->embd.size();

        data_ctx->write(&embeddings_size, sizeof(embeddings_size));

        if (embeddings_size) {
            data_ctx->write(ctx->embd.data(), embeddings_size * sizeof(float));
        }
    }

    const auto & kv_self = ctx->kv_self;

    // copy kv cache
    uint8_t context_count=3;
    data_ctx->write(&context_count, sizeof(context_count));
    for( int i=0; i<context_count; i++ )
    {
        const uint32_t kv_used     = ctx->seq_end;

        data_ctx->write(&separator, 1);
        data_ctx->write(&kv_used,     sizeof(kv_used));

        size_t k_row        = 2048*kv_used;
        size_t v_row_size   = 2*kv_used;
        size_t v_stride     = 2*kv_self->size;

        if (kv_used>0) {
            std::vector<uint8_t> tmp_buf;
            tmp_buf.resize(k_row);
            for (int il = 0; il < 32; ++il) {
                ggml_backend_tensor_get(kv_self->k_l[il], tmp_buf.data(), 0, k_row);
                data_ctx->write(tmp_buf.data(), k_row);

                for (int ir = 0; ir < (int) 1024; ++ir) {
                    ggml_backend_tensor_get(kv_self->v_l[il], tmp_buf.data()+ir*v_row_size, ir*v_stride, v_row_size);
                }
                data_ctx->write(tmp_buf.data(), k_row); // compiled data will be == k_row size
                data_ctx->write(&separator, 1); // separate each layer
            }
        }

        //data_ctx->write( kv_self->inuse, 512 );
        data_ctx->write(&separator, 1); // separate each layer
        data_ctx->DC_WRITE( ctx->sequential_start );
        data_ctx->DC_WRITE( ctx->seq_end );
        data_ctx->write(&separator, 1); // separate each layer
    }

    data_ctx->write(&separator, 1); // final separator
}

size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst) {
    llama_data_buffer_context data_ctx(dst);
    llama_copy_state_data_internal(ctx, &data_ctx);

    return data_ctx.get_size_written();
}

// Sets the state reading from the specified source address
size_t llama_set_state_data(struct llama_context * ctx, const uint8_t * src) {
    uint8_t * inp = (uint8_t*)src;

    int version=0;
    char separator=(char)125, tester;

    ctx->ctx_ready = false;

    // set rng
    {
        size_t rng_size;
        memcpy(&rng_size, inp, sizeof(rng_size));

        if( rng_size == 2121 ) {
            version=1;
        } else if( rng_size == 2122 ) {
            version=2;
        } else if( rng_size == 2123 ) {
            version=3;
        }
        if( version != 0 ) {
            inp += sizeof(rng_size);
            memcpy(&rng_size, inp, sizeof(rng_size));
        }

        LLAMA_LOG_INFO("%s: version %d\n", __func__, version);

        inp += sizeof(rng_size);
        GGML_ASSERT(rng_size <= LLAMA_MAX_RNG_STATE);

        std::string rng_str((const char *)inp, rng_size); inp += rng_size;

        std::istringstream rng_ss(rng_str);
        rng_ss >> ctx->rng;

        GGML_ASSERT(!rng_ss.fail());
    }

    // set logits
    {
        size_t logits_size;
        size_t logits_vocab = ctx->model.hparams.n_vocab*sizeof(float);

        memcpy(&logits_size, inp, sizeof(logits_size)); inp += sizeof(logits_size);

        //GGML_ASSERT(ctx->logits.capacity() >= logits_size);

        if (logits_size) {
            ctx->logits.resize(ctx->model.hparams.n_vocab);

            size_t logits_buf = logits_size * sizeof(float) - logits_vocab;

            memcpy(ctx->logits.data(), inp+logits_buf, logits_vocab);
            inp += logits_buf + logits_vocab;

            LLAMA_LOG_INFO("%s: logits %d\n", __func__, logits_size);
        }
    }

    // set embeddings
    {
        size_t embeddings_size;
        size_t embeddings_embd = ctx->model.hparams.n_embd*sizeof(float);

        memcpy(&embeddings_size, inp, sizeof(embeddings_size)); inp += sizeof(embeddings_size);

        //GGML_ASSERT(ctx->embd.capacity() == embeddings_size);

        if (embeddings_size) {
            ctx->embd.resize(ctx->model.hparams.n_embd);

            size_t embeddings_buf = embeddings_size * sizeof(float) - embeddings_embd;

            memcpy(ctx->embd.data(), inp+embeddings_buf, embeddings_embd);
            inp += embeddings_buf + embeddings_embd;
            LLAMA_LOG_INFO("%s: embd %d\n", __func__, embeddings_size);
        }
    }

    // set kv cache
    {
        auto & kv_self = ctx->kv_self;
        uint32_t kv_used, kv_usable;

        if( version >= 3 ) {
            DC_READ( tester );
            if( separator != tester ) {
                LLAMA_LOG_ERROR("%s: couldn't match separator 1\n", __func__);
                return 0;
            }
        }

        if( version < 3 ) {
            size_t a;
            uint32_t b, kv_full;
            DC_READ( a );
            DC_READ( kv_used );
            DC_READ( kv_full );
            LLAMA_LOG_INFO("version<3 a=%zu b=%lu full=%lu\n", a, kv_used, kv_full);
            //inp += sizeof(size_t) + sizeof(uint32_t);
        } else {
            memcpy(&kv_used,     inp, sizeof(kv_used));     inp += sizeof(kv_used);
        }
        kv_usable =( kv_used > kv_self->size ) ? kv_self->size : kv_used;

        if( kv_usable < kv_used ) {
            LLAMA_LOG_WARN("%s: kv_usable %lu < kv_used %lu, size %lu\n", __func__, kv_usable, kv_used, kv_self->size);
        } else {
            LLAMA_LOG_INFO("%s: kv_usable %lu, kv_used %lu, size %lu\n", __func__, kv_usable, kv_used, kv_self->size);
        }

        if (kv_used > 0) {
            size_t v_row_stride = 2*kv_self->size;
            size_t v_row_size = 2*kv_used;

            size_t v_size_usable = 2*kv_usable;
            size_t v_size_used = 2*kv_used;

            size_t k_size = 2048*kv_used;
            size_t k_size_usable = 2048*kv_usable;
            size_t k_size_used = 2048*kv_used;

            //LLAMA_LOG_INFO("%s: k_l size %d\n", __func__, ggml_element_size(kv_self->k_l[0]));
            //LLAMA_LOG_INFO("%s: v_l size %d\n", __func__, ggml_element_size(kv_self->v_l[0]));
            for (int il = 0; il < (int) 32; ++il) {
                ggml_backend_tensor_set(kv_self->k_l[il], inp+(k_size_used-k_size_usable), 0, k_size_usable);
                inp += k_size;

                for (int ir = 0; ir < (int) 1024; ++ir) {
                    ggml_backend_tensor_set(kv_self->v_l[il], inp+(v_size_used-v_size_usable), ir*v_row_stride, v_size_usable);
                    inp += v_size_used;
                }

                if( version >= 3 ) {
                    DC_READ( tester );
                    if( separator != tester ) {
                        LLAMA_LOG_ERROR("%s: couldn't match layer separator %d\n", __func__, il);
                        return 0;
                    }
                }
            }
        }

        ctx->seq_end = kv_used;
        ctx->sequential_start = 0;

        /*
        if( version >= 2 ) {
            LLAMA_LOG_INFO("Read inuse ptr: %d bytes\n", 4096/8);
            memcpy( kv_self->inuse, inp, kv_self->size/8 );
            inp += kv_self->size/8;
        } else {
            ctx->kv_self->use_tokens(0, kv_used, true);
        }
        */

        if( version >= 3 ) {
            DC_READ( tester );
            if( separator != tester ) {
                LLAMA_LOG_ERROR("%s: couldn't match separator 2\n", __func__);
                return 0;
            }
        }
    }
    if( version >= 3 ) {
        DC_READ( tester );
        if( separator != tester ) {
            LLAMA_LOG_ERROR("%s: couldn't match final separator\n", __func__);
            return 0;
        }
    }


    const size_t nread    = inp - src;

    LLAMA_LOG_INFO("%s: kv context ready, read %zu\n", __func__, nread);
    ctx->ctx_ready = true;

    return nread;
}

static bool llama_load_session_file_internal(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(path_session, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s : unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }

        llama_hparams session_hparams;
        file.read_raw(&session_hparams, sizeof(llama_hparams));

        if (session_hparams != ctx->model.hparams) {
            LLAMA_LOG_INFO("%s : model hparams didn't match from session file!\n", __func__);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s : token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size - file.tell();

        std::vector<uint8_t> state_data(n_state_size_cur);
        file.read_raw(state_data.data(), n_state_size_cur);

        llama_set_state_data(ctx, state_data.data());
    }

    return true;
}

bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_load_session_file_internal(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("error loading session file: %s\n", err.what());
        return false;
    }
}

bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    llama_file file(path_session, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    file.write_raw(&ctx->model.hparams, sizeof(llama_hparams));

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_data_file_context data_ctx(&file);
    llama_copy_state_data_internal(ctx, &data_ctx);

    return true;
}

void llama_set_n_threads(struct llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch) {
    ctx->cparams.n_threads       = n_threads;
    ctx->cparams.n_threads_batch = n_threads_batch;
}

void llama_set_abort_callback(struct llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->abort_callback      = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens,
               llama_pos   pos_0,
            llama_seq_id   seq_id) {
    return {
        /*n_tokens       =*/ n_tokens,
        /*tokens         =*/ tokens,
        /*embd           =*/ nullptr,
        /*all_pos_0      =*/ pos_0,
        /*all_pos_1      =*/ 1
    };
}

struct llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    llama_batch batch = { 0, nullptr, nullptr, 0, 0, 0 };

    if (embd != 0) {
        batch.embd = (float *) pool_alloc(sizeof(float) * n_tokens_alloc * embd);
    } else {
        batch.token = (llama_token *) pool_alloc(sizeof(llama_token) * n_tokens_alloc);
    }

    return batch;
}

void llama_batch_free(struct llama_batch batch) {
    if (batch.token)    pool_free(batch.token);
    if (batch.embd)     pool_free(batch.embd);
}

void llama_set_key( struct llama_context * ctx, std::string keyfor, std::string key, std::string keyval )
{
    System_kb *kv = current_kb;
    System_actor *a;
    uint8_t tgt=99;
    std::vector<llama_token> tokens;

    if( keyfor == "all" ) {
        std::vector<System_actor*>::iterator it;
        for( it = kv->actors.begin(); it != kv->actors.end(); it++ ) {
            a = *it;

            llama_set_key(ctx, a->name, key, keyval);
        }
        return;
    }

    if( key == "self" ) {
        if( !kv->players.contains(keyfor) ) {
            LLAMA_LOG_INFO("%s: load player %s\n", __func__, keyfor.c_str());
            tgt = kv->useactor(keyfor);
            current_kb->active_actor = keyfor; // decide who is responding with logits

            std::string intro = "<|im_start|>System\nWelcome back. It is currently ";
            intro.append(readable_ts());
            intro.append(".<|im_end|>\n");

            Kv_mem *mem = kv->processtokens("System", intro, tokens);
            a = kv->players[ keyfor ];

            LLAMA_LOG_INFO("%s: intro prepared for %s.\n", __func__, keyfor.c_str());
        } else {
            LLAMA_LOG_INFO("%s: not found player %s\n", __func__, keyfor.c_str());
            return;
        }
    } else {
        a = kv->players[ keyfor ];
    }


    // remove any old mem or self key that matches
    if( key == "self" ) {
        // if it's the same keyval, don't process it we already have it.
        if( a->self->what == keyval ) {
            LLAMA_LOG_INFO("ignore duplicate key %s\n", key.c_str());
            return;
        }
        if( a->mine ) {
            a->mine->release();
            pool_free( a->mine );
        }
    } else if( a->keys.contains(key) ) {
        Kv_mem *target = a->keys[key];

        // we have to look for it in mem
        std::vector<Kv_mem *>::iterator it;
        for( it = a->mem.begin(); it != a->mem.end(); it++ ) {
            if( *it == target ) {
                // if it's the same keyval, don't process it we already have it.
                if( (*it)->e->what == keyval ) {
                    LLAMA_LOG_INFO("ignore duplicate key %s\n", key.c_str());
                    return;
                }
                a->mem.erase(it);
                break;
            }
        }

        target->release();
        pool_free(target);
    }

    // process the tokens
    if( tgt == 99 )
        tgt = kv->useactor(a->name);

    Kv_mem *mem = kv->processtokens(a->name, keyval, tokens, true, 0);
    if( key == "self" ) {
        a->self = mem->e;
        a->mine = mem;
    } else {
        a->keys[key] = mem;
        kv->kvuser[tgt]->mem.push_back(mem);
    }
}

void llama_kv_cache_shift_fwd( struct llama_context *ctx, int fwd)
{
    int moving_tokens = (ctx->seq_end - ctx->sequential_start);
    if( moving_tokens == 0 ) {
        ctx->seq_end = ctx->sequential_start = ctx->sequential_start+fwd;
        return;
    }
    LLAMA_LOG_INFO("shift_fwd(fwd=%d, seq_end=%d, moving %d tokens)\n", fwd, ctx->seq_end, moving_tokens);
    if( fwd < 64 )
        fwd=64;
    if( fwd + ctx->seq_end > 4096 )
        moving_tokens -= ( (fwd+ctx->seq_end) - 4096 );

    size_t empty_sz = 2048 * fwd;
    size_t moving_sz = 2048 * moving_tokens, moving_sz2 = 2 * moving_tokens;
    size_t offset = 2048 * ctx->sequential_start;

    int v = 2 * ctx->sequential_start;
    int s = 2 * (ctx->sequential_start + fwd);
    int x = 2 * fwd;

    size_t buffer_sz = 2048*ctx->kv_self->size;

    void *nd = pool_alloc( moving_sz );
    void *nd2 = pool_alloc( empty_sz );
    ggml_fp16_t z = fp32_to_fp16(0);
    for( int i=0; i<empty_sz; i+=2 ) {
        memcpy((void*)((char*)nd2+i),&z,2);
    }
    for( int il=0; il<32; il++ ) {
        ggml_backend_tensor_get(ctx->kv_self->k_l[il], nd, offset, moving_sz );
        ggml_backend_tensor_set(ctx->kv_self->k_l[il], nd, offset+empty_sz, moving_sz );
        if( empty_sz != 0 )
            ggml_backend_tensor_set(ctx->kv_self->k_l[il], nd2, offset, empty_sz);

        for( size_t k=0; k<buffer_sz; k+=8192 ) {
            ggml_backend_tensor_get(ctx->kv_self->v_l[il], nd, k + v, moving_sz2 );
            ggml_backend_tensor_set(ctx->kv_self->v_l[il], nd, k + s, moving_sz2 );
            ggml_backend_tensor_set(ctx->kv_self->v_l[il], nd2, k + v, x );
        }
    }
    pool_free(nd2);
    pool_free(nd);

    //ctx->kv_self->use_tokens(ctx->sequential_start, fwd, false);
    ctx->sequential_start += fwd;
    ctx->seq_end += fwd;
    /*
    if( ctx->amnesia_start > 0 )
        ctx->amnesia_start = ((fwd+ctx->amnesia_start) >= ctx->sequential_start) ? (fwd+ctx->amnesia_start) : ( ctx->sequential_start );
    */
}
void llama_kv_cache_shift_rev( struct llama_context *ctx, int target, int skip, int endpt)
{
    if( skip < 24 ) skip=24;

    LLAMA_LOG_INFO("%s: target %d skip %d endpt %d\n", __func__, target, skip, endpt);

    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * , const char * , int ) {
        //Scott. Asrlrsa.
    };


    Org_context llm(ctx->kv_self, ctx->buf_compute_meta, ctx->sched, ctx->model.hparams);
    llm.init();

    int readptr = target+skip;
    int range = ctx->seq_end-readptr;
    if( range <= 0 ) {
        // just blank the area and reset the pointers
        //ctx->kv_self->use_tokens(target, ctx->seq_end-target, false);
        ctx->seq_end = target;
        return;
    }

    llm.shuffle_kv_now( readptr, ctx->seq_end, -skip );


    //ctx->kv_self->use_tokens(ctx->seq_end-skip, skip, false);
    ctx->seq_end -= skip;

    //llm.run_kv_shuffler();
    llm.free();
    /*
    if( ctx->amnesia_start > readptr )
        ctx->amnesia_start -= skip;
    */
    LLAMA_LOG_INFO("%s: shuffle completed\n", __func__);
}

int llama_kv_cache_reserve( struct llama_context *ctx, int tokens )
{
    tokens += 4;
    if( ctx->seq_end + tokens <= ctx->kv_self->size ) return ctx->seq_end;

    if( tokens < 256 ) tokens=256; // reserve enough tokens to finish a sentence at least...

    llama_kv_cache_shift_rev( ctx, ctx->sequential_start, tokens, ctx->kv_self->size );
    return ctx->seq_end;
}

int32_t llama_decode(
        struct llama_context * ctx,
        struct llama_batch   batch
        ) {

    if( batch.n_tokens + ctx->seq_end > ctx->kv_self->size ) {
        LLAMA_LOG_INFO("kv_cache_over: %d + %d > %d\n", batch.n_tokens, ctx->seq_end, ctx->kv_self->size);
        return -1;
    }

    //current_kb->process_tokens()

    const int ret = llama_decode_internal(*ctx, batch);
    if (ret < 0) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int llama_get_logits_size(struct llama_context * ctx) {
    return ctx->logits.size();
}

float * llama_get_logits(struct llama_context * ctx) {
    return ctx->logits.data();
}

float * llama_get_logits_ith(struct llama_context * ctx, int32_t i) {
    if( ctx->record_all == 0 )
        return ctx->logits.data();
    else
        return ctx->logits.data() + i*ctx->model.hparams.n_vocab;
}

int llama_get_embeddings_size(struct llama_context * ctx) {
    return ctx->embd.size();
}

float * llama_get_embeddings(struct llama_context * ctx) {
    return ctx->embd.data();
}

float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i) {
    if( ctx->record_all == 0 )
        return ctx->embd.data();
    else
        return ctx->embd.data() + i*ctx->model.hparams.n_embd;
}

const char * llama_model_name(const struct llama_model * model) {
    return model->name.c_str();
}

const char * llama_model_arch(const struct llama_model * model) {
    return LLM_ARCH_NAMES.at(model->arch);
}

const char * llama_token_get_text(const struct llama_model * model, llama_token token) {
    return model->vocab.id_to_token[token].text.c_str();
}

float llama_token_get_score(const struct llama_model * model, llama_token token) {
    return model->vocab.id_to_token[token].score;
}

llama_token_type llama_token_get_type(const struct llama_model * model, llama_token token) {
    return model->vocab.id_to_token[token].type;
}

llama_token llama_token_bos(const struct llama_model * model) {
    return model->vocab.special_bos_id;
}

llama_token llama_token_eos(const struct llama_model * model) {
    return model->vocab.special_eos_id;
}

llama_token llama_token_nl(const struct llama_model * model) {
    return model->vocab.linefeed_id;
}

int32_t llama_add_bos_token(const struct llama_model * model) {
    return model->vocab.special_add_bos;
}

int32_t llama_add_eos_token(const struct llama_model * model) {
    return model->vocab.special_add_eos;
}

llama_token llama_token_prefix(const struct llama_model * model) {
    return model->vocab.special_prefix_id;
}

llama_token llama_token_middle(const struct llama_model * model) {
    return model->vocab.special_middle_id;
}

llama_token llama_token_suffix(const struct llama_model * model) {
    return model->vocab.special_suffix_id;
}

llama_token llama_token_eot(const struct llama_model * model) {
    return model->vocab.special_eot_id;
}

bool llama_use_bos(llama_model *model)
{
    int add_bos = llama_add_bos_token(model);
    if (add_bos != -1) { return add_bos; }
    auto vocab_type = llama_vocab_type(model);
    return vocab_type == LLAMA_VOCAB_TYPE_SPM || vocab_type == LLAMA_VOCAB_TYPE_WPM;
}

uint16_t llama_tokenstr(
        llama_model *model,
        std::string text,
        std::vector<llama_token> &tokens,
        bool at_start )
{
    bool want_bos = at_start;
    bool use_bos = llama_use_bos(model);

    auto strCat = want_bos ? " " + text : text;
    tokens.reserve(strCat.size()+4);

    auto res = llama_tokenize_internal(model->vocab, strCat, use_bos, false);
    uint16_t i;

    for (i = 0; i < res.size(); i++) {
        tokens.push_back( (llama_token)res[i] );
    }

    return i;
}
int32_t llama_tokenize(
    const struct llama_model * model,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_max_tokens,
                        bool   add_bos,
                        bool   special) {
    int32_t i;
    auto res = llama_tokenize_internal(model->vocab, std::string(text, text_len), add_bos, special);

    if (n_max_tokens < (int) res.size()) {
        LLAMA_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    LLAMA_LOG_INFO("(%d) %s\n", i, text);

    return i;
}

static std::string llama_decode_text(const std::string & text) {
    std::string decoded_text;
    auto unicode_sequences = codepoints_from_utf8(text);
    for (auto& unicode_sequence : unicode_sequences) {
        decoded_text += unicode_to_bytes_bpe(codepoint_to_utf8(unicode_sequence));
    }

    return decoded_text;
}

// does not write null-terminator to buf
int32_t llama_token_to_piece(const struct llama_model * model, llama_token token, char * buf, int32_t length) {
    if (0 <= token && token < llama_n_vocab(model)) {
        switch (llama_vocab_get_type(model->vocab)) {
        case LLAMA_VOCAB_TYPE_WPM:
        case LLAMA_VOCAB_TYPE_SPM: {
            // NOTE: we accept all unsupported token types,
            // suppressing them like CONTROL tokens.
            if (llama_is_normal_token(model->vocab, token)) {
                std::string result = model->vocab.id_to_token[token].text;
                llama_unescape_whitespace(result);
                if (length < (int) result.length()) {
                    return -(int) result.length();
                }
                memcpy(buf, result.c_str(), result.length());
                return result.length();
            } else if (llama_is_user_defined_token(model->vocab, token)) {
                std::string result = model->vocab.id_to_token[token].text;
                if (length < (int) result.length()) {
                    return -result.length();
                }
                memcpy(buf, result.c_str(), result.length());
                return result.length();
            } else if (llama_is_unknown_token(model->vocab, token)) { // NOLINT
                if (length < 3) {
                    return -3;
                }
                memcpy(buf, "\xe2\x96\x85", 3);
                return 3;
            } else if (llama_is_control_token(model->vocab, token)) {
                ;
            } else if (llama_is_byte_token(model->vocab, token)) {
                if (length < 1) {
                    return -1;
                }
                buf[0] = llama_token_to_byte(model->vocab, token);
                return 1;
            }
            break;
        }
        case LLAMA_VOCAB_TYPE_BPE: {
            // NOTE: we accept all unsupported token types,
            // suppressing them like CONTROL tokens.
            if (llama_is_normal_token(model->vocab, token)) {
                std::string result = model->vocab.id_to_token[token].text;
                result = llama_decode_text(result);
                if (length < (int) result.length()) {
                    return -(int) result.length();
                }
                memcpy(buf, result.c_str(), result.length());
                return result.length();
            } else if (llama_is_user_defined_token(model->vocab, token)) {
                std::string result = model->vocab.id_to_token[token].text;
                if (length < (int) result.length()) {
                    return -result.length();
                }
                memcpy(buf, result.c_str(), result.length());
                return result.length();
            } else if (llama_is_control_token(model->vocab, token)) {
                ;
            }
            break;
        }
        default:
            GGML_ASSERT(false);
        }
    }
    return 0;
}

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && isspace(str[start])) {
        start += 1;
    }
    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }
    return str.substr(start, end - start);
}

// Simple version of "llama_apply_chat_template" that only works with strings
// This function uses heuristic checks to determine commonly used template. It is not a jinja parser.
static int32_t llama_chat_apply_template_internal(
    const std::string & tmpl,
    const std::vector<const llama_chat_message *> & chat,
    std::string & dest, bool add_ass) {
    // Taken from the research: https://github.com/ggerganov/llama.cpp/issues/5527
    std::stringstream ss;
    if (tmpl == "chatml" || tmpl.find("<|im_start|>") != std::string::npos) {
        // chatml template
        for (auto message : chat) {
            ss << "<|im_start|>" << message->role << "\n" << message->content << "<|im_end|>\n";
        }
        if (add_ass) {
            ss << "<|im_start|>assistant\n";
        }
    } else if (tmpl == "llama2" || tmpl.find("[INST]") != std::string::npos) {
        // llama2 template and its variants
        // [variant] support system message
        bool support_system_message = tmpl.find("<<SYS>>") != std::string::npos;
        // [variant] space before + after response
        bool space_around_response = tmpl.find("' ' + eos_token") != std::string::npos;
        // [variant] add BOS inside history
        bool add_bos_inside_history = tmpl.find("bos_token + '[INST]") != std::string::npos;
        // [variant] trim spaces from the input message
        bool strip_message = tmpl.find("content.strip()") != std::string::npos;
        // construct the prompt
        bool is_inside_turn = true; // skip BOS at the beginning
        ss << "[INST] ";
        for (auto message : chat) {
            std::string content = strip_message ? trim(message->content) : message->content;
            std::string role(message->role);
            if (!is_inside_turn) {
                is_inside_turn = true;
                ss << (add_bos_inside_history ? "<s>[INST] " : "[INST] ");
            }
            if (role == "system") {
                if (support_system_message) {
                    ss << "<<SYS>>\n" << content << "\n<</SYS>>\n\n";
                } else {
                    // if the model does not support system message, we still include it in the first message, but without <<SYS>>
                    ss << content << "\n";
                }
            } else if (role == "user") {
                ss << content << " [/INST]";
            } else {
                ss << (space_around_response ? " " : "") << content << (space_around_response ? " " : "") << "</s>";
                is_inside_turn = false;
            }
        }
        // llama2 templates seem to not care about "add_generation_prompt"
    } else if (tmpl == "zephyr" || tmpl.find("<|user|>") != std::string::npos) {
        // zephyr template
        for (auto message : chat) {
            ss << "<|" << message->role << "|>" << "\n" << message->content << "<|endoftext|>\n";
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == "monarch" || tmpl.find("bos_token + message['role']") != std::string::npos) {
        // mlabonne/AlphaMonarch-7B template (the <s> is included inside history)
        for (auto message : chat) {
            std::string bos = (message == chat.front()) ? "" : "<s>"; // skip BOS for first message
            ss << bos << message->role << "\n" << message->content << "</s>\n";
        }
        if (add_ass) {
            ss << "<s>assistant\n";
        }
    } else if (tmpl == "gemma" || tmpl.find("<start_of_turn>") != std::string::npos) {
        // google/gemma-7b-it
        std::string system_prompt = "";
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                // there is no system message for gemma, but we will merge it with user prompt, so nothing is broken
                system_prompt = trim(message->content);
                continue;
            }
            // in gemma, "assistant" is "model"
            role = role == "assistant" ? "model" : message->role;
            ss << "<start_of_turn>" << role << "\n";
            if (!system_prompt.empty() && role != "model") {
                ss << system_prompt << "\n\n";
                system_prompt = "";
            }
            ss << trim(message->content) << "<end_of_turn>\n";
        }
        if (add_ass) {
            ss << "<start_of_turn>model\n";
        }
    } else {
        // template not supported
        return -1;
    }
    dest = ss.str();
    return dest.size();
}

LLAMA_API int32_t llama_chat_apply_template(
                const struct llama_model * model,
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    std::string curr_tmpl(tmpl == nullptr ? "" : tmpl);
    if (tmpl == nullptr) {
        GGML_ASSERT(model != nullptr);
        // load template from model
        std::vector<char> model_template(2048, 0); // longest known template is about 1200 bytes
        std::string template_key = "tokenizer.chat_template";
        int32_t res = llama_model_meta_val_str(model, template_key.c_str(), model_template.data(), model_template.size());
        if (res < 0) {
            // worst case: there is no information about template, we will use chatml by default
            curr_tmpl = "chatml"; // see llama_chat_apply_template_internal
        } else {
            curr_tmpl = std::string(model_template.data(), model_template.size());
        }
    }
    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }
    std::string formatted_chat;
    int32_t res = llama_chat_apply_template_internal(curr_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    strncpy(buf, formatted_chat.c_str(), length);
    return res;
}

struct llama_timings llama_get_timings(struct llama_context * ctx) {
    struct llama_timings result = {
        /*.t_start_ms  =*/ 1e-3 * ctx->t_start_us,
        /*.t_end_ms    =*/ 1.00 * ggml_time_ms(),
        /*.t_load_ms   =*/ 1e-3 * ctx->t_load_us,
        /*.t_sample_ms =*/ 1e-3 * ctx->t_sample_us,
        /*.t_p_eval_ms =*/ 1e-3 * ctx->t_p_eval_us,
        /*.t_eval_ms   =*/ 1e-3 * ctx->t_eval_us,

        /*.n_sample =*/ std::max(1, ctx->n_sample),
        /*.n_p_eval =*/ std::max(1, ctx->n_p_eval),
        /*.n_eval   =*/ std::max(1, ctx->n_eval),
    };

    ctx->t_load_us = 0;

    return result;
}

void llama_print_timings(struct llama_context * ctx) {
    llama_timings timings = llama_get_timings(ctx);

    LLAMA_LOG_INFO("\n");
    if( timings.t_load_ms != 0 ) {
        LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, timings.t_load_ms);
        timings.t_load_ms = 0;
    }
    LLAMA_LOG_INFO("%s:      sample time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_sample_ms, timings.n_sample, timings.t_sample_ms / timings.n_sample, 1e3 / timings.t_sample_ms * timings.n_sample);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_p_eval_ms, timings.n_p_eval, timings.t_p_eval_ms / timings.n_p_eval, 1e3 / timings.t_p_eval_ms * timings.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_eval_ms, timings.n_eval, timings.t_eval_ms / timings.n_eval, 1e3 / timings.t_eval_ms * timings.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (timings.t_end_ms - timings.t_start_ms), (timings.n_p_eval + timings.n_eval));
}

void llama_reset_timings(struct llama_context * ctx) {
    ctx->t_start_us = ggml_time_us();
    ctx->t_sample_us = ctx->n_sample = 0;
    ctx->t_eval_us   = ctx->n_eval   = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "         + std::to_string(ggml_cpu_has_avx())         + " | ";
    s += "AVX_VNNI = "    + std::to_string(ggml_cpu_has_avx_vnni())    + " | ";
    s += "AVX2 = "        + std::to_string(ggml_cpu_has_avx2())        + " | ";
    s += "AVX512 = "      + std::to_string(ggml_cpu_has_avx512())      + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
    s += "FMA = "         + std::to_string(ggml_cpu_has_fma())         + " | ";
    s += "NEON = "        + std::to_string(ggml_cpu_has_neon())        + " | ";
    s += "ARM_FMA = "     + std::to_string(ggml_cpu_has_arm_fma())     + " | ";
    s += "F16C = "        + std::to_string(ggml_cpu_has_f16c())        + " | ";
    s += "FP16_VA = "     + std::to_string(ggml_cpu_has_fp16_va())     + " | ";
    s += "WASM_SIMD = "   + std::to_string(ggml_cpu_has_wasm_simd())   + " | ";
    s += "BLAS = "        + std::to_string(ggml_cpu_has_blas())        + " | ";
    s += "SSE3 = "        + std::to_string(ggml_cpu_has_sse3())        + " | ";
    s += "SSSE3 = "       + std::to_string(ggml_cpu_has_ssse3())       + " | ";
    s += "VSX = "         + std::to_string(ggml_cpu_has_vsx())         + " | ";
    s += "MATMUL_INT8 = " + std::to_string(ggml_cpu_has_matmul_int8()) + " | ";

    return s.c_str();
}

void llama_dump_timing_info_yaml(FILE * stream, const llama_context * ctx) {
    fprintf(stream, "\n");
    fprintf(stream, "###########\n");
    fprintf(stream, "# Timings #\n");
    fprintf(stream, "###########\n");
    fprintf(stream, "\n");

    fprintf(stream, "mst_eval: %.2f  # ms / token during generation\n",
            1.0e-3 * ctx->t_eval_us / ctx->n_eval);
    fprintf(stream, "mst_p_eval: %.2f  # ms / token during prompt processing\n",
            1.0e-3 * ctx->t_p_eval_us / ctx->n_p_eval);
    fprintf(stream, "mst_sample: %.2f  # ms / token during sampling\n",
            1.0e-3 * ctx->t_sample_us / ctx->n_sample);
    fprintf(stream, "n_eval: %d  # number of tokens generated (excluding the first one)\n", ctx->n_eval);
    fprintf(stream, "n_p_eval: %d  # number of tokens processed in batches at the beginning\n", ctx->n_p_eval);
    fprintf(stream, "n_sample: %d  # number of sampled tokens\n", ctx->n_sample);
    fprintf(stream, "t_eval_us: %" PRId64 "  # total microseconds spent generating tokens\n", ctx->t_eval_us);
    fprintf(stream, "t_load_us: %" PRId64 "  # total microseconds spent loading the model\n", ctx->t_load_us);
    fprintf(stream, "t_p_eval_us: %" PRId64 "  # total microseconds spent prompt processing\n", ctx->t_p_eval_us);
    fprintf(stream, "t_sample_us: %" PRId64 "  # total microseconds spent sampling\n", ctx->t_sample_us);
    fprintf(stream, "ts_eval: %.2f  # tokens / second during generation\n",
            1.0e6 * ctx->n_eval / ctx->t_eval_us);
    fprintf(stream, "ts_p_eval: %.2f  # tokens / second during prompt processing\n",
            1.0e6 * ctx->n_p_eval / ctx->t_p_eval_us);
    fprintf(stream, "ts_sample: %.2f  # tokens / second during sampling\n",
            1.0e6 * ctx->n_sample / ctx->t_sample_us);
}

// For internal test use
const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
) {
    return ctx->model.tensors_by_name;
}

void llama_log_set(ggml_log_callback log_callback, void * user_data) {
    g_state.log_callback = log_callback ? log_callback : llama_log_callback_default;
    g_state.log_callback_user_data = user_data;
#ifdef GGML_USE_METAL
    ggml_backend_metal_log_set_callback(g_state.log_callback, g_state.log_callback_user_data);
#endif
}

static void llama_log_internal_v(ggml_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args_copy);
        buffer2[len] = 0;
        g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}

static void llama_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, args);
    va_end(args);
}

static void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}
