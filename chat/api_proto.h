// OpenAI-compatible HTTP API, protocol part: HTTP/1.1 request parsing, a small JSON reader and the JSON the API
// answers with. No console calls here, so the same code is tested on Windows and Linux (tests/api_proto_test.c).
// The server itself is api.c (PS4 sockets); the work on the model is api_run.inc (main.c).
#ifndef CHAT_API_PROTO_H
#define CHAT_API_PROTO_H

#include <stddef.h>
#include <stdint.h>

// ---- growing string -----------------------------------------------------------------------------------------------
typedef struct {
    char* p;
    size_t n, cap;
    int oom;
} Sb;

void sb_init(Sb* s);
void sb_free(Sb* s);
void sb_add(Sb* s, const char* p, size_t n);
void sb_str(Sb* s, const char* z);
void sb_json_str(Sb* s, const char* p, size_t n);   // a JSON string literal, quotes included
void sb_int(Sb* s, long long v);
void sb_base64(Sb* out, const uint8_t* data, size_t len);
// Builds a 24 kHz (or sample_rate) 16-bit mono RIFF WAVE file in out from n_samples 16-bit signed PCM samples.
void api_wav_from_pcm16(Sb* out, const int16_t* pcm, size_t n_samples, int sample_rate);

// ---- HTTP request ---------------------------------------------------------------------------------------------------
typedef struct {
    char method[8];
    char path[256];
    char auth[256];             // the Authorization header's value ("" if none)
    long content_length;        // 0 if none
    const char* body;           // points into the buffer given to api_http_parse
    size_t body_len;
    size_t head_len;            // bytes of head (up to and including the blank line)
} HttpRequest;

// Returns 1 when buf holds a whole request, 0 when more bytes are needed, -1 on a malformed or too big request.
int api_http_parse(const char* buf, size_t len, HttpRequest* req);

// ---- JSON reader --------------------------------------------------------------------------------------------------
// The whole document is parsed into a pool of nodes; children are linked (first child, then next sibling). An
// object's children carry their key in key/klen.
enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ };
typedef struct JNode {
    uint8_t type;
    int b;                      // J_BOOL value
    double num;                 // J_NUM value
    const char* s;              // J_STR: the raw text between the quotes (escapes not decoded)
    size_t slen;
    const char* key;            // member name in an object (raw, as above)
    size_t klen;
    int first, next, count;     // node indices; -1 = none
} JNode;

// Parses one JSON value from s (n bytes) into pool (max nodes). Returns the root index, or -1 on error.
int json_parse(const char* s, size_t n, JNode* pool, int max);
// The member named key of object node obj (-1 if absent or obj is not an object).
int json_get(const JNode* pool, int obj, const char* key);
// Appends the decoded text of a string node to out (escapes, \uXXXX and surrogate pairs turned into UTF-8).
void json_str_sb(const JNode* pool, int node, Sb* out);
// The text of one chat message: its role (copied into role, NUL-terminated) and its content into out. Content may be
// a string or an array of parts (their "text" parts are joined). Returns 0, or -1 if the message is malformed.
int api_message(const JNode* pool, int msg, char* role, size_t rcap, Sb* out);
// Writes any parsed JSON node back as compact JSON text into out.
void json_write(const JNode* pool, int node, Sb* out);

// ---- tool calls ---------------------------------------------------------------------------------------------------
#define API_MAX_NODES 4096
#define API_MAX_TOOL_CALLS 16

enum { TOOL_NONE = 0, TOOL_AUTO = 1, TOOL_REQUIRED = 2, TOOL_NAMED = 3 };

typedef struct {
    char id[40];                // e.g. "call_1"
    char name[64];              // function name
    Sb arguments;               // JSON object string, e.g. "{\"query\":\"ps4\"}"
} ApiToolCall;

void api_free_tool_calls(ApiToolCall* calls, int n);

// ---- JSON writers used by the API ------------------------------------------------------------------------------------
// The JSON of GET /v1/models: one model, the one loaded.
void api_json_models(Sb* out, const char* model, long long created);
// A complete chat.completion (non-streaming).
void api_json_completion(Sb* out, const char* id, const char* model, long long created, const char* text, size_t tlen,
                         const char* finish, int prompt_tokens, int completion_tokens);
// A complete chat.completion with optional tool_calls and optional synthesized audio.
void api_json_completion_tools(Sb* out, const char* id, const char* model, long long created, const char* text,
                               size_t tlen, const ApiToolCall* calls, int n_calls, const char* finish,
                               int prompt_tokens, int completion_tokens);
void api_json_completion_full(Sb* out, const char* id, const char* model, long long created, const char* text,
                              size_t tlen, const ApiToolCall* calls, int n_calls, const char* audio_id,
                              const uint8_t* audio_data, size_t audio_len, const char* finish, int prompt_tokens,
                              int completion_tokens);
// One chat.completion.chunk (streaming). role != NULL: the first chunk, with the role. text may be empty.
void api_json_chunk(Sb* out, const char* id, const char* model, long long created, const char* role, const char* text,
                    size_t tlen, const char* finish);
// One streaming chat.completion.chunk carrying a single tool_call delta at index.
void api_json_chunk_tool(Sb* out, const char* id, const char* model, long long created, int index,
                         const ApiToolCall* call);
// One streaming chat.completion.chunk carrying synthesized audio (base64 data + transcript).
void api_json_chunk_audio(Sb* out, const char* id, const char* model, long long created, const char* audio_id,
                          const uint8_t* audio_data, size_t audio_len, const char* transcript, size_t tlen);
// The error object: {"error": {"message", "type", "code"}}.
void api_json_error(Sb* out, const char* message, const char* type, const char* code);

// ---- chat request -------------------------------------------------------------------------------------------------
typedef struct {
    int root;                   // root node (in the pool the caller gives)
    int messages;               // the messages array node (-1 if missing)
    int n_messages;
    int tools;                  // the tools array node (-1 if missing or empty)
    int n_tools;
    int tool_choice;            // TOOL_NONE, TOOL_AUTO, TOOL_REQUIRED, TOOL_NAMED
    char tool_name[64];         // function name when tool_choice == TOOL_NAMED
    int want_audio;             // 1 if audio reply was requested (via modalities or audio)
    int audio_pcm;              // 0 = wav (default), 1 = pcm16 (raw 24 kHz s16le)
    char audio_voice[16];       // "" / "auto" or language code ("en", "es", "pt", "ru", "it", "fr", "de")
    double temperature;         // default 0.7
    int max_tokens;             // default 512
    int stream;                 // 0 or 1
    unsigned seed;              // default 0 = the engine's own seed
    int has_seed;
} ChatRequest;

// Reads a chat.completions body. Returns 0 if valid (fills req; messages are pool nodes, each an object with role
// and content), or -1 with the reason in err (a fixed string, for the 400 answer).
int api_chat_parse(const char* body, size_t len, JNode* pool, int max, ChatRequest* req, const char** err);

// Builds the Qwen3.5 system prompt with tool declarations (plus user_sys if non-empty).
void api_system_tools(const JNode* pool, const ChatRequest* req, const char* user_sys, Sb* out);
// Formats an assistant message's tool_calls array into Qwen3.5 <tool_call><function=...> tags. Returns count, or -1.
int api_assistant_tool_calls(const JNode* pool, int msg, Sb* out);
// Parses <tool_call> blocks from the model's reply. Text before the first <tool_call> goes to content_out; returns
// the number of tool calls parsed into calls (up to max_calls).
int api_parse_tool_calls(const JNode* pool, const ChatRequest* req, const char* text, size_t len, Sb* content_out,
                         ApiToolCall* calls, int max_calls);

#endif
