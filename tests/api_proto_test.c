// Host tests for chat/api_proto.c (the OpenAI-compatible API's protocol part).
// Build and run: gcc -std=c99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/api_proto_test.c chat/api_proto.c -o build/api_proto_test && build/api_proto_test
#include <stdio.h>
#include <string.h>

#include "api_proto.h"

static int fails, checks;
#define CHECK(c)                                                              \
    do {                                                                      \
        checks++;                                                             \
        if (!(c)) {                                                           \
            fails++;                                                          \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);               \
        }                                                                     \
    } while (0)

static void test_http(void) {
    HttpRequest r;
    const char* post = "POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: ps4\r\nContent-Type: application/json\r\n"
                       "content-length: 13\r\nAuthorization: Bearer abc\r\n\r\n{\"a\":\"b\"}    ";
    CHECK(api_http_parse(post, strlen(post), &r) == 1);
    CHECK(!strcmp(r.method, "POST"));
    CHECK(!strcmp(r.path, "/v1/chat/completions"));
    CHECK(r.content_length == 13);
    CHECK(r.body_len == 13 && !memcmp(r.body, "{\"a\":\"b\"}    ", 13));
    CHECK(!strcmp(r.auth, "Bearer abc"));

    const char* partial = "POST /x HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc";
    CHECK(api_http_parse(partial, strlen(partial), &r) == 0);
    const char* head_only = "GET /v1/models HTTP/1.1\r\nHost: a\r\n";
    CHECK(api_http_parse(head_only, strlen(head_only), &r) == 0);
    const char* get = "GET /v1/models HTTP/1.1\r\nHost: a\r\n\r\n";
    CHECK(api_http_parse(get, strlen(get), &r) == 1);
    CHECK(!strcmp(r.path, "/v1/models") && r.content_length == 0);
    const char* bad = "GET\r\n\r\n";
    CHECK(api_http_parse(bad, strlen(bad), &r) == -1);
    const char* badcl = "POST / HTTP/1.1\r\nContent-Length: nope\r\n\r\n";
    CHECK(api_http_parse(badcl, strlen(badcl), &r) == -1);
    const char* bigcl = "POST / HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n";
    CHECK(api_http_parse(bigcl, strlen(bigcl), &r) == -1);
}

static void test_json(void) {
    JNode pool[64];
    const char* doc = "{ \"a\" : [1, 2.5, -3e2, true, false, null], \"s\": \"x\\\"y\\\\\\n\\u00e9\\ud83d\\ude00\" }";
    int root = json_parse(doc, strlen(doc), pool, 64);
    CHECK(root >= 0);
    int a = json_get(pool, root, "a");
    CHECK(a >= 0 && pool[a].type == J_ARR && pool[a].count == 6);
    int c = pool[a].first;
    CHECK(pool[c].type == J_NUM && pool[c].num == 1);
    c = pool[c].next;
    CHECK(pool[c].num == 2.5);
    c = pool[c].next;
    CHECK(pool[c].num == -300);
    c = pool[c].next;
    CHECK(pool[c].type == J_BOOL && pool[c].b == 1);
    int s = json_get(pool, root, "s");
    Sb out;
    sb_init(&out);
    json_str_sb(pool, s, &out);
    CHECK(out.n == strlen("x\"y\\\n\xc3\xa9\xf0\x9f\x98\x80"));
    CHECK(out.p && !memcmp(out.p, "x\"y\\\n\xc3\xa9\xf0\x9f\x98\x80", out.n));
    sb_free(&out);

    CHECK(json_parse("{\"a\":}", 6, pool, 64) == -1);
    CHECK(json_parse("[1,2", 4, pool, 64) == -1);
    CHECK(json_parse("{} x", 4, pool, 64) == -1);
    CHECK(json_parse("[1,2]", 5, pool, 2) == -1);   // pool too small
    CHECK(json_get(pool, root, "missing") == -1);
}

static void test_chat(void) {
    JNode pool[API_MAX_NODES];
    ChatRequest req;
    const char* err = NULL;
    const char* body =
        "{\"model\":\"x\",\"messages\":[{\"role\":\"system\",\"content\":\"Be brief.\"},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hi \"},{\"type\":\"text\",\"text\":\"there\"}]}],"
        "\"temperature\":0.2,\"max_tokens\":64,\"stream\":true,\"seed\":7}";
    CHECK(api_chat_parse(body, strlen(body), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_messages == 2 && req.stream == 1 && req.max_tokens == 64 && req.has_seed && req.seed == 7);
    CHECK(req.temperature > 0.19 && req.temperature < 0.21);
    char role[16];
    Sb content;
    sb_init(&content);
    int m = pool[req.messages].first;
    CHECK(api_message(pool, m, role, sizeof(role), &content) == 0 && !strcmp(role, "system"));
    CHECK(content.n == 9 && !memcmp(content.p, "Be brief.", 9));
    sb_free(&content);
    sb_init(&content);
    m = pool[m].next;
    CHECK(api_message(pool, m, role, sizeof(role), &content) == 0 && !strcmp(role, "user"));
    CHECK(content.n == 8 && !memcmp(content.p, "Hi there", 8));
    sb_free(&content);

    const char* defaults = "{\"messages\":[{\"role\":\"user\",\"content\":\"yo\"}]}";
    CHECK(api_chat_parse(defaults, strlen(defaults), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.max_tokens == 512 && req.stream == 0 && !req.has_seed);

    const char* bad[] = {
        "[]",
        "{\"messages\":[]}",
        "{\"messages\":[1]}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"n\":2}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"temperature\":3}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"max_tokens\":0}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"stream\":\"yes\"}",
        "{\"messages\":[{\"role\":\"user\"",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        CHECK(api_chat_parse(bad[i], strlen(bad[i]), pool, API_MAX_NODES, &req, &err) == -1);
        CHECK(err != NULL);
    }
}

static void test_writers(void) {
    Sb s;
    sb_init(&s);
    api_json_completion(&s, "chatcmpl-1", "Qwen3.5-9B", 1700000000, "Hola \"mundo\"\n", 13, "stop", 5, 4);
    CHECK(s.p && strstr(s.p, "\"content\":\"Hola \\\"mundo\\\"\\n\"") != NULL);
    CHECK(strstr(s.p, "\"total_tokens\":9") != NULL);
    CHECK(strstr(s.p, "\"finish_reason\":\"stop\"") != NULL);
    sb_free(&s);

    sb_init(&s);
    api_json_chunk(&s, "c", "m", 1, "assistant", "", 0, NULL);
    CHECK(!strcmp(s.p, "{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\","
                       "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}"));
    sb_free(&s);

    sb_init(&s);
    api_json_error(&s, "bad \x01 input", "invalid_request_error", "bad_body");
    CHECK(!strcmp(s.p, "{\"error\":{\"message\":\"bad \\u0001 input\",\"type\":\"invalid_request_error\","
                       "\"code\":\"bad_body\"}}"));
    sb_free(&s);

    sb_init(&s);
    api_json_models(&s, "Qwen3.5-9B", 0);
    CHECK(strstr(s.p, "\"data\":[{\"id\":\"Qwen3.5-9B\",\"object\":\"model\"") != NULL);
    sb_free(&s);
}

static void test_tools(void) {
    JNode pool[API_MAX_NODES];
    ChatRequest req;
    const char* err = NULL;
    const char* body =
        "{"
        "\"messages\":["
        "{\"role\":\"system\",\"content\":\"Be concise.\"},"
        "{\"role\":\"user\",\"content\":\"Search for PS4 homebrew\"},"
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":["
        "{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"PS4 homebrew\\\",\\\"max_results\\\":3}\"}}"
        "]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"Result 1: Garlic LLM\"}"
        "],"
        "\"tools\":["
        "{\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"description\":\"Search the web\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"max_results\":{\"type\":\"integer\"}},\"required\":[\"query\"]}}}"
        "],"
        "\"tool_choice\":\"required\""
        "}";
    CHECK(api_chat_parse(body, strlen(body), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_tools == 1 && req.tools >= 0 && req.tool_choice == TOOL_REQUIRED);

    // System prompt with tools
    Sb sys;
    sb_init(&sys);
    api_system_tools(pool, &req, "Be concise.", &sys);
    CHECK(sys.p && strstr(sys.p, "<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"web_search\"") != NULL);
    CHECK(strstr(sys.p, "</IMPORTANT>\n\nBe concise.") != NULL);
    sb_free(&sys);

    // Assistant message with tool_calls -> Qwen3.5 XML format
    int m = pool[req.messages].first;
    m = pool[m].next;   // user
    m = pool[m].next;   // assistant
    Sb tc_out;
    sb_init(&tc_out);
    CHECK(api_assistant_tool_calls(pool, m, &tc_out) == 1);
    CHECK(tc_out.p && !strcmp(tc_out.p,
                              "<tool_call>\n<function=web_search>\n"
                              "<parameter=query>\nPS4 homebrew\n</parameter>\n"
                              "<parameter=max_results>\n3\n</parameter>\n"
                              "</function>\n</tool_call>"));
    sb_free(&tc_out);

    // Parsing Qwen3.5 XML tool call output (with string + integer parameter type coercion from schema)
    const char* reply =
        "Let me look that up.\n\n"
        "<tool_call>\n"
        "<function=web_search>\n"
        "<parameter=query>\n"
        "Garlic LLM PS4\n"
        "</parameter>\n"
        "<parameter=max_results>\n"
        "5\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";
    ApiToolCall calls[API_MAX_TOOL_CALLS];
    Sb clean;
    sb_init(&clean);
    int n_calls = api_parse_tool_calls(pool, &req, reply, strlen(reply), &clean, calls, API_MAX_TOOL_CALLS);
    CHECK(n_calls == 1);
    CHECK(clean.p && !strcmp(clean.p, "Let me look that up."));
    CHECK(!strcmp(calls[0].id, "call_1"));
    CHECK(!strcmp(calls[0].name, "web_search"));
    CHECK(calls[0].arguments.p && !strcmp(calls[0].arguments.p, "{\"query\":\"Garlic LLM PS4\",\"max_results\":5}"));

    // Non-streaming completion writer with tool_calls (content null vs non-null)
    Sb comp;
    sb_init(&comp);
    api_json_completion_tools(&comp, "chatcmpl-9", "Qwen3.5-9B", 1700000000, NULL, 0, calls, n_calls, "tool_calls", 40,
                              18);
    CHECK(comp.p && strstr(comp.p, "\"content\":null,\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\","
                                   "\"function\":{\"name\":\"web_search\",\"arguments\":"
                                   "\"{\\\"query\\\":\\\"Garlic LLM PS4\\\",\\\"max_results\\\":5}\"}}]") != NULL);
    CHECK(strstr(comp.p, "\"finish_reason\":\"tool_calls\"") != NULL);
    sb_free(&comp);

    // Streaming chunk writer for tool_calls
    Sb chunk;
    sb_init(&chunk);
    api_json_chunk_tool(&chunk, "chatcmpl-9", "Qwen3.5-9B", 1700000000, 0, &calls[0]);
    CHECK(chunk.p && strstr(chunk.p, "\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\"") != NULL);
    sb_free(&chunk);

    api_free_tool_calls(calls, n_calls);
    sb_free(&clean);

    // Parsing JSON-style <tool_call> fallback
    const char* json_reply = "<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"orbis\"}}\n</tool_call>";
    sb_init(&clean);
    n_calls = api_parse_tool_calls(pool, &req, json_reply, strlen(json_reply), &clean, calls, API_MAX_TOOL_CALLS);
    CHECK(n_calls == 1 && clean.n == 0);
    CHECK(!strcmp(calls[0].name, "web_search"));
    CHECK(calls[0].arguments.p && !strcmp(calls[0].arguments.p, "{\"query\":\"orbis\"}"));
    api_free_tool_calls(calls, n_calls);
    sb_free(&clean);

    // Named tool_choice
    const char* named =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"web_search\"}}],"
        "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"web_search\"}}}";
    CHECK(api_chat_parse(named, strlen(named), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.tool_choice == TOOL_NAMED && !strcmp(req.tool_name, "web_search"));

    // Audio modalities + audio options + WAV & base64 writers
    const char* aud_req =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],"
        "\"modalities\":[\"text\",\"audio\"],"
        "\"audio\":{\"format\":\"wav\",\"voice\":\"es\"}}";
    CHECK(api_chat_parse(aud_req, strlen(aud_req), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.want_audio == 1 && req.audio_pcm == 0 && !strcmp(req.audio_voice, "es"));

    const char* pcm_req =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],"
        "\"audio\":{\"format\":\"pcm16\",\"voice\":\"en\"}}";
    CHECK(api_chat_parse(pcm_req, strlen(pcm_req), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.want_audio == 1 && req.audio_pcm == 1 && !strcmp(req.audio_voice, "en"));

    int16_t samples[4] = {0, 1000, -1000, 32767};
    Sb wav;
    sb_init(&wav);
    api_wav_from_pcm16(&wav, samples, 4, 24000);
    CHECK(wav.n == 44 + 8 && !memcmp(wav.p, "RIFF", 4) && !memcmp(wav.p + 8, "WAVEfmt ", 8));
    Sb aud_comp;
    sb_init(&aud_comp);
    api_json_completion_full(&aud_comp, "chatcmpl-a", "Qwen3.5-9B", 1700000000, "Hola", 4, NULL, 0, "audio-1",
                             (const uint8_t*)wav.p, wav.n, "stop", 5, 2);
    CHECK(aud_comp.p && strstr(aud_comp.p, "\"audio\":{\"id\":\"audio-1\",\"data\":\"UklGR") != NULL);
    CHECK(strstr(aud_comp.p, "\"transcript\":\"Hola\"") != NULL);
    sb_free(&aud_comp);
    sb_free(&wav);

    // Invalid tool / audio requests
    const char* bad_tools[] = {
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"tools\":{}}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"tools\":[{}]}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"tool_choice\":\"required\"}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"tools\":[{\"name\":\"f\"}],\"tool_choice\":\"bogus\"}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"audio\":{\"format\":\"mp3\"}}",
    };
    for (size_t i = 0; i < sizeof(bad_tools) / sizeof(bad_tools[0]); i++) {
        CHECK(api_chat_parse(bad_tools[i], strlen(bad_tools[i]), pool, API_MAX_NODES, &req, &err) == -1);
    }
}

int main(void) {
    test_http();
    test_json();
    test_chat();
    test_writers();
    test_tools();
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
