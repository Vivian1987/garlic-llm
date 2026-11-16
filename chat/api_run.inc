// OpenAI-compatible API jobs (api.c queues them; included in main.c, so it sees the engine and the conversation).
// A job runs on the model between the chat's turns: the request's messages are the whole conversation, so the
// conversation on screen is cleared, and the reply is written to the connection (whole, or as server-sent events).
// The reply is generated as the chat generates its own: several tokens per step with the MTP head where the app has
// it (GARLIC_SPEC), and Circle on the controller stops it. What the screen shows meanwhile is main.c's part
// (api_chat_begin / _reply / _step / _end).

static uint32_t api_ids[16384];        // the prompt of a job
static uint32_t api_snap_ids[16384];   // tokens of the saved prefix snapshot in eng.asr.ws_begin (api_snap_n in main.c)
#define API_MAX_TEXT 256000            // bytes of message text a request may carry (token count is checked against 16380)
#define API_MAX_TOOL_BYTES 3500        // max bytes kept per tool-result message (~850 tokens, so scraped pages prefill fast)
#define TOOL_CALL (IM_START + 13)      // <tool_call>
#define TOOL_CALL_END (IM_START + 14)  // </tool_call>
#define TOOL_RESP (IM_START + 21)      // <tool_response>
#define TOOL_RESP_END (IM_START + 22)  // </tool_response>

static int api_can_snap(void) {
    uint64_t need = engine_state_size(&eng);
    return eng.qwen35 && eng.asr.ws_end && (eng.asr.ws_end - eng.asr.ws_begin) >= need;
}

// Runs engine_run in chunks of 256 tokens so a client disconnect or Circle on the controller aborts a long prefill early.
static int api_run_prefill(const uint32_t* ids, int count, int pos0, float temp, int fd, int* cancelled, uint32_t* next) {
    int done = 0;
    while (done < count) {
        if (done > 0 && (api_client_gone(fd) || api_chat_step(0))) {
            *cancelled = 1;
            return 0;
        }
        int step = count - done < 256 ? count - done : 256;
        if (engine_run(&eng, ids + done, step, pos0 + done, temp, seed++, next)) return -1;
        done += step;
    }
    return 0;
}

// How many bytes at the start of p (n) are whole UTF-8 characters (the rest waits for the next token).
static size_t api_utf8_whole(const uint8_t* p, size_t n) {
    size_t i = n;
    for (int back = 1; back <= 4 && i > 0; back++) {
        i--;
        uint8_t c = p[i];
        if ((c & 0xC0) == 0x80) continue;   // a continuation byte: keep looking for its lead byte
        size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        return back < (int)need ? i : n;
    }
    return n;
}

// If p[0..n) ends with a non-empty prefix of "<tool_call>", returns the offset where that prefix starts; else n.
static size_t api_safe_tool_prefix(const char* p, size_t n) {
    static const char tag[] = "<tool_call>";
    size_t tlen = sizeof(tag) - 1;
    size_t start = n > tlen ? n - tlen : 0;
    for (size_t i = start; i < n; i++) {
        if (p[i] == '<' && !memcmp(p + i, tag, n - i)) return i;
    }
    return n;
}

// Encodes text into ids, turning <tool_call>, </tool_call>, <tool_response> and </tool_response> into their special
// token ids so the prompt matches Qwen's tokenizer.
static int api_encode_text(uint32_t* ids, int n, int max, const char* text) {
    static const struct {
        const char* tag;
        size_t len;
        uint32_t off;
    } tags[] = {
        {"<tool_call>", 11, 13},
        {"</tool_call>", 12, 14},
        {"<tool_response>", 15, 21},
        {"</tool_response>", 16, 22},
    };
    const char* p = text;
    while (*p && n < max) {
        const char* best = NULL;
        int best_k = -1;
        for (int k = 0; k < 4; k++) {
            const char* hit = strstr(p, tags[k].tag);
            if (hit && (!best || hit < best)) {
                best = hit;
                best_k = k;
            }
        }
        if (!best) return encode_into(ids, n, max, p);
        if (best > p) {
            size_t span = (size_t)(best - p);
            char* tmp = malloc(span + 1);
            if (!tmp) return -1;
            memcpy(tmp, p, span);
            tmp[span] = 0;
            n = encode_into(ids, n, max, tmp);
            free(tmp);
        }
        if (n < max) ids[n++] = IM_START + tags[best_k].off;
        p = best + tags[best_k].len;
    }
    return n;
}

// one server-sent event; -1 if the client went away
static int api_send_event(int fd, const char* json) {
    Sb s;
    sb_init(&s);
    sb_str(&s, "data: ");
    sb_str(&s, json);
    sb_str(&s, "\n\n");
    int r = s.p && !s.oom ? api_send(fd, s.p, s.n) : -1;
    sb_free(&s);
    return r;
}

// Builds the prompt: each message as <|im_start|>role\ncontent<|im_end|>\n (with Qwen3.5's tool system block,
// <tool_call> turns and grouped <tool_response> user blocks when tools are used), then the assistant's turn with the
// empty thought block (enable_thinking=false). Returns the number of ids, or -1 if it does not fit.
static int api_prompt(const JNode* pool, const ChatRequest* cr, Sb* prefill, int* head_out, int* is_tool_turn_out) {
    int n = 0;
    size_t total = 0;
    int has_tools = cr->n_tools > 0 && cr->tool_choice != TOOL_NONE;
    int last_user_idx = -1, idx = 0;
    for (int m = pool[cr->messages].first; m >= 0; m = pool[m].next, idx++) {
        Sb content;
        sb_init(&content);
        char role[16] = "user";
        if (api_message(pool, m, role, sizeof(role), &content) == 0 && !strcmp(role, "user")) {
            const char* cp = content.p ? content.p : "";
            if (strncmp(cp, "<tool_response>", 15) != 0) last_user_idx = idx;
        }
        sb_free(&content);
    }

    int m = pool[cr->messages].first;
    idx = 0;
    if (has_tools) {
        Sb user_sys, sys_full;
        sb_init(&user_sys);
        sb_init(&sys_full);
        if (m >= 0) {
            char role0[16] = "";
            if (api_message(pool, m, role0, sizeof(role0), &user_sys) == 0 &&
                (!strcmp(role0, "system") || !strcmp(role0, "developer"))) {
                m = pool[m].next;
                idx++;
            } else {
                sb_free(&user_sys);
            }
        }
        api_system_tools(pool, cr, user_sys.p, &sys_full);
        sb_free(&user_sys);
        total += sys_full.n;
        if (sys_full.oom || total > API_MAX_TEXT) {
            sb_free(&sys_full);
            return -1;
        }
        api_ids[n++] = IM_START;
        n = encode_into(api_ids, n, 16384, "system\n");
        if (sys_full.p) n = api_encode_text(api_ids, n, 16384, sys_full.p);
        api_ids[n++] = IM_END;
        n = encode_into(api_ids, n, 16384, "\n");
        sb_free(&sys_full);
        if (n < 0 || n >= 16380) return -1;
    }

    int in_tool_group = 0;
    for (; m >= 0; m = pool[m].next, idx++) {
        Sb content, tc_text;
        sb_init(&content);
        sb_init(&tc_text);
        char role[16] = "user";
        int ok = api_message(pool, m, role, sizeof(role), &content) == 0;
        if (!strcmp(role, "developer")) strcpy(role, "system");
        int ntc = ok && !strcmp(role, "assistant") ? api_assistant_tool_calls(pool, m, &tc_text) : 0;
        if (ntc < 0) ok = 0;
        if (ok) total += content.n + tc_text.n;
        if (!ok || total > API_MAX_TEXT) {
            sb_free(&content);
            sb_free(&tc_text);
            return -1;
        }
        if (strcmp(role, "system") && strcmp(role, "assistant") && strcmp(role, "user") && strcmp(role, "tool"))
            strcpy(role, "user");
        if (in_tool_group && strcmp(role, "tool") != 0) {
            api_ids[n++] = IM_END;
            n = encode_into(api_ids, n, 16384, "\n");
            in_tool_group = 0;
        }
        if (!strcmp(role, "tool")) {
            if (content.n > API_MAX_TOOL_BYTES) {
                size_t keep = api_utf8_whole((const uint8_t*)content.p, API_MAX_TOOL_BYTES);
                content.n = keep;
                content.p[keep] = 0;
                sb_str(&content, "\n...[truncated]");
            }
            if (!in_tool_group) {
                api_ids[n++] = IM_START;
                n = encode_into(api_ids, n, 16384, "user");
                in_tool_group = 1;
            }
            n = encode_into(api_ids, n, 16384, "\n");
            api_ids[n++] = TOOL_RESP;
            n = encode_into(api_ids, n, 16384, "\n");
            if (content.p) n = api_encode_text(api_ids, n, 16384, content.p);
            n = encode_into(api_ids, n, 16384, "\n");
            api_ids[n++] = TOOL_RESP_END;
        } else if (!strcmp(role, "assistant")) {
            api_ids[n++] = IM_START;
            n = encode_into(api_ids, n, 16384, "assistant\n");
            if (eng.qwen35 && idx > last_user_idx && last_user_idx >= 0) {
                api_ids[n++] = THINK;
                n = encode_into(api_ids, n, 16384, "\n\n");
                api_ids[n++] = THINK_END;
                n = encode_into(api_ids, n, 16384, "\n\n");
            }
            if (content.n > 0) n = api_encode_text(api_ids, n, 16384, content.p);
            if (tc_text.n > 0) {
                if (content.n > 0) n = encode_into(api_ids, n, 16384, "\n\n");
                n = api_encode_text(api_ids, n, 16384, tc_text.p);
            }
            api_ids[n++] = IM_END;
            n = encode_into(api_ids, n, 16384, "\n");
        } else {
            api_ids[n++] = IM_START;
            n = encode_into(api_ids, n, 16384, role);
            n = encode_into(api_ids, n, 16384, "\n");
            if (content.p) n = api_encode_text(api_ids, n, 16384, content.p);
            api_ids[n++] = IM_END;
            n = encode_into(api_ids, n, 16384, "\n");
        }
        sb_free(&content);
        sb_free(&tc_text);
        if (n < 0 || n >= 16380) return -1;
    }
    *is_tool_turn_out = in_tool_group;
    if (in_tool_group) {
        api_ids[n++] = IM_END;
        n = encode_into(api_ids, n, 16384, "\n");
    }
    api_ids[n++] = IM_START;
    n = encode_into(api_ids, n, 16384, "assistant\n");
    *head_out = n;
    if (eng.qwen35) {
        api_ids[n++] = THINK;
        n = encode_into(api_ids, n, 16384, "\n\n");
        api_ids[n++] = THINK_END;
        n = encode_into(api_ids, n, 16384, "\n\n");
    }
    if (has_tools && cr->tool_choice == TOOL_REQUIRED) {
        sb_str(prefill, "<tool_call>\n<function=");
        n = api_encode_text(api_ids, n, 16384, prefill->p);
    } else if (has_tools && cr->tool_choice == TOOL_NAMED) {
        sb_str(prefill, "<tool_call>\n<function=");
        sb_str(prefill, cr->tool_name);
        sb_str(prefill, ">\n");
        n = api_encode_text(api_ids, n, 16384, prefill->p);
    }
    return n < 16380 ? n : -1;
}

static void api_error(int fd, const char* status_line, const char* message, const char* type, const char* code) {
    Sb s;
    sb_init(&s);
    api_json_error(&s, message, type, code);
    Sb h;
    sb_init(&h);
    sb_str(&h, status_line);
    sb_str(&h, "\r\nContent-Type: application/json\r\nContent-Length: ");
    sb_int(&h, (long long)s.n);
    sb_str(&h, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (h.p && s.p && api_send(fd, h.p, h.n) == 0) api_send(fd, s.p, s.n);
    sb_free(&h);
    sb_free(&s);
}

// Synthesizes 24 kHz 16-bit mono WAV (or raw PCM16 when cr->audio_pcm) for reply_text using Chatterbox (T3 + S3Gen).
static int api_synthesize_audio(const ChatRequest* cr, const char* reply_text, size_t reply_len, int produced,
                                Sb* audio_out) {
    if (!tts_ready || !reply_text || reply_len == 0) return -1;
    api_snap_n = 0;
    double t0 = now();
#ifdef GARLIC_SPEC
    if (spec_lent) {
        spec_unload(&eng);
        spec_lent = 0;
        if (t3_reload(&eng)) {
            say("api audio: T3 reload failed (%s)", eng.error);
            return -1;
        }
    }
#endif
    uint64_t wsize = 0, w = engine_kv_lend(&eng, &wsize);
    if (!w) {
        say("api audio: cannot lend KV memory for S3Gen");
        return -1;
    }
    if (!eng.s3.loaded && s3g_load(&eng, w, wsize, eng.asr.ws_begin, eng.asr.ws_end - eng.asr.ws_begin)) {
        say("api audio: cannot load S3Gen (%s)", eng.error);
        engine_kv_return(&eng);
        return -1;
    }

    const char* forced_lang = NULL;
    static const char* known_langs[] = {"es", "en", "pt", "ru", "it", "fr", "de"};
    for (size_t i = 0; i < sizeof(known_langs) / sizeof(known_langs[0]); i++) {
        if (!strcmp(cr->audio_voice, known_langs[i])) {
            forced_lang = known_langs[i];
            break;
        }
    }
    char base_lang[4];
    snprintf(base_lang, sizeof(base_lang), "%s", forced_lang ? forced_lang : st_lang(reply_text, lang_code()));

    Sb pcm;
    sb_init(&pcm);
    int cur = 0, len = (int)reply_len;
    while (cur < len && !pcm.oom) {
        if (api_chat_step(produced)) break;
        int end = st_next_piece(reply_text, len, cur, 1);
        if (end <= cur) break;
        while (end - cur < 80) {
            int e2 = st_next_piece(reply_text, len, end, 1);
            if (e2 <= end || st_spoken_len(reply_text + cur, e2 - cur) > 200) break;
            if (!forced_lang && strcmp(st_text_lang(reply_text + cur, end - cur, base_lang),
                                       st_text_lang(reply_text + end, e2 - end, base_lang)))
                break;
            end = e2;
        }
        int start = cur;
        cur = end;
        static char words[1024];
        int wl = st_clean(reply_text + start, end - start, words, sizeof(words) - 2);
        if (!wl) continue;
        const char* lang = forced_lang ? forced_lang : st_lang(words, base_lang);
        wl = st_speak(words, wl, (int)sizeof(words) - 2, lang);
        if (!wl) continue;
        for (char* c = words; *c; c++)
            if (*c == ':' && !(c > words && c[-1] >= '0' && c[-1] <= '9' && c[1] >= '0' && c[1] <= '9')) *c = ',';

        static uint16_t toks[512];
        int nt = ttstok_encode(&tts_tok, words, lang, toks, 400);
        int max_out = (int)strlen(words) * 3 + 40;
        if (max_out > 450) max_out = 450;
        t3_voice(&eng, lang);
        s3g_voice(&eng, lang);
        if (t3_start(&eng, toks, nt, max_out, ++tts.seed, 0)) break;
        int n_speech = 0;
        for (;;) {
            int t = t3_next(&eng);
            if (t >= 0 && n_speech < 512) {
                tts.speech[n_speech++] = (uint16_t)t;
                continue;
            }
            break;
        }
        static const uint16_t sil[6] = {6324, 6405, 4137, 4218, 6081, 6082};
        int run = 0;
        for (; run < n_speech; run++) {
            uint16_t s = tts.speech[n_speech - 1 - run];
            int is = 0;
            for (int q = 0; q < 6; q++) is |= s == sil[q];
            if (!is) break;
        }
        if (run > 7) n_speech -= run - 7;
        if (n_speech >= 2 && s3g_begin(&eng, tts.speech, n_speech, tts.seed) > 0) {
            for (;;) {
                int r = s3g_step(&eng, tts_pcm, (int)(sizeof(tts_pcm) / sizeof(tts_pcm[0])));
                if (r <= 0) break;
                float gain = eng.s3.gain;
                for (int i = 0; i < r; i++) {
                    float v = tts_pcm[i] * gain;
                    if (v > 1.0f) v = 1.0f;
                    if (v < -1.0f) v = -1.0f;
                    int16_t s16 = (int16_t)(v * 32767.0f);
                    uint8_t b[2] = {(uint8_t)(s16 & 0xFF), (uint8_t)((uint16_t)s16 >> 8)};
                    sb_add(&pcm, (const char*)b, 2);
                }
            }
        }
    }
    if (!tts_keep()) eng.s3.loaded = 0;
    engine_kv_return(&eng);
    if (pcm.oom || pcm.n == 0) {
        sb_free(&pcm);
        return -1;
    }
    size_t n_samples = pcm.n / 2;
    if (cr->audio_pcm) {
        *audio_out = pcm;
    } else {
        api_wav_from_pcm16(audio_out, (const int16_t*)pcm.p, n_samples, 24000);
        sb_free(&pcm);
    }
    say("api audio: %.2f s of 24 kHz speech synthesized in %.2f s (%s)", n_samples / 24000.0, now() - t0,
        cr->audio_pcm ? "pcm16" : "wav");
    return 0;
}

static void api_job(ApiJob* job) {
    static unsigned api_count;
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    ChatRequest cr;
    const char* err = NULL;
    if (!pool || api_chat_parse(job->body, job->body_len, pool, API_MAX_NODES, &cr, &err)) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", err ? err : "Out of memory.", "invalid_request_error",
                  "invalid_body");
        return;
    }
    Sb prefill;
    sb_init(&prefill);
    int head = 0, is_tool_turn = 0;
    int n = api_prompt(pool, &cr, &prefill, &head, &is_tool_turn);
    if (n < 0) {
        sb_free(&prefill);
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "The messages are malformed or too long.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    if (n + 16 > (int)eng.ctx_limit) {
        sb_free(&prefill);
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "The messages do not fit the context window.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    if (n + cr.max_tokens + 2 > (int)eng.ctx_limit) cr.max_tokens = (int)eng.ctx_limit - n - 2;
    int has_tools = cr.n_tools > 0 && cr.tool_choice != TOOL_NONE;
    // the model answers this request: the conversation on screen starts again from nothing (a request refused
    // above leaves it as it was)
    pos = 0;
    need_close = 0;
    hist_n = hist_done = 0;
    thinking = think_turn = 0;
    sys_pre = 0;
    view.scroll = 0;
    api_chat_begin();   // (clears the conversation: a note says that a request is being answered)
    if (cr.has_seed) seed = cr.seed;
    char id[32];
    snprintf(id, sizeof(id), "chatcmpl-%u", ++api_count);
    const char* model = api_model();
    long long created = (long long)time(NULL);

    if (api_client_gone(job->fd)) {
        say("api: client disconnected before prefill started");
        sb_free(&prefill);
        free(pool);
        return;
    }

    int lcp = 0;
    while (lcp < api_snap_n && lcp < head && api_ids[lcp] == api_snap_ids[lcp]) lcp++;
    int snap_hit = api_snap_n > 0 && lcp == api_snap_n && api_can_snap();
    double t_prefill = now();
    int prefill_err = 0, prefill_cancelled = 0;
    if (snap_hit) {
        if (engine_state_copy(&eng, eng.asr.ws_begin, 0)) {
            snap_hit = 0;
            api_snap_n = 0;
        }
    }
    if (snap_hit) {
        int cached = api_snap_n;
        if (head > cached) {
            if (api_run_prefill(api_ids + cached, head - cached, cached, (float)cr.temperature, job->fd,
                                &prefill_cancelled, &next_token))
                prefill_err = 1;
            else if (!prefill_cancelled && !is_tool_turn && engine_state_copy(&eng, eng.asr.ws_begin, 1) == 0) {
                memcpy(api_snap_ids, api_ids, (size_t)head * sizeof(uint32_t));
                api_snap_n = head;
            }
        }
        if (!prefill_err && !prefill_cancelled && n > head) {
            if (api_run_prefill(api_ids + head, n - head, head, (float)cr.temperature, job->fd, &prefill_cancelled,
                                &next_token))
                prefill_err = 1;
        }
        say("api: prompt of %d tokens (%d cached, %d new, %d tools) in %.2f s, up to %d new, T %.2f%s%s", n, cached,
            n - cached, cr.n_tools, now() - t_prefill, cr.max_tokens, cr.temperature, cr.stream ? ", streamed" : "",
            prefill_cancelled ? " (cancelled)" : "");
    } else {
        api_snap_n = 0;
        engine_reset(&eng);
        if (head > 0 && head < n && api_can_snap()) {
            if (api_run_prefill(api_ids, head, 0, (float)cr.temperature, job->fd, &prefill_cancelled, &next_token))
                prefill_err = 1;
            else if (!prefill_cancelled) {
                if (engine_state_copy(&eng, eng.asr.ws_begin, 1) == 0) {
                    memcpy(api_snap_ids, api_ids, (size_t)head * sizeof(uint32_t));
                    api_snap_n = head;
                }
                if (api_run_prefill(api_ids + head, n - head, head, (float)cr.temperature, job->fd, &prefill_cancelled,
                                    &next_token))
                    prefill_err = 1;
            }
        } else {
            if (api_run_prefill(api_ids, n, 0, (float)cr.temperature, job->fd, &prefill_cancelled, &next_token))
                prefill_err = 1;
        }
        say("api: prompt of %d tokens (%d tools) in %.2f s, up to %d new, T %.2f%s%s", n, cr.n_tools, now() - t_prefill,
            cr.max_tokens, cr.temperature, cr.stream ? ", streamed" : "", prefill_cancelled ? " (cancelled)" : "");
    }
    if (prefill_err || prefill_cancelled) {
        if (prefill_err) api_snap_n = 0;
        sb_free(&prefill);
        free(pool);
        if (prefill_err) api_error(job->fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
        pos = 0;
        api_chat_end();
        return;
    }
    pos = n;

    int fd = job->fd, streamed = 0, client_gone = 0, error = 0;
    if (cr.stream) {
        const char* h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        if (api_send(fd, h, strlen(h))) client_gone = 1;
        streamed = 1;
        Sb first;
        sb_init(&first);
        api_json_chunk(&first, id, model, created, "assistant", "", 0, NULL);
        if (!client_gone && first.p && api_send_event(fd, first.p)) client_gone = 1;
        sb_free(&first);
    }

    Sb text, pend;   // the reply (all of it), and the bytes of the last tokens not yet a whole character
    sb_init(&text);
    sb_init(&pend);
    int in_tool_call = 0;
    if (prefill.n > 0) {
        sb_add(&text, prefill.p, prefill.n);
        in_tool_call = 1;
    }
    sb_free(&prefill);
    int produced = 0, stopped = 0;
    const char* finish = "length";
    // The tokens to write: the one a step drew, after those it kept from the MTP head's draft (the model has read
    // these already). Without the head a step is one token.
    uint32_t q[9];
    int nq = 1, iq = 0;
    q[0] = next_token;
    double t0 = now();
    api_chat_reply();
    while (produced < cr.max_tokens) {
        uint32_t t = q[iq++];
        if (t == IM_END || t == eng.eos2 || t == eng.eos) {
            finish = "stop";
            break;
        }
        const uint8_t* bytes = NULL;
        int nb = 0;
        if (has_tools && t == TOOL_CALL) {
            bytes = (const uint8_t*)"<tool_call>";
            nb = 11;
            in_tool_call = 1;
        } else if (has_tools && t == TOOL_CALL_END) {
            bytes = (const uint8_t*)"</tool_call>";
            nb = 12;
        } else {
            nb = tok_piece(&tok, t, &bytes);
        }
        produced++;
        if (nb > 0) {
            sb_add(&text, (const char*)bytes, nb);
            if (!in_tool_call) sb_add(&pend, (const char*)bytes, nb);
        }
        if (has_tools && !in_tool_call && pend.n) {
            const char* tc_pos = strstr(pend.p, "<tool_call>");
            if (tc_pos) {
                size_t before = (size_t)(tc_pos - pend.p);
                size_t whole = api_utf8_whole((const uint8_t*)pend.p, before);
                if (streamed && whole) {
                    Sb ev;
                    sb_init(&ev);
                    api_json_chunk(&ev, id, model, created, NULL, pend.p, whole, NULL);
                    if (ev.p && api_send_event(fd, ev.p)) client_gone = 1;
                    sb_free(&ev);
                }
                pend.n = 0;
                if (pend.p) pend.p[0] = 0;
                in_tool_call = 1;
            }
        }
        if (streamed && !in_tool_call && pend.n) {
            size_t safe = has_tools ? api_safe_tool_prefix(pend.p, pend.n) : pend.n;
            size_t whole = api_utf8_whole((const uint8_t*)pend.p, safe);
            if (whole) {
                Sb ev;
                sb_init(&ev);
                api_json_chunk(&ev, id, model, created, NULL, pend.p, whole, NULL);
                if (ev.p && api_send_event(fd, ev.p)) client_gone = 1;
                sb_free(&ev);
                memmove(pend.p, pend.p + whole, pend.n - whole);
                pend.n -= whole;
                pend.p[pend.n] = 0;
            }
        }
        if (client_gone || produced == cr.max_tokens) break;
        if (iq < nq) continue;   // (the next token is one the last step kept)
        if (api_chat_step(produced)) {   // Circle: the reply ends here, cut short as by max_tokens
            stopped = 1;
            break;
        }
        int ne = 0;   // tokens of the draft this step keeps
#ifdef GARLIC_SPEC
        if (spec_active(&eng)) {   // (never more than max_tokens: the step keeps at most what is left of them)
            if (engine_spec_step(&eng, t, pos, (float)cr.temperature, seed++, cr.max_tokens - produced + 1, q, &ne,
                                 &next_token)) {
                error = 1;
                break;
            }
        } else
#endif
        if (engine_run(&eng, &t, 1, pos, (float)cr.temperature, seed++, &next_token)) {
            error = 1;
            break;
        }
        pos += 1 + ne;
        q[ne] = next_token;
        nq = ne + 1, iq = 0;
    }

    ApiToolCall calls[API_MAX_TOOL_CALLS];
    Sb clean_text;
    sb_init(&clean_text);
    int n_calls = has_tools && text.p
                      ? api_parse_tool_calls(pool, &cr, text.p, text.n, &clean_text, calls, API_MAX_TOOL_CALLS)
                      : 0;
    if (n_calls > 0 && !stopped && !error) finish = "tool_calls";

    double dt = now() - t0;
    say("api: %d tokens in %.1f s (%.1f tokens/s), %d tool calls, %s", produced, dt, dt > 0 ? produced / dt : 0.0,
        n_calls, error ? "engine error" : stopped ? "stopped on the console" : finish);

    Sb audio_bin;
    sb_init(&audio_bin);
    char audio_id[32];
    snprintf(audio_id, sizeof(audio_id), "audio-%u", api_count);
    if (cr.want_audio && !error && !stopped && !client_gone && n_calls == 0 && text.n > 0) {
        api_synthesize_audio(&cr, text.p, text.n, produced, &audio_bin);
    }

    if (streamed) {
        if (!client_gone && !error) {
            if (n_calls == 0 && !in_tool_call && pend.n) {
                size_t whole = api_utf8_whole((const uint8_t*)pend.p, pend.n);
                if (whole) {
                    Sb ev;
                    sb_init(&ev);
                    api_json_chunk(&ev, id, model, created, NULL, pend.p, whole, NULL);
                    if (ev.p && api_send_event(fd, ev.p)) client_gone = 1;
                    sb_free(&ev);
                }
            }
            for (int i = 0; i < n_calls && !client_gone; i++) {
                Sb ev;
                sb_init(&ev);
                api_json_chunk_tool(&ev, id, model, created, i, &calls[i]);
                if (ev.p && api_send_event(fd, ev.p)) client_gone = 1;
                sb_free(&ev);
            }
            if (audio_bin.p && audio_bin.n > 0 && !client_gone) {
                Sb ev;
                sb_init(&ev);
                api_json_chunk_audio(&ev, id, model, created, audio_id, (const uint8_t*)audio_bin.p, audio_bin.n,
                                     text.p ? text.p : "", text.n);
                if (ev.p && api_send_event(fd, ev.p)) client_gone = 1;
                sb_free(&ev);
            }
            Sb last;
            sb_init(&last);
            api_json_chunk(&last, id, model, created, NULL, "", 0, finish);
            if (!client_gone && last.p) api_send_event(fd, last.p);
            sb_free(&last);
            if (!client_gone) api_send(fd, "data: [DONE]\n\n", 14);
        }
        if (error && !client_gone) {   // the headers are out: the error goes in the stream
            Sb e;
            sb_init(&e);
            api_json_error(&e, eng.error, "server_error", NULL);
            if (e.p) api_send_event(fd, e.p);
            sb_free(&e);
        }
    } else if (error) {
        api_error(fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
    } else {
        Sb body, head;
        sb_init(&body);
        sb_init(&head);
        const char* out_txt = n_calls > 0 ? clean_text.p : (text.p ? text.p : "");
        size_t out_len = n_calls > 0 ? clean_text.n : text.n;
        api_json_completion_full(&body, id, model, created, out_txt, out_len, calls, n_calls, audio_id,
                                 (const uint8_t*)audio_bin.p, audio_bin.n, finish, n, produced);
        sb_str(&head, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
        sb_int(&head, (long long)body.n);
        sb_str(&head, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        if (head.p && body.p && api_send(fd, head.p, head.n) == 0) api_send(fd, body.p, body.n);
        sb_free(&body);
        sb_free(&head);
    }
    api_free_tool_calls(calls, n_calls);
    sb_free(&audio_bin);
    sb_free(&clean_text);
    sb_free(&text);
    sb_free(&pend);
    free(pool);
    // the engine's state is this request's; the conversation on screen starts again at position 0
    pos = 0;
    need_close = 0;
    api_chat_end();
#ifdef GARLIC_SPEC
    if (!spec_lent) spec_lend();
#endif
}
