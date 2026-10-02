// A small JSON reader/writer for the engine's line protocol (objects, arrays, strings, numbers, bools, null).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bnk {

struct Json {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::map<std::string, Json> o;

    bool has(const std::string & k) const { return kind == Obj && o.count(k); }
    const Json & operator[](const std::string & k) const {
        static const Json null;
        auto it = o.find(k);
        return it == o.end() ? null : it->second;
    }
    double num(double def = 0) const { return kind == Num ? n : kind == Bool ? (b ? 1 : 0) : def; }
    std::string str(const std::string & def = "") const { return kind == Str ? s : def; }
    bool boolean(bool def = false) const { return kind == Bool ? b : kind == Num ? n != 0 : def; }

    static Json parse(const std::string & text) {
        size_t i = 0;
        Json j = parse_value(text, i);
        skip_ws(text, i);
        if (i != text.size()) throw std::runtime_error("json: trailing characters");
        return j;
    }

private:
    static void skip_ws(const std::string & t, size_t & i) {
        while (i < t.size() && (t[i] == ' ' || t[i] == '\t' || t[i] == '\n' || t[i] == '\r')) ++i;
    }
    static Json parse_value(const std::string & t, size_t & i) {
        skip_ws(t, i);
        if (i >= t.size()) throw std::runtime_error("json: unexpected end");
        Json j;
        const char c = t[i];
        if (c == '{') {
            j.kind = Obj;
            ++i;
            skip_ws(t, i);
            if (t[i] == '}') { ++i; return j; }
            while (true) {
                skip_ws(t, i);
                Json k = parse_value(t, i);
                if (k.kind != Str) throw std::runtime_error("json: key must be a string");
                skip_ws(t, i);
                if (t[i++] != ':') throw std::runtime_error("json: expected ':'");
                j.o[k.s] = parse_value(t, i);
                skip_ws(t, i);
                if (t[i] == ',') { ++i; continue; }
                if (t[i] == '}') { ++i; break; }
                throw std::runtime_error("json: expected ',' or '}'");
            }
        } else if (c == '[') {
            j.kind = Arr;
            ++i;
            skip_ws(t, i);
            if (t[i] == ']') { ++i; return j; }
            while (true) {
                j.a.push_back(parse_value(t, i));
                skip_ws(t, i);
                if (t[i] == ',') { ++i; continue; }
                if (t[i] == ']') { ++i; break; }
                throw std::runtime_error("json: expected ',' or ']'");
            }
        } else if (c == '"') {
            j.kind = Str;
            ++i;
            while (i < t.size() && t[i] != '"') {
                if (t[i] == '\\') {
                    ++i;
                    const char e = t[i++];
                    switch (e) {
                        case 'n': j.s += '\n'; break;
                        case 't': j.s += '\t'; break;
                        case 'r': j.s += '\r'; break;
                        case 'b': j.s += '\b'; break;
                        case 'f': j.s += '\f'; break;
                        case 'u': {
                            unsigned cp = std::stoul(t.substr(i, 4), nullptr, 16);
                            i += 4;
                            if (cp < 0x80) j.s += (char) cp;
                            else if (cp < 0x800) { j.s += (char) (0xC0 | (cp >> 6)); j.s += (char) (0x80 | (cp & 63)); }
                            else { j.s += (char) (0xE0 | (cp >> 12)); j.s += (char) (0x80 | ((cp >> 6) & 63)); j.s += (char) (0x80 | (cp & 63)); }
                            break;
                        }
                        default: j.s += e;
                    }
                } else {
                    j.s += t[i++];
                }
            }
            ++i;
        } else if (t.compare(i, 4, "true") == 0) {
            j.kind = Bool; j.b = true; i += 4;
        } else if (t.compare(i, 5, "false") == 0) {
            j.kind = Bool; j.b = false; i += 5;
        } else if (t.compare(i, 4, "null") == 0) {
            i += 4;
        } else {
            size_t e = i;
            while (e < t.size() && (isdigit((unsigned char) t[e]) || t[e] == '-' || t[e] == '+' || t[e] == '.' || t[e] == 'e' || t[e] == 'E')) ++e;
            j.kind = Num;
            j.n = std::stod(t.substr(i, e - i));
            i = e;
        }
        return j;
    }
};

// Builds one JSON object line.
class JsonOut {
public:
    JsonOut & kv(const std::string & k, const std::string & v) { key(k); str(v); return *this; }
    JsonOut & kv(const std::string & k, const char * v) { return kv(k, std::string(v)); }
    JsonOut & kv(const std::string & k, double v) {
        key(k);
        if (std::isfinite(v)) {
            char b[64];
            // timestamps and big counters need every digit; small readings stay short
            snprintf(b, sizeof b, std::fabs(v) >= 1e5 ? "%.15g" : "%.6g", v);
            os_ << b;
        } else {
            os_ << "null";
        }
        return *this;
    }
    JsonOut & kv(const std::string & k, int64_t v) { key(k); os_ << v; return *this; }
    JsonOut & kv(const std::string & k, int v) { key(k); os_ << v; return *this; }
    JsonOut & kv(const std::string & k, bool v) { key(k); os_ << (v ? "true" : "false"); return *this; }
    JsonOut & kv(const std::string & k, const std::vector<int32_t> & v) {
        key(k);
        os_ << '[';
        for (size_t i = 0; i < v.size(); ++i) os_ << (i ? "," : "") << v[i];
        os_ << ']';
        return *this;
    }
    JsonOut & raw(const std::string & k, const std::string & json) { key(k); os_ << json; return *this; }
    std::string done() { return "{" + os_.str() + "}"; }

private:
    void key(const std::string & k) {
        if (n_++) os_ << ',';
        str(k);
        os_ << ':';
    }
    void str(const std::string & v) {
        os_ << '"';
        for (unsigned char c : v) {
            switch (c) {
                case '"': os_ << "\\\""; break;
                case '\\': os_ << "\\\\"; break;
                case '\n': os_ << "\\n"; break;
                case '\r': os_ << "\\r"; break;
                case '\t': os_ << "\\t"; break;
                default:
                    if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); os_ << b; }
                    else os_ << c;
            }
        }
        os_ << '"';
    }
    std::ostringstream os_;
    int n_ = 0;
};

}  // namespace bnk
