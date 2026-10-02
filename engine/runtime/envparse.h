// Strict environment-variable parsing (host only, no CUDA): unset = the default; set = the WHOLE string must be one of
// the allowed decimal values, else the caller dies before any work. (atoi fails open: "", "x" -> 0, "1junk" -> 1, "2.5" -> 2.)
#pragma once
#include <cstdlib>
#include <cstring>
#include <string>

// value of `name` if it is unset (-> dflt) or exactly one of "0".."max" with no sign, space, leading zero or suffix;
// otherwise -1 and *err = a message naming the variable and the bad value
inline int env_choice(const char* name, int max, int dflt, std::string* err)
{
    const char* v = std::getenv(name);
    if (!v) return dflt;
    if (std::strlen(v) == 1 && v[0] >= '0' && v[0] - '0' <= max) return v[0] - '0';
    if (err) *err = std::string(name) + "='" + v + "' is not one of 0.." + std::to_string(max) + " (unset = " + std::to_string(dflt) + ")";
    return -1;
}
