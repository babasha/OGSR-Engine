#pragma once

#define VPUSH(a) a.x, a.y, a.z

void XRCORE_API __cdecl Msg(const char* format, ...);
void XRCORE_API Log(const std::string& msg);
void XRCORE_API Log(const char* msg);
void XRCORE_API Log(const char* msg, const Fvector& dop);
void XRCORE_API Log(const char* msg, const Fmatrix& dop);

using LogCallback = std::function<void(const char*)>;
void XRCORE_API SetLogCB(LogCallback cb);

// Log accounting: lines written since process start and the rdtsc they cost (total,
// and the write+flush part of it). One level load writes ~15000 lines, each one an
// ofstream write plus a flush, and no phase timer has ever counted them.
void XRCORE_API LogProfGet(u64& lines, u64& clkTotal, u64& clkWrite);

// Push the stream buffer out now. The per-line flush can be switched off with
// XROS_LOG_FLUSH=0, in which case a clean exit path should call this.
void XRCORE_API LogFlushNow();
void CreateLog(BOOL no_log = FALSE);

extern XRCORE_API xr_vector<std::string> LogFile;
extern XRCORE_API string_path logFName;
