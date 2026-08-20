#pragma once

struct _processor_info;

namespace CPU
{
XRCORE_API extern _processor_info ID;
XRCORE_API extern u64 QPC();
// Ticks per second of the counter QPC() reads. Constant for the life of the
// machine, so a tick count can be turned into a duration without asking the OS
// again -- which the load-time instrumentation does at every phase boundary.
XRCORE_API u64 QPCFreq();

inline u64 GetCLK() { return __rdtsc(); }
} // namespace CPU

XRCORE_API void _initialize_cpu();
XRCORE_API void set_current_thread_name(const char* threadName);
XRCORE_API void set_thread_name(const char* threadName, std::thread& thread);
