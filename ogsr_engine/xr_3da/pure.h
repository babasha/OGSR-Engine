#pragma once

// messages
#define REG_PRIORITY_LOW 0x11111111ul
#define REG_PRIORITY_NORMAL 0x22222222ul
#define REG_PRIORITY_HIGH 0x33333333ul

#define REG_PRIORITY_INVALID 0xfffffffful

typedef void RP_FUNC(void* obj);

// Set by xrGame at startup: peek the pending Lua error text — a raw LuaJIT unwind
// (caught only as `catch(...)`, no std::exception) leaves it on the lua stack top.
// Lets engine-side guards/FATALs name the script error they intercepted.
extern ENGINE_API const char* (*g_lua_error_peek)();

// Per-member seq profiling (VK profiler): when g_seq_profile_cb is non-null,
// CRegistrator::Process times each member and reports (typeid name, ms).
// device.cpp arms it from g_seq_profile_hook around seqFrame.Process ONLY —
// EngineTOTAL grows on vistas while every sub-timer reads 0, so this names the
// seqFrame member the time actually goes to. Installed by the VK profiler.
using seq_profile_cb = void (*)(const char* typeName, float ms);
extern ENGINE_API seq_profile_cb g_seq_profile_cb;    // live only inside seqFrame.Process
extern ENGINE_API seq_profile_cb g_seq_profile_hook;  // provider (VK profiler); null = off

extern ENGINE_API RP_FUNC rp_Frame;
class ENGINE_API pureFrame
{
public:
    virtual void OnFrame(void) = 0;
};
extern ENGINE_API RP_FUNC rp_Render;
class ENGINE_API pureRender
{
public:
    virtual void OnRender(void) = 0;
};
extern ENGINE_API RP_FUNC rp_AppActivate;
class ENGINE_API pureAppActivate
{
public:
    virtual void OnAppActivate(void) = 0;
};
extern ENGINE_API RP_FUNC rp_AppDeactivate;
class ENGINE_API pureAppDeactivate
{
public:
    virtual void OnAppDeactivate(void) = 0;
};
extern ENGINE_API RP_FUNC rp_AppStart;
class ENGINE_API pureAppStart
{
public:
    virtual void OnAppStart(void) = 0;
};
extern ENGINE_API RP_FUNC rp_AppEnd;
class ENGINE_API pureAppEnd
{
public:
    virtual void OnAppEnd(void) = 0;
};
extern ENGINE_API RP_FUNC rp_DeviceReset;
class ENGINE_API pureDeviceReset
{
public:
    virtual void OnDeviceReset(void) = 0;
};
extern ENGINE_API RP_FUNC rp_ScreenResolutionChanged;
class ENGINE_API pureScreenResolutionChanged
{
public:
    virtual void OnScreenResolutionChanged(void) = 0;
};

template <class T>
class CRegistrator // the registrator itself
{
    //-----------------------------------------------------------------------------
    struct _REG_INFO
    {
        T* Object;
        int Prio;
    };

public:
    xr_vector<_REG_INFO> R;

    // constructor
    struct
    {
        u32 in_process : 1;
        u32 changed : 1;
    };

    CRegistrator()
    {
        in_process = false;
        changed = false;
    }

    constexpr void Add(T* object) { Add({object, REG_PRIORITY_NORMAL}); }
    constexpr void Add(T* object, const int priority) { Add({object, priority}); }

    void Add(_REG_INFO&& newMessage)
    {
        bool found = false;

        for (u32 i = 0; i < R.size(); i++)
        {
            if (R[i].Object == newMessage.Object)
            {
                if (R[i].Prio == newMessage.Prio)
                {
                    return; // found with same priority
                }

                R[i].Prio = newMessage.Prio;
                found = true;
            }
        }

        if (!found)
            R.emplace_back(newMessage);

        if (in_process)
            changed = true;
        else
            Resort();
    }

    void Remove(T* obj)
    {
        for (u32 i = 0; i < R.size(); i++)
        {
            if (R[i].Object == obj)
                R[i].Prio = REG_PRIORITY_INVALID;
        }

        if (in_process)
            changed = true;
        else
            Resort();
    }

    void Process(RP_FUNC* f)
    {
        if (R.empty())
            return;

        in_process = true;

        // Per-member guard: an exception (Lua error on foreign-map content) thrown by
        // one member used to unwind the WHOLE chain — every later member (physics
        // step, scheduler, weather) silently skipped EVERY frame: the world freezes
        // while rendering runs. Name the offender, skip it, keep the chain alive.
        {
            for (u32 i = 0; i < R.size(); i++)
                if (R[i].Prio != REG_PRIORITY_INVALID)
                {
                    // Optional per-member timing (see g_seq_profile_cb above). The
                    // name is taken BEFORE the call — the member may unregister
                    // itself inside it. ~µs QPC cost, only while armed.
                    const seq_profile_cb prof = g_seq_profile_cb;
                    const char* profName = prof ? typeid(*R[i].Object).name() : nullptr;
                    CTimer profT;
                    if (prof) profT.Start();
                    try
                    {
                        f(R[i].Object);
                    }
                    catch (const std::exception& e)
                    {
                        static u32 s_throws = 0;
                        const u32 n = ++s_throws;
                        if ((n & (n - 1)) == 0) // 1,2,4,8,…
                            Msg("!![CRegistrator<%s>] member #%u (%s) threw [%s: %s] — skipped (total %u)", typeid(T).name(), i, typeid(*R[i].Object).name(), typeid(e).name(),
                                e.what(), n);
                    }
                    catch (...)
                    {
                        static u32 s_throws = 0;
                        const u32 n = ++s_throws;
                        if ((n & (n - 1)) == 0) // 1,2,4,8,…
                            Msg("!![CRegistrator<%s>] member #%u (%s) threw [non-std; lua top: %s] — skipped (total %u)", typeid(T).name(), i, typeid(*R[i].Object).name(),
                                g_lua_error_peek ? g_lua_error_peek() : "<no hook>", n);
                    }
                    if (prof) prof(profName, profT.GetElapsed_sec() * 1000.f);
                }
        }

        if (changed)
            Resort();

        in_process = false;
    }

    void Resort(void)
    {
        if (!R.empty())
        {
            std::sort(std::begin(R), std::end(R), [](const auto& a, const auto& b) { return a.Prio > b.Prio; });
        }

        while (!R.empty() && R[R.size() - 1].Prio == REG_PRIORITY_INVALID)
        {
            R.pop_back();
        }

        if (R.empty())
            R.clear();

        changed = false;
    }
};
