#include "stdafx.h"

#include "log.h"

#include <fstream> //для std::ofstream
#include <iomanip> //для std::strftime
#include <array> //для std::array
#include <iostream>
#include <atomic>
#include <cstdlib>          // getenv — XROS_LOG_FLUSH

static LogCallback LogCB = nullptr;
xr_vector<std::string> LogFile;
static std::ofstream logstream;
string_path logFName{};

// What a level load spends writing its own log: 14926 lines for one pripyat_full
// load, each of them an ofstream write followed by a flush -- i.e. a syscall per
// line, taken on whatever thread called Msg, and never counted by any phase timer
// because it is spread across all of them. Measured here so the argument about it
// is about a number: 6859 lines cost 60 ms of the render phases, 23 of them once
// the flush stopped being per-line. XROS_LOG_FLUSH=1 puts the old behaviour back.
namespace {
std::atomic<u64> g_logLines{0}, g_logClk{0}, g_logWriteClk{0};
u32  g_logSinceFlush = 0;
u64  g_logLastFlushQpc = 0;
constexpr u32 kLogFlushRun = 64;

bool LogFlushEachLine()
{
    static const bool s_each = [] {
        if (const char* e = std::getenv("XROS_LOG_FLUSH")) return atoi(e) != 0;
        return false;
    }();
    return s_each;
}
}   // namespace

void LogProfGet(u64& lines, u64& clkTotal, u64& clkWrite)
{
    lines    = g_logLines.load(std::memory_order_relaxed);
    clkTotal = g_logClk.load(std::memory_order_relaxed);
    clkWrite = g_logWriteClk.load(std::memory_order_relaxed);
}

void LogFlushNow()
{
    if (logstream.is_open()) logstream.flush();
    g_logSinceFlush = 0;
}
static void AddOne(std::string& split, bool first_line)
{
    static std::recursive_mutex logCS;
    std::scoped_lock lock(logCS);

    const u64 clkEnter = CPU::GetCLK();
    const bool isError = !split.empty() && split.front() == '!';   // before insert_time prefixes it
    g_logLines.fetch_add(1, std::memory_order_relaxed);

    if (IsDebuggerPresent())
    { //Вывод в отладчик студии
        OutputDebugString(split.c_str());
        OutputDebugString("\n");
    }

    if (LogCB)
        LogCB(split.c_str()); //Вывод в логкаллбек

    auto insert_time = [&] {
        if (first_line)
        {
            string64 buf, curTime;
            using namespace std::chrono;
            const auto now = system_clock::now();
            const auto time = system_clock::to_time_t(now);
            const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) - duration_cast<seconds>(now.time_since_epoch());
            std::strftime(buf, sizeof(buf), "%d.%m.%y %H:%M:%S", std::localtime(&time));
            sprintf_s(curTime, "\n[%s.%03lld] [%u] ", buf, ms.count(), _Thrd_id());
            split = curTime + split;
        }
        else
        {
            split = "\n" + split;
        }
    };

    if (!logstream.is_open())
        insert_time();

    static std::string last_str;
    static int items_count;

    //LogFile.push_back(split); //Вывод в консоль

    if (last_str == split)
    {
        std::string tmp = split;

        if (items_count == 0)
            items_count = 2;
        else
            items_count++;

        tmp += " [";
        tmp += std::to_string(items_count);
        tmp += "]";

        LogFile.erase(LogFile.end() - 1);
        LogFile.push_back(tmp);
    }
    else
    {
        // DUMP_PHASE;
        LogFile.push_back(split);
        last_str = split;
        items_count = 0;
    }


    if (logstream.is_open())
    {
        insert_time();

        //Вывод в лог-файл
        const u64 clkWrite0 = CPU::GetCLK();
        logstream << split;
        // The flush is what makes the log survive a hard kill (quick_exit in
        // xrDebug::backend runs no destructors -- it calls LogFlushNow instead), and
        // it costs a syscall per line: measured 40 ms of the render phases alone for
        // 6859 lines, ~100 ms over a whole level load. So it is bounded in TIME
        // instead of taken every line: errors flush immediately, everything else at
        // most 2 ms behind. XROS_LOG_FLUSH=1 restores the line-by-line behaviour.
        const u64 now = CPU::QPC();
        if (LogFlushEachLine() || isError || ++g_logSinceFlush >= kLogFlushRun
            || (now - g_logLastFlushQpc) * 500 > CPU::QPCFreq()) {
            logstream.flush();
            g_logSinceFlush   = 0;
            g_logLastFlushQpc = now;
        }
        g_logWriteClk.fetch_add(CPU::GetCLK() - clkWrite0, std::memory_order_relaxed);
    }
    g_logClk.fetch_add(CPU::GetCLK() - clkEnter, std::memory_order_relaxed);
}

void Log(const std::string& str)
{
    if (str.empty())
        return;

    bool not_first_line = false;

    constexpr std::array<char, 5> color_codes{'-', '~', '!', '*', '#'}; //Зелёный, Жёлтый, Красный, Серый, Бирюзовый
    const char& color_s = str.front();
    const bool have_color = std::find(color_codes.begin(), color_codes.end(), color_s) != color_codes.end(); //Ищем в начале строки цветовой код

    xr_vector<std::string> substrs;
    size_t beg{};
    for (size_t end{}; (end = str.find("\n", end)) != std::string::npos; ++end) // Разбиваем текст по "\n"
    {
        substrs.push_back(str.substr(beg, end - beg));
        beg = end + 1;
    }
    substrs.push_back(str.substr(beg));

    for (auto& str : substrs)
    {
        if (not_first_line && have_color)
        {
            str = ' ' + str;
            str = color_s + str; //Если надо, перед каждой строкой вставляем спец-символ цвета, чтобы в консоли цветными были все строки текста, а не только первая.
        }
        AddOne(str, !not_first_line);
        not_first_line = true;
    }
}

void Log(const char* s) { Log(std::string{s}); }

void __cdecl Msg(const char* format, ...)
{
    string4096 strBuf;
    va_list args;
    va_start(args, format);
    int buf_len = std::vsnprintf(strBuf, sizeof(strBuf), format, args);
    va_end(args);
    if (buf_len > 0)
        Log(strBuf);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void Log(const char* msg, const Fvector& dop)
{
    char buf[1024];
    std::snprintf(buf, sizeof(buf), "%s (%f,%f,%f)", msg, dop.x, dop.y, dop.z);
    Log(buf);
}

void Log(const char* msg, const Fmatrix& dop)
{
    char buf[1024];
    std::snprintf(buf, sizeof(buf), "%s:\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n", msg, dop.i.x, dop.i.y, dop.i.z, dop._14_, dop.j.x, dop.j.y, dop.j.z, dop._24_,
                  dop.k.x, dop.k.y, dop.k.z, dop._34_, dop.c.x, dop.c.y, dop.c.z, dop._44_);
    Log(buf);
}
/////////////////////////////////////////////////////////////////////////////////////////////

void SetLogCB(LogCallback cb) { LogCB = cb; }

void CreateLog(BOOL nl)
{
    if (!nl)
    {
        std::cout.rdbuf(logstream.rdbuf()); // redirect std::cout to log

        if (!strstr(Core.Params, "-no_unique_logs"))
        {
            string32 TimeBuf;
            using namespace std::chrono;
            const auto now = system_clock::now();
            const auto time = system_clock::to_time_t(now);
            std::strftime(TimeBuf, sizeof(TimeBuf), "%d-%m-%y_%H-%M-%S", std::localtime(&time));

            xr_strconcat(logFName, Core.ApplicationName, "_", Core.UserName, "_", TimeBuf, ".log");
        }
        else
        {
            xr_strconcat(logFName, Core.ApplicationName, "_", Core.UserName, ".log");
        }

        try
        {
            if (FS.path_exist("$logs$"))
            {
                FS.update_path(logFName, "$logs$", logFName);
            }
            else
            { //Для компрессора
                string_path temp;
                strcpy_s(temp, logFName);
                xr_strconcat(logFName, "logs\\", temp);
            }

            //logstream.imbue(std::locale(""));
            VerifyPath(logFName);
            logstream.open(logFName);
        }
        catch (...)
        {
            Debug.do_exit("Can't create log file!");
        }

        for (const auto& str : LogFile)
            logstream << str;

        logstream.flush();
    }

    LogFile.reserve(5000);
}
