// Copyright (C) 2020 ReSpace AB
// Licensed under the MIT License
//
// Simple logging system for use in DTCC C++ code modeled after the standard
// standard Python logging module with the following configuration:
//
// import logging
// format = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
// logging.basicConfig(format=format)
// logging.addLevelName(25, 'PROGRESS')

#ifndef LOGGING_H
#define LOGGING_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <iso646.h>

namespace
{
    // Log levels
    enum LogLevel
    {
        DEBUG = 10,
        INFO = 20,
        WARNING = 30,
        ERROR = 40,
        PROGRESS = 25
    };

    // Global log level
    LogLevel __log_level__ = INFO;

    // Interface for printable objects
    class Printable
    {
    public:
        // Convert to string (pretty-print)
        virtual std::string __str__() const = 0;

        // Conversion operator
        operator std::string() const { return __str__(); }
    };

    // Return current time
    static inline std::string current_time()
    {
        // Stackoverflow: get-current-time-in-milliseconds-or-hhmmssmmm-format
        using namespace std::chrono;
        auto now = system_clock::now();
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
        auto timer = system_clock::to_time_t(now);
        std::tm bt = *std::localtime(&timer);
        std::ostringstream oss;
        oss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    // Format message
    static inline std::string __format__(LogLevel log_level, const std::string &message)
    {
        // Set component (hard-coded for now)
        std::string component{"[dtcc-embree]"};

        // Format level
        std::string level{};
        switch (log_level)
        {
        case DEBUG:
            level = "[DEBUG]";
            break;
        case INFO:
            level = "[INFO]";
            break;
        case WARNING:
            level = "[WARNING]";
            break;
        case ERROR:
            level = "[ERROR]";
            break;
        case PROGRESS:
            level = "[PROGRESS]";
            break;
        default:
            level = "[UNKNOWN]";
        };

        return current_time() + " " + component + " " + level + " " + message;
    }

    // print message to stdout
    static inline void __print__(const std::string &message)
    {
        std::cout << message << std::endl;
    }

    //--- Public interface of logging system ---

    // Set log level
    static inline void set_log_level(LogLevel log_level) { __log_level__ = log_level; }

    // print message at given log level
    static inline void log(LogLevel log_level, const std::string &message = "")
    {
        // Skip if below log level threshold
        if (log_level < __log_level__)
            return;

        // Format and print
        std::string formatted_message = __format__(log_level, message);
        __print__(formatted_message);
    }

    // print debug message
    static inline void debug(const std::string &message)
    {
        log(DEBUG, message);
    }

    // print information message (string)
    static inline void info(const std::string &message = "")
    {
        log(INFO, message);
    }

    // print warning message
    static inline void warning(const std::string &message)
    {
        log(WARNING, message);
    }

    // print error message and throw exception
    static inline void error(const std::string &message)
    {
        log(ERROR, message);
        throw std::runtime_error(message);
    }

    // report progress (a number between 0 and 1)
    static inline void progress(double x)
    {
        x = std::max(0.0, std::min(1.0, x));
        std::ostringstream ss{};
        ss << std::setprecision(2) << std::fixed << 100.0 * x << "%";
        log(PROGRESS, ss.str());
    }

    //--- Utility functions for string conversion ---

    // Convert printable object to string
    static inline std::string str(const Printable &x) { return x.__str__(); }

    // Convert const char* to string
    static inline std::string str(const char *x)
    {
        std::string s(x);
        return s;
    }

    // Convert unsigned integer to string
    static inline std::string str(int x) { return std::to_string(x); }

    // Convert unsigned integer to string
    static inline std::string str(size_t x) { return std::to_string(x); }

    // Convert unsigned integer to string
    // std::string str(unsigned int x) {return std::to_string(x); }

    // Convert double to string
    static inline std::string str(double x, std::streamsize precision = 6)
    {
        std::ostringstream ss{};
        ss << std::setprecision(precision) << std::defaultfloat << x;
        return ss.str();
    }
}

#endif // LOGGING_H