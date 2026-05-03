// cs2-cv-aimbot-improved.cpp
// g++ -std=c++20 -O3 cs2-cv-aimbot-improved.cpp $(pkg-config --cflags --libs opencv4) -lonnxruntime -levdev -lpthread -o cs2-cv-aimbot-improved
// sudo ./cs2-cv-aimbot-improved

#include <linux/uinput.h>
#include <linux/prctl.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <libevdev/libevdev.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <csignal>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include <sstream>
#include <mutex>

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running = false;
}

// ---------------- Runtime Configuration -------------------------
struct Config {
    std::string model_path = "cs2head.onnx";
    int fov = 120;
    float smooth = 8.0f;
    float trigger_threshold = 0.22f;
    float conf_threshold = 0.35f;
    float nms_threshold = 0.4f;
    int max_fps = 60;
    int downscale_width = 1280;
    int downscale_height = 720;
    int input_size = 640;
    int cpu_threads = 4;
    bool target_t_default = true;
    bool auto_shoot = true;
    int reaction_delay_min = 50;
    int reaction_delay_max = 200;
    float miss_probability = 0.05f;
    int max_aim_speed = 100;
    int team_toggle_key = KEY_F7;
    int aimbot_toggle_key = KEY_F8;
    int color_threshold = 50;

    // Stealth
    float jitter_intensity = 0.8f;
    float overshoot_chance = 0.15f;
    float overshoot_amount = 0.12f;
    float speed_variance = 0.2f;
    int min_fire_interval_ms = 80;
    float aim_offset_range = 0.15f;
    float target_switch_penalty = 0.3f;
    int overlay_monitor = 1;

    // Anti-recoil
    bool recoil_enabled = true;
    float recoil_strength = 2.5f;
    int recoil_delay_ms = 60;

    // Center crop for performance
    float center_crop_ratio = 0.5f;

    // Aim modes: 0=assist (smooth), 1=lock (snap while key held)
    int aim_mode = 0;
    int aim_lock_key = KEY_LEFTALT;

    // Bunny hop
    bool bhop_enabled = false;

    // Dead zone: don't aim if target is within this many pixels of crosshair
    int dead_zone = 3;

    bool use_gpu = true;
    bool debug_overlay = false;

    static std::string pick_process_name() {
        static const char* names[] = {
            "htop", "btop", "konsole", "dolphin", "kate",
            "plasmashell", "kwin_x11", "kded5", "baloo_file",
            "xfce4-terminal", "thunar", "nm-applet", "polkitd",
            "dbus-daemon", "gvfsd", "tracker-miner", "pulseaudio",
            "pipewire", "wireplumber", "udisksd", "packagekitd",
            "gsd-color", "gsd-power", "at-spi-bus-launcher",
            "ibus-daemon", "colord", "accounts-daemon",
        };
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, (sizeof(names)/sizeof(names[0])) - 1);
        return names[dist(rng)];
    }

    static Config load(const std::string& path) {
        Config cfg;
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cout << "[!] No config.json found, using defaults" << std::endl;
            return cfg;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        auto get_string = [&](const std::string& section, const std::string& key) -> std::string {
            std::string search = "\"" + key + "\"";
            auto sect_pos = content.find("\"" + section + "\"");
            if (sect_pos == std::string::npos) return "";
            auto key_pos = content.find(search, sect_pos);
            if (key_pos == std::string::npos) return "";
            auto colon = content.find(':', key_pos);
            if (colon == std::string::npos) return "";
            auto start = content.find_first_not_of(" \t\n\r", colon + 1);
            if (start == std::string::npos) return "";
            if (content[start] == '"') {
                auto end = content.find('"', start + 1);
                if (end == std::string::npos) return "";
                return content.substr(start + 1, end - start - 1);
            }
            auto end = content.find_first_of(",}\n", start);
            return content.substr(start, end - start);
        };

        auto get_float = [&](const std::string& section, const std::string& key, float def) -> float {
            std::string val = get_string(section, key);
            if (val.empty()) return def;
            try { return std::stof(val); } catch (...) { return def; }
        };

        auto get_int = [&](const std::string& section, const std::string& key, int def) -> int {
            std::string val = get_string(section, key);
            if (val.empty()) return def;
            try { return std::stoi(val); } catch (...) { return def; }
        };

        auto get_bool = [&](const std::string& section, const std::string& key, bool def) -> bool {
            std::string val = get_string(section, key);
            if (val.empty()) return def;
            return val.find("true") != std::string::npos;
        };

        cfg.model_path = get_string("detection", "model_path");
        if (cfg.model_path.empty()) cfg.model_path = "cs2head.onnx";

        cfg.fov = get_int("aimbot", "fov", cfg.fov);
        cfg.smooth = get_float("aimbot", "smooth", cfg.smooth);
        cfg.trigger_threshold = get_float("aimbot", "trigger_threshold", cfg.trigger_threshold);
        cfg.conf_threshold = get_float("aimbot", "confidence_threshold", cfg.conf_threshold);
        cfg.nms_threshold = get_float("detection", "nms_threshold", cfg.nms_threshold);
        cfg.max_fps = get_int("aimbot", "max_fps", cfg.max_fps);
        cfg.downscale_width = get_int("aimbot", "downscale_width", cfg.downscale_width);
        cfg.downscale_height = get_int("aimbot", "downscale_height", cfg.downscale_height);
        cfg.input_size = get_int("detection", "input_size", cfg.input_size);
        cfg.cpu_threads = get_int("performance", "cpu_threads", cfg.cpu_threads);
        cfg.auto_shoot = get_bool("aimbot", "auto_shoot", cfg.auto_shoot);
        cfg.reaction_delay_min = get_int("safety", "reaction_delay_min", cfg.reaction_delay_min);
        cfg.reaction_delay_max = get_int("safety", "reaction_delay_max", cfg.reaction_delay_max);
        cfg.miss_probability = get_float("safety", "miss_probability", cfg.miss_probability);
        cfg.max_aim_speed = get_int("safety", "max_aim_speed", cfg.max_aim_speed);
        cfg.color_threshold = get_int("detection", "color_threshold", cfg.color_threshold);

        cfg.jitter_intensity = get_float("stealth", "jitter_intensity", cfg.jitter_intensity);
        cfg.overshoot_chance = get_float("stealth", "overshoot_chance", cfg.overshoot_chance);
        cfg.overshoot_amount = get_float("stealth", "overshoot_amount", cfg.overshoot_amount);
        cfg.speed_variance = get_float("stealth", "speed_variance", cfg.speed_variance);
        cfg.min_fire_interval_ms = get_int("stealth", "min_fire_interval_ms", cfg.min_fire_interval_ms);
        cfg.aim_offset_range = get_float("stealth", "aim_offset_range", cfg.aim_offset_range);
        cfg.target_switch_penalty = get_float("stealth", "target_switch_penalty", cfg.target_switch_penalty);
        cfg.overlay_monitor = get_int("display", "overlay_monitor", cfg.overlay_monitor);

        cfg.recoil_enabled = get_bool("recoil", "enabled", cfg.recoil_enabled);
        cfg.recoil_strength = get_float("recoil", "strength", cfg.recoil_strength);
        cfg.recoil_delay_ms = get_int("recoil", "delay_ms", cfg.recoil_delay_ms);
        cfg.center_crop_ratio = get_float("performance", "center_crop_ratio", cfg.center_crop_ratio);
        cfg.aim_mode = get_int("aimbot", "aim_mode", cfg.aim_mode);
        cfg.bhop_enabled = get_bool("aimbot", "bhop_enabled", cfg.bhop_enabled);
        cfg.dead_zone = get_int("aimbot", "dead_zone", cfg.dead_zone);
        cfg.use_gpu = get_bool("performance", "use_gpu", cfg.use_gpu);
        cfg.debug_overlay = get_bool("display", "debug_overlay", cfg.debug_overlay);

        std::string team = get_string("aimbot", "target_team");
        cfg.target_t_default = (team != "CT");

        // Clamp values to prevent division-by-zero and UB
        cfg.max_fps = std::max(cfg.max_fps, 1);
        cfg.fov = std::max(cfg.fov, 1);
        cfg.smooth = std::max(cfg.smooth, 0.1f);
        cfg.input_size = std::max(cfg.input_size, 32);
        cfg.cpu_threads = std::max(cfg.cpu_threads, 1);
        cfg.downscale_width = std::max(cfg.downscale_width, 320);
        cfg.downscale_height = std::max(cfg.downscale_height, 240);
        cfg.center_crop_ratio = std::clamp(cfg.center_crop_ratio, 0.1f, 1.0f);
        cfg.conf_threshold = std::clamp(cfg.conf_threshold, 0.01f, 1.0f);
        cfg.nms_threshold = std::clamp(cfg.nms_threshold, 0.01f, 1.0f);
        cfg.miss_probability = std::clamp(cfg.miss_probability, 0.0f, 1.0f);
        cfg.max_aim_speed = std::max(cfg.max_aim_speed, 1);
        cfg.dead_zone = std::max(cfg.dead_zone, 0);
        if (cfg.reaction_delay_min > cfg.reaction_delay_max)
            std::swap(cfg.reaction_delay_min, cfg.reaction_delay_max);

        std::cout << "[+] Config loaded from " << path << std::endl;
        return cfg;
    }
};

// ---------------- uinput mouse ----------------------------------
class UMouse {
    int fd = -1;
    std::mt19937 rng{std::random_device{}()};
public:
    UMouse() {
        fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
        if (fd < 0) throw std::runtime_error("Cannot open /dev/uinput (run as root or fix permissions)");

        if (ioctl(fd, UI_SET_EVBIT, EV_REL) < 0 ||
            ioctl(fd, UI_SET_EVBIT, EV_KEY) < 0 ||
            ioctl(fd, UI_SET_RELBIT, REL_X) < 0 ||
            ioctl(fd, UI_SET_RELBIT, REL_Y) < 0 ||
            ioctl(fd, UI_SET_KEYBIT, BTN_LEFT) < 0) {
            close(fd);
            throw std::runtime_error("Failed to configure uinput device");
        }

        uinput_setup usetup{};
        std::strcpy(usetup.name, "virtual-mouse");
        usetup.id.bustype = BUS_USB;
        usetup.id.vendor  = 0x1234;
        usetup.id.product = 0x5678;

        if (ioctl(fd, UI_DEV_SETUP, &usetup) < 0 ||
            ioctl(fd, UI_DEV_CREATE) < 0) {
            close(fd);
            throw std::runtime_error("Failed to create uinput device");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    bool emit(const void* data, size_t len) {
        return ::write(fd, data, len) == (ssize_t)len;
    }

    void move(int dx, int dy) {
        if (dx == 0 && dy == 0) return;

        input_event evs[3]{};
        evs[0].type = EV_REL; evs[0].code = REL_X; evs[0].value = dx;
        evs[1].type = EV_REL; evs[1].code = REL_Y; evs[1].value = dy;
        evs[2].type = EV_SYN; evs[2].code = SYN_REPORT;
        if (!emit(evs, sizeof(evs)))
            std::cerr << "[!] Mouse move write failed" << std::endl;
    }

    void click() {
        input_event ev{};
        ev.type = EV_KEY; ev.code = BTN_LEFT; ev.value = 1;
        input_event syn{};
        syn.type = EV_SYN; syn.code = SYN_REPORT;

        if (!emit(&ev, sizeof(ev)) || !emit(&syn, sizeof(syn))) return;

        std::uniform_int_distribution<int> d(8, 19);
        std::this_thread::sleep_for(std::chrono::milliseconds(d(rng)));

        ev.value = 0;
        emit(&ev, sizeof(ev));
        emit(&syn, sizeof(syn));
    }

    ~UMouse() {
        if (fd >= 0) {
            ioctl(fd, UI_DEV_DESTROY);
            close(fd);
        }
    }

    UMouse(const UMouse&) = delete;
    UMouse& operator=(const UMouse&) = delete;
};

// ---------------- Anti-Recoil System ----------------------------
class AntiRecoil {
    bool enabled;
    float strength;
    int delay_ms;
    bool firing = false;
    int fire_ticks = 0;
    std::mt19937 rng{std::random_device{}()};
    float horizontal_drift = 0.0f;

public:
    AntiRecoil(const Config& cfg)
        : enabled(cfg.recoil_enabled), strength(cfg.recoil_strength),
          delay_ms(cfg.recoil_delay_ms) {}

    void set_firing(bool f) {
        if (f && !firing) {
            fire_ticks = 0;
            std::uniform_real_distribution<float> drift(-0.4f, 0.4f);
            horizontal_drift = drift(rng);
        }
        firing = f;
        if (!f) fire_ticks = 0;
    }

    struct RecoilResult { int dy; int dx; };

    RecoilResult get_compensation() {
        if (!enabled || !firing) return {0, 0};
        fire_ticks++;
        float ramp = std::min((float)fire_ticks / 10.0f, 1.0f);
        std::normal_distribution<float> noise_y(0.0f, 0.4f);
        std::normal_distribution<float> noise_x(0.0f, 0.25f);
        int dy = (int)(strength * ramp + noise_y(rng));
        int dx = (int)(horizontal_drift * ramp + noise_x(rng));
        if (fire_ticks % 7 == 0) {
            std::uniform_real_distribution<float> pause(-0.3f, 0.1f);
            dy = (int)(dy * (1.0f + pause(rng)));
        }
        return {dy, dx};
    }
};

// ---------------- Bunny Hop Engine ------------------------------
class BhopEngine {
    int uinput_fd = -1;
    bool enabled;
    bool hw_ready = false;
    std::mt19937 rng{std::random_device{}()};

    bool init_device() {
        uinput_fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
        if (uinput_fd < 0) {
            std::cerr << "[!] Bhop: cannot open /dev/uinput" << std::endl;
            return false;
        }

        if (ioctl(uinput_fd, UI_SET_EVBIT, EV_KEY) < 0 ||
            ioctl(uinput_fd, UI_SET_KEYBIT, KEY_SPACE) < 0) {
            std::cerr << "[!] Bhop: ioctl setup failed" << std::endl;
            close(uinput_fd); uinput_fd = -1;
            return false;
        }

        uinput_setup usetup{};
        std::strcpy(usetup.name, "virtual-kbd");
        usetup.id.bustype = BUS_USB;
        usetup.id.vendor  = 0x1234;
        usetup.id.product = 0x9abc;

        if (ioctl(uinput_fd, UI_DEV_SETUP, &usetup) < 0 ||
            ioctl(uinput_fd, UI_DEV_CREATE) < 0) {
            std::cerr << "[!] Bhop: device creation failed" << std::endl;
            close(uinput_fd); uinput_fd = -1;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return true;
    }

public:
    BhopEngine(const Config& cfg) : enabled(cfg.bhop_enabled) {
        if (enabled) {
            hw_ready = init_device();
            if (!hw_ready) enabled = false;
        }
    }

    bool emit(const void* data, size_t len) {
        return ::write(uinput_fd, data, len) == (ssize_t)len;
    }

    void tap_space() {
        if (!enabled || !hw_ready) return;
        input_event ev{};
        ev.type = EV_KEY; ev.code = KEY_SPACE; ev.value = 1;
        input_event syn{}; syn.type = EV_SYN; syn.code = SYN_REPORT;

        if (!emit(&ev, sizeof(ev)) || !emit(&syn, sizeof(syn))) return;

        std::uniform_int_distribution<int> d(10, 24);
        std::this_thread::sleep_for(std::chrono::milliseconds(d(rng)));

        ev.value = 0;
        emit(&ev, sizeof(ev));
        emit(&syn, sizeof(syn));
    }

    bool is_enabled() const { return enabled; }

    void toggle() {
        enabled = !enabled;
        if (enabled && !hw_ready) {
            hw_ready = init_device();
            if (!hw_ready) {
                enabled = false;
                std::cerr << "[!] Bhop: failed to initialize on toggle" << std::endl;
            }
        }
    }

    ~BhopEngine() {
        if (uinput_fd >= 0) {
            ioctl(uinput_fd, UI_DEV_DESTROY);
            close(uinput_fd);
        }
    }

    BhopEngine(const BhopEngine&) = delete;
    BhopEngine& operator=(const BhopEngine&) = delete;
};

// ---------------- Status Overlay on Secondary Monitor -----------
class StatusOverlay {
    using Clock = std::chrono::steady_clock;
    std::string window_name = "sys-monitor";
    std::mutex mtx;
    std::string status_text;
    Clock::time_point show_until;
    std::atomic<bool> dirty{false};
    std::atomic<bool> running{true};
    std::thread render_thread;
    int mon_x, mon_y;

    void render_loop() {
        try {
            cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_GUI_NORMAL);
            cv::resizeWindow(window_name, 320, 80);
            cv::moveWindow(window_name, mon_x + 20, mon_y + 20);
            cv::setWindowProperty(window_name, cv::WND_PROP_TOPMOST, 1);
        } catch (...) {
            std::cerr << "[!] Overlay window creation failed" << std::endl;
            return;
        }

        while (running) {
            Clock::time_point deadline;
            std::string text;
            {
                std::lock_guard<std::mutex> lock(mtx);
                deadline = show_until;
                text = status_text;
            }
            auto now = Clock::now();
            if (now < deadline && dirty.load()) {
                cv::Mat img(80, 320, CV_8UC3, cv::Scalar(20, 20, 20));
                cv::putText(img, text, cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 220, 0), 1);
                cv::imshow(window_name, img);
                dirty = false;
            } else if (now >= deadline) {
                cv::Mat img(80, 320, CV_8UC3, cv::Scalar(20, 20, 20));
                cv::imshow(window_name, img);
            }
            cv::waitKey(50);
        }
        cv::destroyWindow(window_name);
    }

public:
    StatusOverlay(int overlay_monitor) {
        mon_x = overlay_monitor * 1920;
        mon_y = 0;
        render_thread = std::thread([this]{ render_loop(); });
    }

    void show(const std::string& text, int duration_ms = 3000) {
        std::lock_guard<std::mutex> lock(mtx);
        status_text = text;
        show_until = Clock::now() + std::chrono::milliseconds(duration_ms);
        dirty = true;
    }

    ~StatusOverlay() {
        running = false;
        if (render_thread.joinable()) render_thread.join();
    }

    StatusOverlay(const StatusOverlay&) = delete;
    StatusOverlay& operator=(const StatusOverlay&) = delete;
};

// ---------------- Toggle System ---------------------------------
class AdvancedToggle {
    libevdev *dev = nullptr;
    int fd = -1;
    std::atomic<bool> aimbot_active{true};
    std::atomic<bool> target_t_team{true};
    std::atomic<bool> show_debug{false};
    std::atomic<bool> panicked{false};
    std::atomic<bool> reload_config{false};
    std::atomic<int>  aim_mode{0};
    std::atomic<bool> aim_lock_held{false};
    std::atomic<bool> space_held{false};
    std::atomic<bool> bhop_toggled{false};
    std::atomic<bool> running{true};
    std::thread input_thread;
    int team_key;
    int aimbot_key;
    int aim_lock_key;
    StatusOverlay* overlay;

    void handle_key(unsigned int code, int value) {
        // Track held state for aim-lock and bhop
        if (code == (unsigned)aim_lock_key) {
            aim_lock_held = (value != 0);
            return;
        }
        if (code == KEY_SPACE) {
            space_held = (value != 0);
            return;
        }

        // Only handle press events (value==1) for toggles
        if (value != 1) return;

        if ((int)code == team_key) {
            target_t_team = !target_t_team;
            overlay->show(std::string("Target: ") + (target_t_team ? "T" : "CT"));
        } else if ((int)code == aimbot_key) {
            aimbot_active = !aimbot_active;
            overlay->show(std::string("Aimbot: ") + (aimbot_active ? "ON" : "OFF"));
        } else if (code == KEY_F5) {
            aim_mode = (aim_mode.load() + 1) % 2;
            overlay->show(std::string("Aim: ") + (aim_mode == 0 ? "ASSIST" : "LOCK"));
        } else if (code == KEY_F6) {
            bhop_toggled = true;
            overlay->show("Bhop toggled");
        } else if (code == KEY_F9) {
            show_debug = !show_debug;
            overlay->show(std::string("Debug: ") + (show_debug ? "ON" : "OFF"));
        } else if (code == KEY_F10) {
            std::string msg = std::string("Aim:") + (aimbot_active ? "ON" : "OFF")
                + " Tgt:" + (target_t_team ? "T" : "CT")
                + " Mode:" + (aim_mode == 0 ? "AST" : "LCK");
            overlay->show(msg, 5000);
        } else if (code == KEY_F11) {
            reload_config = true;
            overlay->show("Config reloading...");
        } else if (code == KEY_F12) {
            panicked = !panicked;
            if (panicked) {
                aimbot_active = false;
                show_debug = false;
                overlay->show("PANIC - ALL OFF", 1000);
            } else {
                overlay->show("Panic released");
            }
        }
    }

public:
    AdvancedToggle(const Config& cfg, StatusOverlay* ovl)
        : team_key(cfg.team_toggle_key), aimbot_key(cfg.aimbot_toggle_key),
          aim_lock_key(cfg.aim_lock_key), overlay(ovl) {
        aim_mode = cfg.aim_mode;
        target_t_team = cfg.target_t_default;
        show_debug = cfg.debug_overlay;

        bool found = false;
        for (int i = 0; i < 20 && !found; i++) {
            std::string device = "/dev/input/event" + std::to_string(i);
            int try_fd = open(device.c_str(), O_RDONLY | O_NONBLOCK);
            if (try_fd < 0) continue;

            libevdev *try_dev = nullptr;
            if (libevdev_new_from_fd(try_fd, &try_dev) == 0) {
                if (libevdev_has_event_type(try_dev, EV_KEY) &&
                    libevdev_has_event_code(try_dev, EV_KEY, KEY_F7) &&
                    libevdev_has_event_code(try_dev, EV_KEY, KEY_F8)) {
                    fd = try_fd;
                    dev = try_dev;
                    found = true;
                    const char* name = libevdev_get_name(try_dev);
                    std::cout << "[+] Found keyboard: " << device
                             << " (" << (name ? name : "unknown") << ")" << std::endl;
                    continue;
                }
                libevdev_free(try_dev);
            }
            close(try_fd);
        }

        if (!found) {
            throw std::runtime_error("No keyboard found in /dev/input/event*");
        }

        input_thread = std::thread([this]{
            input_event ev{};
            while (running) {
                int rc = libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &ev);
                if (rc == LIBEVDEV_READ_STATUS_SUCCESS) {
                    if (ev.type == EV_KEY) {
                        handle_key(ev.code, ev.value);
                    }
                } else if (rc == LIBEVDEV_READ_STATUS_SYNC) {
                    while (libevdev_next_event(dev, LIBEVDEV_READ_FLAG_SYNC, &ev)
                           == LIBEVDEV_READ_STATUS_SYNC) {}
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });

        std::cout << "[+] F5=aim-mode F6=bhop F7=team F8=aimbot F9=debug F10=status F11=reload F12=panic" << std::endl;
        std::cout << "[+] Default target: " << (target_t_team ? "T" : "CT") << std::endl;
    }

    bool is_active()           const { return aimbot_active.load() && !panicked.load(); }
    bool is_targeting_t()      const { return target_t_team.load(); }
    bool is_debug_enabled()    const { return show_debug.load(); }
    bool is_panicked()         const { return panicked.load(); }
    bool wants_config_reload() { return reload_config.exchange(false); }
    int  get_aim_mode()        const { return aim_mode.load(); }
    bool is_aim_lock_held()    const { return aim_lock_held.load(); }
    bool is_space_held()       const { return space_held.load(); }
    bool consume_bhop_toggle()       { return bhop_toggled.exchange(false); }

    ~AdvancedToggle() {
        running = false;
        if (input_thread.joinable()) input_thread.join();
        if (dev) libevdev_free(dev);
        if (fd >= 0) close(fd);
    }

    AdvancedToggle(const AdvancedToggle&) = delete;
    AdvancedToggle& operator=(const AdvancedToggle&) = delete;
};

// ---------------- Screen Capture --------------------------------
class ScreenCapture {
    cv::VideoCapture cap;
    int screen_w = 0, screen_h = 0;
    int ds_w, ds_h;

public:
    ScreenCapture(const Config& cfg) : ds_w(cfg.downscale_width), ds_h(cfg.downscale_height) {
        bool opened = false;

        if (!opened && getenv("DISPLAY")) {
            std::string pipeline =
                "ximagesrc use-damage=0 ! "
                "video/x-raw,framerate=" + std::to_string(cfg.max_fps) + "/1 ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
            opened = cap.open(pipeline, cv::CAP_GSTREAMER);
            if (opened) std::cout << "[+] Screen capture: X11 (ximagesrc)" << std::endl;
        }

        if (!opened && getenv("WAYLAND_DISPLAY")) {
            opened = cap.open(
                "pipewiresrc ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1",
                cv::CAP_GSTREAMER);
            if (opened) std::cout << "[+] Screen capture: Wayland (PipeWire)" << std::endl;
        }

        if (!opened) {
            throw std::runtime_error(
                "Screen capture failed. Install gstreamer1.0-plugins-good (X11) "
                "or pipewire-gstreamer (Wayland).");
        }

        screen_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        screen_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "[+] Screen: " << screen_w << "x" << screen_h << std::endl;
    }

    cv::Mat grab(float crop_ratio = 1.0f) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            throw std::runtime_error("Screen grab failed");
        }
        if (frame.cols > ds_w || frame.rows > ds_h) {
            cv::resize(frame, frame, cv::Size(ds_w, ds_h));
        }
        if (crop_ratio < 1.0f && crop_ratio > 0.1f) {
            int cw = (int)(frame.cols * crop_ratio);
            int ch = (int)(frame.rows * crop_ratio);
            int x = (frame.cols - cw) / 2;
            int y = (frame.rows - ch) / 2;
            frame = frame(cv::Rect(x, y, cw, ch)).clone();
        }
        return frame;
    }

    float scale_x() const {
        int d = std::min(screen_w, ds_w);
        return d > 0 ? (float)screen_w / d : 1.0f;
    }
    float scale_y() const {
        int d = std::min(screen_h, ds_h);
        return d > 0 ? (float)screen_h / d : 1.0f;
    }
    int width() const { return ds_w; }
    int height() const { return ds_h; }

    ~ScreenCapture() { cap.release(); }
};

// ---------------- Detection result ------------------------------
struct Detection {
    cv::Rect box;
    float confidence;
};

// ---------------- Head Detector ---------------------------------
class HeadDetector {
    Ort::Env env;
    Ort::Session session{nullptr};
    std::vector<Ort::AllocatedStringPtr> name_ptrs;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    cv::Size input_size;
    float conf_threshold;
    float nms_threshold;

public:
    HeadDetector(const Config& cfg)
        : env(ORT_LOGGING_LEVEL_WARNING, "cs2"),
          input_size(cfg.input_size, cfg.input_size),
          conf_threshold(cfg.conf_threshold),
          nms_threshold(cfg.nms_threshold) {

        Ort::SessionOptions opt;
        opt.SetIntraOpNumThreads(cfg.cpu_threads);
        opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.use_gpu) {
            try {
                OrtCUDAProviderOptions cuda_opts{};
                cuda_opts.device_id = 0;
                opt.AppendExecutionProvider_CUDA(cuda_opts);
                std::cout << "[+] CUDA GPU enabled" << std::endl;
            } catch (...) {
                std::cout << "[+] GPU requested but CUDA unavailable, using CPU" << std::endl;
            }
        } else {
            std::cout << "[+] Using CPU inference (GPU disabled in config)" << std::endl;
        }

        {
            std::ifstream test(cfg.model_path);
            if (!test.good())
                throw std::runtime_error("Model file not found: " + cfg.model_path);
        }
        session = Ort::Session(env, cfg.model_path.c_str(), opt);

        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name = session.GetInputNameAllocated(0, alloc);
        input_names.push_back(in_name.get());
        name_ptrs.push_back(std::move(in_name));

        auto out_name = session.GetOutputNameAllocated(0, alloc);
        output_names.push_back(out_name.get());
        name_ptrs.push_back(std::move(out_name));

        std::cout << "[+] Model loaded: " << cfg.model_path << std::endl;
    }

    std::vector<Detection> detect(const cv::Mat& img) {
        std::vector<Detection> results;

        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0/255.0, input_size, cv::Scalar(), true, false);

        std::vector<int64_t> shape = {1, 3, input_size.height, input_size.width};
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(
            mem_info, reinterpret_cast<float*>(blob.data), blob.total(),
            shape.data(), shape.size());

        auto outputs = session.Run(Ort::RunOptions{nullptr},
            input_names.data(), &tensor, 1,
            output_names.data(), output_names.size());

        float* data = outputs[0].GetTensorMutableData<float>();
        auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        int num_features = (int)out_shape[1];
        int num_dets     = (int)out_shape[2];

        float sx = (float)img.cols / input_size.width;
        float sy = (float)img.rows / input_size.height;

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;

        for (int i = 0; i < num_dets; ++i) {
            float cx = data[0 * num_dets + i];
            float cy = data[1 * num_dets + i];
            float w  = data[2 * num_dets + i];
            float h  = data[3 * num_dets + i];

            float max_score = 0;
            for (int c = 4; c < num_features; ++c) {
                max_score = std::max(max_score, data[c * num_dets + i]);
            }
            if (max_score < conf_threshold) continue;

            int x1 = std::max(0, (int)((cx - w / 2) * sx));
            int y1 = std::max(0, (int)((cy - h / 2) * sy));
            int bw = std::max(1, (int)(w * sx));
            int bh = std::max(1, (int)(h * sy));
            if (x1 + bw > img.cols) bw = img.cols - x1;
            if (y1 + bh > img.rows) bh = img.rows - y1;
            if (bw <= 0 || bh <= 0) continue;

            boxes.emplace_back(x1, y1, bw, bh);
            scores.push_back(max_score);
        }

        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);
        }

        for (int idx : indices) {
            results.push_back({boxes[idx], scores[idx]});
        }

        return results;
    }
};

// ---------------- Team Color Detection --------------------------
class TeamDetector {
public:
    enum Team { UNKNOWN, TERRORIST, COUNTER_TERRORIST };

    // Sample from the body region below the head for more reliable team colors
    static Team detect_team(const cv::Mat& frame, const cv::Rect& head_box, int sat_threshold) {
        int body_y = head_box.y + head_box.height;
        int body_h = head_box.height * 3;
        cv::Rect body_region(
            head_box.x - head_box.width / 4,
            body_y,
            head_box.width + head_box.width / 2,
            body_h
        );
        body_region &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (body_region.empty() || body_region.width < 4 || body_region.height < 4)
            return UNKNOWN;

        cv::Mat hsv;
        cv::cvtColor(frame(body_region), hsv, cv::COLOR_BGR2HSV);

        int t_pixels = 0, ct_pixels = 0, total = 0;

        for (int r = 0; r < hsv.rows; r += 2) {
            for (int c = 0; c < hsv.cols; c += 2) {
                auto px = hsv.at<cv::Vec3b>(r, c);
                float h = px[0], s = px[1], v = px[2];

                if (s < sat_threshold || v < 40) continue;
                total++;

                // T-side: reds, oranges, yellows (hue 0-25 and 155-180)
                if (h < 25 || h > 155) t_pixels++;
                // CT-side: blues (hue 95-135)
                else if (h > 95 && h < 135) ct_pixels++;
            }
        }

        if (total < 10) return UNKNOWN;

        float t_ratio = (float)t_pixels / total;
        float ct_ratio = (float)ct_pixels / total;

        if (t_ratio > 0.25f && t_ratio > ct_ratio * 1.5f) return TERRORIST;
        if (ct_ratio > 0.25f && ct_ratio > t_ratio * 1.5f) return COUNTER_TERRORIST;

        return UNKNOWN;
    }

    static const char* team_str(Team t) {
        switch (t) {
            case TERRORIST:         return "T";
            case COUNTER_TERRORIST: return "CT";
            default:                return "?";
        }
    }
};

// ---------------- Stealth Engine --------------------------------
class StealthEngine {
    std::mt19937 rng{std::random_device{}()};
    float jitter_intensity;
    float overshoot_chance;
    float overshoot_amount;
    float speed_variance;
    float aim_offset_range;
    float target_switch_penalty;
    int min_fire_interval_ms;

    using Clock = std::chrono::steady_clock;
    Clock::time_point last_fire_time;
    int prev_target_id = -1;
    float overshoot_phase = 0.0f;
    bool in_overshoot = false;

    // Per-target aim offset so we don't always aim dead center
    float offset_x = 0.0f;
    float offset_y = 0.0f;

public:
    StealthEngine(const Config& cfg)
        : jitter_intensity(cfg.jitter_intensity),
          overshoot_chance(cfg.overshoot_chance),
          overshoot_amount(cfg.overshoot_amount),
          speed_variance(cfg.speed_variance),
          aim_offset_range(cfg.aim_offset_range),
          target_switch_penalty(cfg.target_switch_penalty),
          min_fire_interval_ms(cfg.min_fire_interval_ms),
          last_fire_time(Clock::now() - std::chrono::seconds(5)) {}

    void reload(const Config& cfg) {
        jitter_intensity = cfg.jitter_intensity;
        overshoot_chance = cfg.overshoot_chance;
        overshoot_amount = cfg.overshoot_amount;
        speed_variance = cfg.speed_variance;
        aim_offset_range = cfg.aim_offset_range;
        target_switch_penalty = cfg.target_switch_penalty;
        min_fire_interval_ms = cfg.min_fire_interval_ms;
    }

    // Add micro-jitter to movement so the path isn't a perfect line
    void add_jitter(int& dx, int& dy) {
        if (jitter_intensity <= 0.0f) return;
        std::normal_distribution<float> noise(0.0f, jitter_intensity);
        dx += (int)noise(rng);
        dy += (int)noise(rng);
    }

    // Randomize aim speed slightly each frame
    float get_speed_multiplier() {
        std::uniform_real_distribution<float> dist(1.0f - speed_variance, 1.0f + speed_variance);
        return dist(rng);
    }

    // Generate per-target aim offset within the bounding box
    void new_target_offset(const cv::Rect& box) {
        std::uniform_real_distribution<float> dist(-aim_offset_range, aim_offset_range);
        offset_x = dist(rng) * box.width;
        offset_y = dist(rng) * box.height;
    }

    void apply_aim_offset(int& target_cx, int& target_cy) {
        target_cx += (int)offset_x;
        target_cy += (int)offset_y;
    }

    // Handle overshoot: sometimes move past the target then correct back
    void update_overshoot(int& dx, int& dy, float dist) {
        std::uniform_real_distribution<float> chance(0.0f, 1.0f);
        if (!in_overshoot && dist > 30.0f && chance(rng) < overshoot_chance * 0.02f) {
            in_overshoot = true;
            overshoot_phase = 0.0f;
        }
        if (in_overshoot) {
            overshoot_phase += 0.15f;
            if (overshoot_phase >= 1.0f) {
                in_overshoot = false;
            } else {
                float factor = 1.0f + overshoot_amount * std::sin(overshoot_phase * 3.14159f);
                dx = (int)(dx * factor);
                dy = (int)(dy * factor);
            }
        }
    }

    // Target stickiness: penalize switching to a different target
    float get_target_score_modifier(int target_id) {
        if (prev_target_id >= 0 && target_id != prev_target_id) {
            return 1.0f - target_switch_penalty;
        }
        return 1.0f;
    }

    void set_current_target(int id) { prev_target_id = id; }
    void clear_target() { prev_target_id = -1; in_overshoot = false; }

    bool can_fire() {
        auto now = Clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fire_time).count();
        return elapsed >= min_fire_interval_ms;
    }

    void record_fire() { last_fire_time = Clock::now(); }
};

// ---------------- Aim Smoother ----------------------------------
class AimSmoother {
    float base_smooth;
    int max_speed;

    // Cubic ease-out: fast initial movement, slow fine-tuning at the end
    static float ease_out_cubic(float t) {
        t = 1.0f - t;
        return 1.0f - t * t * t;
    }

public:
    AimSmoother(float smooth, int max_spd) : base_smooth(smooth), max_speed(max_spd) {}

    void reload(float smooth, int max_spd) { base_smooth = smooth; max_speed = max_spd; }

    struct MoveResult { int dx; int dy; };

    MoveResult compute(int raw_dx, int raw_dy, float scale_x, float scale_y) {
        float dist = std::sqrt((float)(raw_dx * raw_dx + raw_dy * raw_dy));
        if (dist < 1.0f) return {0, 0};

        // Dynamic smoothing: more smoothing when close to target
        float proximity = std::min(dist / 200.0f, 1.0f);
        float ease = ease_out_cubic(1.0f - proximity);
        float dynamic_smooth = base_smooth + ease * base_smooth * 2.0f;

        float move_x = (raw_dx * scale_x) / dynamic_smooth;
        float move_y = (raw_dy * scale_y) / dynamic_smooth;

        // Clamp to max aim speed
        float move_dist = std::sqrt(move_x * move_x + move_y * move_y);
        if (move_dist > max_speed) {
            float ratio = max_speed / move_dist;
            move_x *= ratio;
            move_y *= ratio;
        }

        // Minimum movement of 1 pixel if there's still meaningful offset
        int ix = (int)move_x;
        int iy = (int)move_y;
        if (ix == 0 && std::abs(raw_dx) > 2) ix = (raw_dx > 0) ? 1 : -1;
        if (iy == 0 && std::abs(raw_dy) > 2) iy = (raw_dy > 0) ? 1 : -1;

        return {ix, iy};
    }
};

// ---------------- Main Application ------------------------------
class CS2Aimbot {
    Config cfg;
    UMouse mouse;
    StatusOverlay overlay;
    AdvancedToggle toggle;
    ScreenCapture capture;
    HeadDetector detector;
    AimSmoother smoother;
    StealthEngine stealth;
    AntiRecoil recoil;
    BhopEngine bhop;
    std::string config_path;

    std::mt19937 rng{std::random_device{}()};

    using Clock = std::chrono::steady_clock;
    Clock::time_point fps_timer;
    int fps_count = 0;
    float avg_fps = 0;
    Clock::time_point last_target_acquired;
    bool had_target = false;

    void update_fps() {
        fps_count++;
        auto now = Clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_timer).count();
        if (ms >= 1000) {
            avg_fps = fps_count * 1000.0f / ms;
            fps_timer = now;
            fps_count = 0;
        }
    }

    int random_delay() {
        std::uniform_int_distribution<int> dist(cfg.reaction_delay_min, cfg.reaction_delay_max);
        return dist(rng);
    }

    bool should_miss() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng) < cfg.miss_probability;
    }

public:
    CS2Aimbot(Config c, const std::string& cfg_path)
        : cfg(std::move(c)),
          overlay(cfg.overlay_monitor),
          toggle(cfg, &overlay),
          capture(cfg),
          detector(cfg),
          smoother(cfg.smooth, cfg.max_aim_speed),
          stealth(cfg),
          recoil(cfg),
          bhop(cfg),
          config_path(cfg_path),
          fps_timer(Clock::now()) {
        std::string masked = Config::pick_process_name();
        prctl(PR_SET_NAME, masked.c_str(), 0, 0, 0);
        std::cout << "[+] Process masked as: " << masked << std::endl;
        std::cout << "\n[+] Initialized. F10=status, F12=panic, Ctrl+C=quit\n" << std::endl;
    }

    void hot_reload() {
        try {
            Config new_cfg = Config::load(config_path);
            cfg.smooth = new_cfg.smooth;
            cfg.fov = new_cfg.fov;
            cfg.trigger_threshold = new_cfg.trigger_threshold;
            cfg.conf_threshold = new_cfg.conf_threshold;
            cfg.auto_shoot = new_cfg.auto_shoot;
            cfg.reaction_delay_min = new_cfg.reaction_delay_min;
            cfg.reaction_delay_max = new_cfg.reaction_delay_max;
            cfg.miss_probability = new_cfg.miss_probability;
            cfg.max_aim_speed = new_cfg.max_aim_speed;
            cfg.dead_zone = new_cfg.dead_zone;
            cfg.recoil_strength = new_cfg.recoil_strength;
            cfg.center_crop_ratio = new_cfg.center_crop_ratio;
            cfg.jitter_intensity = new_cfg.jitter_intensity;
            cfg.overshoot_chance = new_cfg.overshoot_chance;
            cfg.overshoot_amount = new_cfg.overshoot_amount;
            cfg.speed_variance = new_cfg.speed_variance;
            cfg.min_fire_interval_ms = new_cfg.min_fire_interval_ms;
            cfg.aim_offset_range = new_cfg.aim_offset_range;
            cfg.target_switch_penalty = new_cfg.target_switch_penalty;
            cfg.color_threshold = new_cfg.color_threshold;
            stealth.reload(cfg);
            smoother.reload(cfg.smooth, cfg.max_aim_speed);
            overlay.show("Config reloaded OK");
        } catch (...) {
            overlay.show("Config reload FAILED");
        }
    }

    void run() {
        auto frame_duration = std::chrono::microseconds(1'000'000 / cfg.max_fps);

        while (g_running) {
            auto frame_start = Clock::now();

            try {
                // Config hot-reload
                if (toggle.wants_config_reload()) hot_reload();

                // Panic mode: everything off, minimal CPU
                if (toggle.is_panicked()) {
                    cv::destroyAllWindows();
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    continue;
                }

                if (!toggle.is_active()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    had_target = false;
                    recoil.set_firing(false);
                    continue;
                }

                if (toggle.consume_bhop_toggle()) bhop.toggle();

                // Bunny hop: auto-jump when space is held
                if (bhop.is_enabled() && toggle.is_space_held()) {
                    bhop.tap_space();
                }

                cv::Mat frame = capture.grab(cfg.center_crop_ratio);
                int cx = frame.cols / 2;
                int cy = frame.rows / 2;

                // In lock mode, only aim while aim-lock key is held
                bool should_aim = (toggle.get_aim_mode() == 0) ||
                                  (toggle.get_aim_mode() == 1 && toggle.is_aim_lock_held());

                auto detections = detector.detect(frame);

                Detection* best = nullptr;
                float best_score = 0;
                int best_idx = -1;

                for (int i = 0; i < (int)detections.size(); i++) {
                    auto& det = detections[i];
                    int det_cx = det.box.x + det.box.width / 2;
                    int det_cy = det.box.y + det.box.height / 2;
                    int dx = det_cx - cx;
                    int dy = det_cy - cy;
                    float dist = std::sqrt((float)(dx * dx + dy * dy));

                    if (dist > cfg.fov) continue;

                    auto team = TeamDetector::detect_team(frame, det.box, cfg.color_threshold);
                    bool want_t = toggle.is_targeting_t();

                    bool valid = (team == TeamDetector::UNKNOWN) ||
                                 (team == TeamDetector::TERRORIST && want_t) ||
                                 (team == TeamDetector::COUNTER_TERRORIST && !want_t);
                    if (!valid) continue;

                    float norm_dist = dist / cfg.fov;
                    float score = det.confidence * (1.0f - norm_dist * 0.7f);
                    score *= stealth.get_target_score_modifier(i);

                    if (score > best_score) {
                        best_score = score;
                        best = &det;
                        best_idx = i;
                    }

                    if (toggle.is_debug_enabled()) {
                        cv::Scalar color = (team == TeamDetector::TERRORIST) ?
                            cv::Scalar(0, 0, 255) : (team == TeamDetector::COUNTER_TERRORIST) ?
                            cv::Scalar(255, 0, 0) : cv::Scalar(128, 128, 128);
                        cv::rectangle(frame, det.box, color, 2);
                        char label[32];
                        snprintf(label, sizeof(label), "%s %.0f%%",
                            TeamDetector::team_str(team), det.confidence * 100);
                        cv::putText(frame, label,
                            cv::Point(det.box.x, det.box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                    }
                }

                if (best && should_aim) {
                    int target_cx = best->box.x + best->box.width / 2;
                    int target_cy = best->box.y + best->box.height / 2;
                    int dx = target_cx - cx;
                    int dy = target_cy - cy;
                    float dist = std::sqrt((float)(dx * dx + dy * dy));

                    // Dead zone: skip aiming if already very close
                    if (dist <= cfg.dead_zone) {
                        // Still trigger if on target
                        if (cfg.auto_shoot && stealth.can_fire() && !should_miss()) {
                            mouse.click();
                            stealth.record_fire();
                            recoil.set_firing(true);
                        }
                    } else {
                        if (!had_target) {
                            had_target = true;
                            last_target_acquired = Clock::now();
                            stealth.new_target_offset(best->box);
                            int delay = random_delay();
                            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                        }

                        stealth.set_current_target(best_idx);
                        stealth.apply_aim_offset(target_cx, target_cy);
                        dx = target_cx - cx;
                        dy = target_cy - cy;
                        dist = std::sqrt((float)(dx * dx + dy * dy));

                        stealth.update_overshoot(dx, dy, dist);

                        AimSmoother::MoveResult move;
                        if (toggle.get_aim_mode() == 1) {
                            // Lock mode: faster, more direct movement
                            move.dx = (int)(dx * capture.scale_x() / (cfg.smooth * 0.3f));
                            move.dy = (int)(dy * capture.scale_y() / (cfg.smooth * 0.3f));
                        } else {
                            move = smoother.compute(dx, dy, capture.scale_x(), capture.scale_y());
                        }

                        float speed_mult = stealth.get_speed_multiplier();
                        move.dx = (int)(move.dx * speed_mult);
                        move.dy = (int)(move.dy * speed_mult);
                        stealth.add_jitter(move.dx, move.dy);

                        auto recoil_comp = recoil.get_compensation();
                        move.dy += recoil_comp.dy;
                        move.dx += recoil_comp.dx;

                        mouse.move(move.dx, move.dy);

                        if (cfg.auto_shoot && stealth.can_fire()) {
                            bool crosshair_on_target =
                                cx >= best->box.x &&
                                cx <= best->box.x + best->box.width &&
                                cy >= best->box.y &&
                                cy <= best->box.y + best->box.height;

                            if (crosshair_on_target && !should_miss()) {
                                mouse.click();
                                stealth.record_fire();
                                recoil.set_firing(true);
                            } else {
                                recoil.set_firing(false);
                            }
                        }
                    }
                } else {
                    had_target = false;
                    stealth.clear_target();
                    recoil.set_firing(false);
                }

                if (toggle.is_debug_enabled()) {
                    cv::circle(frame, cv::Point(cx, cy), cfg.fov, cv::Scalar(0, 255, 0), 1);
                    if (cfg.dead_zone > 0) {
                        cv::circle(frame, cv::Point(cx, cy), cfg.dead_zone,
                                   cv::Scalar(0, 255, 255), 1);
                    }
                    cv::line(frame, cv::Point(cx - 10, cy), cv::Point(cx + 10, cy),
                             cv::Scalar(0, 255, 0), 1);
                    cv::line(frame, cv::Point(cx, cy - 10), cv::Point(cx, cy + 10),
                             cv::Scalar(0, 255, 0), 1);

                    char buf[96];
                    snprintf(buf, sizeof(buf), "Tgt:%s Mode:%s FPS:%.0f Dets:%d",
                        toggle.is_targeting_t() ? "T" : "CT",
                        toggle.get_aim_mode() == 0 ? "AST" : "LCK",
                        avg_fps, (int)detections.size());
                    cv::putText(frame, buf, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1);

                    cv::imshow("Debug", frame);
                    cv::waitKey(1);
                }

                update_fps();

                auto elapsed = Clock::now() - frame_start;
                auto remaining = frame_duration - elapsed;
                if (remaining > std::chrono::microseconds(500)) {
                    std::this_thread::sleep_for(remaining);
                }

            } catch (const std::exception& e) {
                std::cerr << "[!] Error: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    ~CS2Aimbot() {
        cv::destroyAllWindows();
    }
};

// ---------------- Main Function ---------------------------------
int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        std::cout << R"(
   ____ ____  ____     _    _           _           _
  / ___|  _ \|___ \   / \  (_)_ __ ___ | |__   ___ | |_
 | |   | |_) | __) | / _ \ | | '_ ` _ \| '_ \ / _ \| __|
 | |___|  _ < / __/ / ___ \| | | | | | | |_) | (_) | |_
  \____|_| \_\_____/_/   \_\_|_| |_| |_|_.__/ \___/ \__|
  Computer Vision Aimbot v3.0
)" << std::endl;

        std::string config_path = "config.json";
        if (argc > 1) config_path = argv[1];

        Config cfg = Config::load(config_path);
        CS2Aimbot aimbot(std::move(cfg), config_path);
        aimbot.run();

    } catch (const std::exception& e) {
        std::cerr << "[!] Fatal: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
