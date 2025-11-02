// cs2-cv-aimbot-improved.cpp
// Enhanced version with team toggle, improved performance, and better configuration
// g++ -std=c++20 -O3 cs2-cv-aimbot-improved.cpp -lonnxruntime -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -levdev -o cs2-cv-aimbot-improved
// sudo ./cs2-cv-aimbot-improved

#include <linux/uinput.h>
#include <libevdev/libevdev.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>
#include <fstream>
#include <map>

// Configuration constants - easily adjustable
static const char* MODEL = "cs2head.onnx";   // 1-class model: 0=head
static const int   FOV     = 120;              // pixels radius
static const float SMOOTH   = 8.0f;
static const float TRIGGER  = 0.22f;           // crosshair overlap 22% -> shoot

// Team targeting configuration
static const int   TEAM_TOGGLE_KEY = KEY_F7;   // Toggle between T/CT targeting
static const int   AIMBOT_TOGGLE_KEY = KEY_F8; // Toggle aimbot on/off
static const bool  TARGET_T_BY_DEFAULT = true; // Default team to target (T=true, CT=false)

// Performance settings
static const int   MAX_FPS = 60;
static const int   DOWNSCALE_WIDTH = 1280;     // Downscale for faster processing
static const int   DOWNSCALE_HEIGHT = 720;

// ---------------- uinput mouse ----------------------------------
class UMouse {
    int fd;
public:
    UMouse() {
        fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
        if (fd < 0) throw std::runtime_error("uinput");
        
        // Enable relative mouse movement and button events
        ui_set_evbit(fd, EV_REL);
        ui_set_evbit(fd, EV_KEY);
        ui_set_relbit(fd, REL_X);
        ui_set_relbit(fd, REL_Y);
        ui_set_keybit(fd, BTN_LEFT);
        
        uinput_setup usetup{};
        std::strcpy(usetup.name, "cs2-aimbot");
        usetup.id.bustype = BUS_USB;
        usetup.id.vendor = 0x1234;
        usetup.id.product = 0x5678;
        
        ioctl(fd, UI_DEV_SETUP, &usetup);
        ioctl(fd, UI_DEV_CREATE);
        
        // Small delay to ensure device is created
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    void move(int dx, int dy) {
        if (dx == 0 && dy == 0) return;
        
        input_event ev[2] = {};
        ev[0] = {.type = EV_REL, .code = REL_X, .value = dx};
        ev[1] = {.type = EV_REL, .code = REL_Y, .value = dy};
        write(fd, ev, sizeof(ev));
        
        input_event sync = {.type = EV_SYN, .code = SYN_REPORT};
        write(fd, &sync, sizeof(sync));
    }
    
    void click() {
        input_event down = {.type = EV_KEY, .code = BTN_LEFT, .value = 1};
        input_event up   = {.type = EV_KEY, .code = BTN_LEFT, .value = 0};
        input_event sync = {.type = EV_SYN, .code = SYN_REPORT};
        
        write(fd, &down, sizeof(down));
        write(fd, &sync, sizeof(sync));
        write(fd, &up,   sizeof(up));
        write(fd, &sync, sizeof(sync));
    }
    
    ~UMouse() { 
        ioctl(fd, UI_DEV_DESTROY); 
        close(fd); 
    }
};

// ---------------- Advanced Toggle System ------------------------
class AdvancedToggle {
    libevdev *dev = nullptr;
    int fd = -1;
    std::atomic<bool> aimbot_active{true};
    std::atomic<bool> target_t_team{TARGET_T_BY_DEFAULT};
    std::atomic<bool> show_debug{false};
    std::thread input_thread;
    std::map<int, std::string> key_names;
    
    void setup_key_names() {
        key_names[KEY_F1] = "F1";
        key_names[KEY_F2] = "F2";
        key_names[KEY_F3] = "F3";
        key_names[KEY_F4] = "F4";
        key_names[KEY_F5] = "F5";
        key_names[KEY_F6] = "F6";
        key_names[KEY_F7] = "F7";
        key_names[KEY_F8] = "F8";
        key_names[KEY_F9] = "F9";
        key_names[KEY_F10] = "F10";
        key_names[KEY_F11] = "F11";
        key_names[KEY_F12] = "F12";
    }
    
public:
    AdvancedToggle() {
        setup_key_names();
        
        // Try multiple event devices to find keyboard
        bool found = false;
        for (int i = 0; i < 10 && !found; i++) {
            std::string device = "/dev/input/event" + std::to_string(i);
            fd = open(device.c_str(), O_RDONLY | O_NONBLOCK);
            if (fd >= 0) {
                if (libevdev_new_from_fd(fd, &dev) == 0) {
                    const char* name = libevdev_get_name(dev);
                    if (name && std::string(name).find("keyboard") != std::string::npos) {
                        found = true;
                        std::cout << "Found keyboard at " << device << " (" << name << ")" << std::endl;
                    } else {
                        libevdev_free(dev);
                        dev = nullptr;
                        close(fd);
                        fd = -1;
                    }
                }
            }
        }
        
        if (!found) {
            std::cerr << "Warning: No keyboard found, using default /dev/input/event3" << std::endl;
            fd = open("/dev/input/event3", O_RDONLY | O_NONBLOCK);
            if (fd < 0) throw std::runtime_error("keyboard");
            libevdev_new_from_fd(fd, &dev);
        }
        
        input_thread = std::thread([this]{
            input_event ev;
            while (true) {
                int status = libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &ev);
                if (status == 0 || libevdev_next_event(dev, LIBEVDEV_READ_FLAG_SYNC, &ev) == 0) {
                    if (ev.type == EV_KEY && ev.value == 1) { // Key press
                        switch (ev.code) {
                            case TEAM_TOGGLE_KEY:
                                target_t_team = !target_t_team;
                                std::cout << "\n[!] Targeting team: " 
                                         << (target_t_team ? "TERRORISTS" : "COUNTER-TERRORISTS") 
                                         << " (F7 to toggle)" << std::endl;
                                break;
                                
                            case AIMBOT_TOGGLE_KEY:
                                aimbot_active = !aimbot_active;
                                std::cout << "\n[!] Aimbot " 
                                         << (aimbot_active ? "ENABLED" : "DISABLED") 
                                         << " (F8 to toggle)" << std::endl;
                                break;
                                
                            case KEY_F9:
                                show_debug = !show_debug;
                                std::cout << "\n[!] Debug overlay " 
                                         << (show_debug ? "ENABLED" : "DISABLED") 
                                         << " (F9 to toggle)" << std::endl;
                                break;
                                
                            case KEY_F10:
                                std::cout << "\n[!] Status - Aimbot: " << (aimbot_active ? "ON" : "OFF")
                                         << " | Target: " << (target_t_team ? "T" : "CT")
                                         << " | Debug: " << (show_debug ? "ON" : "OFF") << std::endl;
                                break;
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
        input_thread.detach();
        
        std::cout << "[!] Controls:" << std::endl;
        std::cout << "    F7 - Toggle target team (T/CT)" << std::endl;
        std::cout << "    F8 - Toggle aimbot on/off" << std::endl;
        std::cout << "    F9 - Toggle debug overlay" << std::endl;
        std::cout << "    F10 - Show status" << std::endl;
        std::cout << "    Default target: " << (TARGET_T_BY_DEFAULT ? "TERRORISTS" : "COUNTER-TERRORISTS") << std::endl;
    }
    
    bool is_active() const { return aimbot_active.load(); }
    bool is_targeting_t() const { return target_t_team.load(); }
    bool is_debug_enabled() const { return show_debug.load(); }
    
    ~AdvancedToggle() { 
        if (dev) libevdev_free(dev); 
        if (fd >= 0) close(fd); 
    }
};

// ---------------- Enhanced Screen Capture -----------------------
class ScreenCapture {
    cv::VideoCapture cap;
    int width, height;
    bool initialized;
    
public:
    ScreenCapture() : initialized(false) {
        // Try multiple capture methods
        std::vector<int> backends = {
            cv::CAP_V4L2,
            cv::CAP_GSTREAMER,
            cv::CAP_FFMPEG
        };
        
        for (int backend : backends) {
            cap.open(0, backend);
            if (cap.isOpened()) {
                initialized = true;
                std::cout << "[+] Screen capture initialized with backend: " << backend << std::endl;
                break;
            }
        }
        
        if (!initialized) {
            cap.open(0); // Fallback to default
            if (!cap.isOpened()) {
                throw std::runtime_error("Failed to initialize screen capture");
            }
            std::cout << "[+] Screen capture initialized with default backend" << std::endl;
        }
        
        // Set capture properties for better performance
        cap.set(cv::CAP_PROP_FRAME_WIDTH, DOWNSCALE_WIDTH);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, DOWNSCALE_HEIGHT);
        cap.set(cv::CAP_PROP_FPS, MAX_FPS);
        
        width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        std::cout << "[+] Capture resolution: " << width << "x" << height << std::endl;
    }
    
    cv::Mat grab() {
        cv::Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            throw std::runtime_error("Screen grab failed - empty frame");
        }
        
        return frame;
    }
    
    cv::Size get_size() const { return cv::Size(width, height); }
    
    ~ScreenCapture() {
        if (cap.isOpened()) {
            cap.release();
        }
    }
};

// ---------------- Enhanced Head Detector ------------------------
class HeadDetector {
    Ort::Session session;
    Ort::Env env;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    cv::Size input_size;
    
public:
    HeadDetector() : env(ORT_LOGGING_LEVEL_WARNING, "cs2"), input_size(640, 640) {
        try {
            Ort::SessionOptions opt;
            
            // Try to use GPU if available
            try {
                opt.AppendExecutionProvider_CUDA();
                std::cout << "[+] Using CUDA GPU for inference" << std::endl;
            } catch (...) {
                std::cout << "[!] CUDA not available, using CPU" << std::endl;
            }
            
            session = Ort::Session(env, MODEL, opt);
            
            // Get input/output names
            Ort::AllocatorWithDefaultOptions allocator;
            input_names.push_back(session.GetInputName(0, allocator));
            output_names.push_back(session.GetOutputName(0, allocator));
            
            std::cout << "[+] Model loaded: " << MODEL << std::endl;
            std::cout << "[+] Input name: " << input_names[0] << std::endl;
            std::cout << "[+] Output name: " << output_names[0] << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "[!] Failed to load model: " << e.what() << std::endl;
            throw;
        }
    }
    
    std::vector<cv::Rect> detect(cv::Mat& img) {
        std::vector<cv::Rect> detections;
        
        try {
            // Preprocess image
            cv::Mat resized, blob;
            cv::resize(img, resized, input_size);
            
            // Create blob (normalize to [0,1])
            cv::dnn::blobFromImage(resized, blob, 1/255.0, input_size, cv::Scalar(), true, false);
            
            // Create input tensor
            std::vector<int64_t> shape = {1, 3, input_size.height, input_size.width};
            Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(info, (float*)blob.data, blob.total(), shape.data(), shape.size());
            
            // Run inference
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_names.data(), &input_tensor, 1,
                output_names.data(), output_names.size()
            );
            
            // Parse output
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            // YOLOv8 output format: [batch, num_detections, features]
            // features: [x_center, y_center, width, height, confidence, class_scores...]
            int num_detections = output_shape[1];
            int num_features = output_shape[2];
            
            for (int i = 0; i < num_detections; ++i) {
                float* detection = output_data + i * num_features;
                
                float x_center = detection[0];
                float y_center = detection[1];
                float width = detection[2];
                float height = detection[3];
                float confidence = detection[4];
                
                // Filter by confidence
                if (confidence < 0.35f) continue;
                
                // Convert to pixel coordinates
                float x1 = (x_center - width/2) * img.cols / input_size.width;
                float y1 = (y_center - height/2) * img.rows / input_size.height;
                float x2 = (x_center + width/2) * img.cols / input_size.width;
                float y2 = (y_center + height/2) * img.rows / input_size.height;
                
                detections.emplace_back(
                    cv::Point(x1, y1),
                    cv::Point(x2, y2)
                );
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[!] Detection error: " << e.what() << std::endl;
        }
        
        return detections;
    }
    
    ~HeadDetector() {
        // Cleanup handled by Ort destructors
    }
};

// ---------------- Team Color Detection --------------------------
class TeamDetector {
    // Simple color-based team detection
    // T-side typically has warmer colors (red/orange)
    // CT-side typically has cooler colors (blue/green)
    
public:
    enum Team {
        UNKNOWN,
        TERRORIST,
        COUNTER_TERRORIST
    };
    
    static Team detect_team(const cv::Mat& frame, const cv::Rect& head_box) {
        try {
            // Extract head region
            cv::Mat head_region = frame(head_box);
            if (head_region.empty()) return UNKNOWN;
            
            // Convert to HSV for better color analysis
            cv::Mat hsv;
            cv::cvtColor(head_region, hsv, cv::COLOR_BGR2HSV);
            
            // Calculate average color
            cv::Scalar avg_color = cv::mean(hsv);
            float hue = avg_color[0]; // 0-180 in OpenCV
            float saturation = avg_color[1];
            float value = avg_color[2];
            
            // T-side colors (red/orange range: 0-20 and 160-180)
            bool is_t_color = (hue < 20 || hue > 160) && saturation > 50;
            
            // CT-side colors (blue range: 100-140)
            bool is_ct_color = (hue > 100 && hue < 140) && saturation > 50;
            
            if (is_t_color) return TERRORIST;
            if (is_ct_color) return COUNTER_TERRORIST;
            
            return UNKNOWN;
            
        } catch (...) {
            return UNKNOWN;
        }
    }
    
    static std::string team_to_string(Team team) {
        switch (team) {
            case TERRORIST: return "T";
            case COUNTER_TERRORIST: return "CT";
            default: return "?";
        }
    }
};

// ---------------- Main Application ------------------------------
class CS2Aimbot {
    UMouse mouse;
    AdvancedToggle toggle;
    ScreenCapture capture;
    HeadDetector detector;
    
    // Performance tracking
    std::chrono::steady_clock::time_point last_frame_time;
    int frame_count = 0;
    float avg_fps = 0.0f;
    
    void update_performance() {
        auto now = std::chrono::steady_clock::now();
        frame_count++;
        
        if (frame_count % 60 == 0) { // Update every 60 frames
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame_time);
            avg_fps = 60000.0f / duration.count();
            last_frame_time = now;
        }
    }
    
public:
    CS2Aimbot() {
        last_frame_time = std::chrono::steady_clock::now();
        std::cout << "\n[+] CS2 Aimbot initialized successfully!" << std::endl;
        std::cout << "[+] Press F10 for status, Ctrl+C to quit\n" << std::endl;
    }
    
    void run() {
        const int crosshair_size = 10;
        
        while (true) {
            try {
                if (!toggle.is_active()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }
                
                // Capture frame
                cv::Mat frame = capture.grab();
                if (frame.empty()) continue;
                
                int center_x = frame.cols / 2;
                int center_y = frame.rows / 2;
                
                // Detect heads
                auto heads = detector.detect(frame);
                
                // Filter by team and find best target
                cv::Rect best_target;
                float best_score = 0.0f;
                bool found_target = false;
                
                for (auto& head : heads) {
                    // Calculate distance from crosshair
                    int dx = (head.x + head.width/2) - center_x;
                    int dy = (head.y + head.height/2) - center_y;
                    float distance = std::sqrt(dx*dx + dy*dy);
                    
                    // Skip if outside FOV
                    if (distance > FOV) continue;
                    
                    // Team detection
                    auto team = TeamDetector::detect_team(frame, head);
                    std::string team_str = TeamDetector::team_to_string(team);
                    
                    // Check if this target matches our preferred team
                    bool target_team = toggle.is_targeting_t();
                    bool is_valid_target = false;
                    
                    if (team == TeamDetector::TERRORIST && target_team) is_valid_target = true;
                    if (team == TeamDetector::COUNTER_TERRORIST && !target_team) is_valid_target = true;
                    if (team == TeamDetector::UNKNOWN) is_valid_target = true; // Target unknown teams
                    
                    if (!is_valid_target) continue;
                    
                    // Score based on distance and head size
                    float score = head.area() / (distance + 1.0f);
                    
                    if (score > best_score) {
                        best_score = score;
                        best_target = head;
                        found_target = true;
                    }
                    
                    // Draw detection
                    if (toggle.is_debug_enabled()) {
                        cv::Scalar color = (team == TeamDetector::TERRORIST) ? 
                            cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
                        cv::rectangle(frame, head, color, 2);
                        
                        // Draw team label
                        cv::putText(frame, team_str, 
                                   cv::Point(head.x, head.y - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                    }
                }
                
                // Aim at best target
                if (found_target) {
                    int dx = (best_target.x + best_target.width/2) - center_x;
                    int dy = (best_target.y + best_target.height/2) - center_y;
                    
                    // Smooth movement
                    int move_x = dx / SMOOTH;
                    int move_y = dy / SMOOTH;
                    
                    mouse.move(move_x, move_y);
                    
                    // Trigger bot
                    cv::Rect crosshair_box(center_x - 5, center_y - 5, 10, 10);
                    cv::Rect target_box(center_x + dx - 5, center_y + dy - 5, 10, 10);
                    
                    double overlap = (crosshair_box & target_box).area() / double(target_box.area());
                    if (overlap > TRIGGER) {
                        mouse.click();
                    }
                }
                
                // Debug overlay
                if (toggle.is_debug_enabled()) {
                    // Draw FOV circle
                    cv::circle(frame, cv::Point(center_x, center_y), FOV, cv::Scalar(0, 255, 0), 1);
                    
                    // Draw crosshair
                    cv::line(frame, 
                            cv::Point(center_x - crosshair_size, center_y),
                            cv::Point(center_x + crosshair_size, center_y),
                            cv::Scalar(0, 255, 0), 1);
                    cv::line(frame,
                            cv::Point(center_x, center_y - crosshair_size),
                            cv::Point(center_x, center_y + crosshair_size),
                            cv::Scalar(0, 255, 0), 1);
                    
                    // Draw status info
                    std::string status = "Target: " + std::string(toggle.is_targeting_t() ? "T" : "CT");
                    cv::putText(frame, status, cv::Point(10, 30),
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                    
                    char fps_text[32];
                    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", avg_fps);
                    cv::putText(frame, fps_text, cv::Point(10, 60),
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                    
                    // Show frame
                    cv::imshow("CS2 Aimbot - Debug", frame);
                    cv::waitKey(1);
                }
                
                update_performance();
                
                // Frame rate limiting
                std::this_thread::sleep_for(std::chrono::milliseconds(1000 / MAX_FPS));
                
            } catch (const std::exception& e) {
                std::cerr << "[!] Frame processing error: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
    
    ~CS2Aimbot() {
        std::cout << "\n[!] Shutting down CS2 Aimbot..." << std::endl;
        cv::destroyAllWindows();
    }
};

// ---------------- Main Function ---------------------------------
int main() {
    try {
        // Print banner
        std::cout << R"(
   ____ ____  _   _ _____      _     
  / ___|  _ \| | | |_   _|   / \    
 | |   | |_) | | | | | |    / _ \   
 | |___|  _ <| |_| | | |   / ___ \  
  \____|_| \\\\___/  |_|  /_/   \_\ 
  Computer Vision Aimbot v2.0 Enhanced
        )" << std::endl;
        
        std::cout << "[!] Starting enhanced CS2 CV aimbot..." << std::endl;
        std::cout << "[!] This tool is for educational purposes only!" << std::endl;
        std::cout << "[!] Use at your own risk - may result in game bans\n" << std::endl;
        
        CS2Aimbot aimbot;
        aimbot.run();
        
    } catch (const std::exception& e) {
        std::cerr << "[!] Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}