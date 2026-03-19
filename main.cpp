#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

const int MAX_ITERATIONS = 50000;

struct Point {
  int x, y;
};

struct Rect {
  int x1 = 2000000000, y1 = 2000000000, x2 = -2000000000, y2 = -2000000000;
  void add(int x, int y) {
    if (x < x1)
      x1 = x;
    if (x > x2)
      x2 = x;
    if (y < y1)
      y1 = y;
    if (y > y2)
      y2 = y;
  }
  int width() const { return x2 - x1 + 1; }
  int height() const { return y2 - y1 + 1; }
};

struct DynamicGrid {
  int origin_x, origin_y;
  int width, height;
  std::vector<uint8_t> types;
  std::vector<int> last_frame;
  std::vector<int> dirty_indices;

  DynamicGrid(int initial_size = 256) {
    width = initial_size;
    height = initial_size;
    origin_x = -width / 2;
    origin_y = -height / 2;
    types.assign(width * height, 255);
    last_frame.assign(width * height, -1);
    dirty_indices.reserve(initial_size * initial_size);
  }

  void expand() {
    int new_width = width * 2;
    int new_height = height * 2;
    int new_origin_x = origin_x - width / 2;
    int new_origin_y = origin_y - height / 2;

    std::vector<uint8_t> new_types(new_width * new_height, 255);
    std::vector<int> new_last_frame(new_width * new_height, -1);

    for (size_t i = 0; i < dirty_indices.size(); ++i) {
      int idx = dirty_indices[i];
      int old_x = origin_x + (idx % width);
      int old_y = origin_y + (idx / width);
      int new_idx = (old_y - new_origin_y) * new_width + (old_x - new_origin_x);

      new_types[new_idx] = types[idx];
      new_last_frame[new_idx] = last_frame[idx];
      dirty_indices[i] = new_idx;
    }

    width = new_width;
    height = new_height;
    origin_x = new_origin_x;
    origin_y = new_origin_y;
    types = std::move(new_types);
    last_frame = std::move(new_last_frame);
  }

  void ensure_capacity(int min_x, int min_y, int max_x, int max_y) {
    while (min_x - 2 < origin_x || max_x + 2 >= origin_x + width ||
           min_y - 2 < origin_y || max_y + 2 >= origin_y + height) {
      expand();
    }
  }

  inline int get_idx_fast(int x, int y) const {
    return (y - origin_y) * width + (x - origin_x);
  }

  void clear_dirty() {
    for (int idx : dirty_indices) {
      types[idx] = 255;
      last_frame[idx] = -1;
    }
    dirty_indices.clear();
  }
};

std::atomic<bool> keep_searching(true);
std::atomic<bool> in_render_phase(false);
std::atomic<uint64_t> global_seed_counter(0);
std::mutex best_mutex;
uint32_t best_seed = 0;
std::atomic<int> max_frames_atomic(0);

void handle_sigint(int sig) {
  if (in_render_phase) {
    std::cout << "\n\n[Interrupt] Ctrl+C caught during render! Exiting "
                 "immediately.\n";
    std::exit(0);
  } else {
    std::cout << "\n\n[Interrupt] Ctrl+C caught! Stopping search and moving to "
                 "render phase...\n";
    keep_searching = false;
  }
}

const int dx[] = {0, 1, 0, -1};
const int dy[] = {-1, 0, 1, 0};
const uint8_t conn_masks[] = {3, 6, 12, 9};

inline int get_initial_state(int x, int y, uint32_t seed) {
  uint64_t h = seed;
  h ^= (uint64_t)(uint32_t)x * 0x9e3779b97f4a7c15ULL;
  h ^= (uint64_t)(uint32_t)y * 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 30;
  h *= 0x85ebca6bULL;
  h ^= h >> 27;
  h *= 0xc2b2ae35ULL;
  h ^= h >> 16;
  return h & 3;
}

struct NoOpCallback {
  inline void operator()(int, const std::vector<Point> &) const {}
};

template <typename Callback>
int run_simulation(uint32_t seed, DynamicGrid &grid,
                   std::vector<Point> &current_wave,
                   std::vector<Point> &next_wave, Rect *bound, Callback cb) {
  grid.clear_dirty();
  current_wave.clear();

  int start_x = 0;
  int start_y = 0;
  current_wave.push_back({start_x, start_y});
  current_wave.push_back({start_x + 1, start_y});
  current_wave.push_back({start_x, start_y + 1});
  current_wave.push_back({start_x + 1, start_y + 1});

  grid.ensure_capacity(start_x, start_y, start_x + 1, start_y + 1);

  for (auto &p : current_wave) {
    int idx = grid.get_idx_fast(p.x, p.y);
    if (grid.types[idx] == 255) {
      grid.types[idx] = get_initial_state(p.x, p.y, seed);
      grid.dirty_indices.push_back(idx);
    }
    grid.last_frame[idx] = 0;
    if (bound)
      bound->add(p.x, p.y);
  }

  int frame_index = 0;
  while (!current_wave.empty() && frame_index < MAX_ITERATIONS) {
    int min_x = current_wave[0].x, max_x = min_x;
    int min_y = current_wave[0].y, max_y = min_y;
    for (size_t i = 1; i < current_wave.size(); ++i) {
      int x = current_wave[i].x;
      int y = current_wave[i].y;
      if (x < min_x)
        min_x = x;
      if (x > max_x)
        max_x = x;
      if (y < min_y)
        min_y = y;
      if (y > max_y)
        max_y = y;
    }

    grid.ensure_capacity(min_x, min_y, max_x, max_y);

    for (auto &p : current_wave) {
      int idx = grid.get_idx_fast(p.x, p.y);
      grid.types[idx] = (grid.types[idx] + 3) & 3;
    }

    cb(frame_index, current_wave);

    next_wave.clear();
    for (auto &p : current_wave) {
      int idx = grid.get_idx_fast(p.x, p.y);
      int type = grid.types[idx];

      int active_dirs[] = {type, (type + 1) & 3};

      for (int i = 0; i < 2; ++i) {
        int dir = active_dirs[i];
        int nx = p.x + dx[dir], ny = p.y + dy[dir], opp = (dir + 2) & 3;

        int nidx = grid.get_idx_fast(nx, ny);

        if (grid.types[nidx] == 255) {
          grid.types[nidx] = get_initial_state(nx, ny, seed);
          grid.dirty_indices.push_back(nidx);
        }

        if ((conn_masks[grid.types[nidx]] >> opp) & 1) {
          if (grid.last_frame[nidx] != frame_index + 1) {
            grid.last_frame[nidx] = frame_index + 1;
            next_wave.push_back({nx, ny});
            if (bound)
              bound->add(nx, ny);
          }
        }
      }
    }
    current_wave.swap(next_wave);
    frame_index++;
  }
  return frame_index;
}

void save_bmp(const std::string &filename, int W, int H,
              const std::vector<Point> &wave, int min_x, int min_y) {
  static std::vector<uint8_t> full_buffer;
  int row_stride = (W * 3 + 3) & ~3;
  size_t total_size = (size_t)row_stride * H;

  if (full_buffer.size() < total_size)
    full_buffer.resize(total_size);
  std::fill(full_buffer.begin(), full_buffer.begin() + total_size, 0);

  for (auto &p : wave) {
    int lx = p.x - min_x;
    int ly = p.y - min_y;
    int file_row = H - 1 - ly;
    int idx = file_row * row_stride + lx * 3;
    full_buffer[idx] = 255;     // B
    full_buffer[idx + 1] = 255; // G
    full_buffer[idx + 2] = 0;   // R
  }

  std::ofstream f(filename, std::ios::binary);
  uint8_t header[54] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
  *(uint32_t *)(header + 2) = 54 + total_size;
  *(uint32_t *)(header + 10) = 54;
  *(uint32_t *)(header + 14) = 40;
  *(int32_t *)(header + 18) = W;
  *(int32_t *)(header + 22) = H;
  *(uint16_t *)(header + 26) = 1;
  *(uint16_t *)(header + 28) = 24;
  f.write((char *)header, 54);
  f.write((char *)full_buffer.data(), total_size);
}

void search_worker() {
  DynamicGrid grid;
  std::vector<Point> w1, w2;
  w1.reserve(4096);
  w2.reserve(4096);

  while (keep_searching) {
    uint64_t next_seed_64 =
        global_seed_counter.fetch_add(1, std::memory_order_relaxed);
    if (next_seed_64 > 0xFFFFFFFFULL) {
      keep_searching = false;
      break;
    }
    uint32_t seed = (uint32_t)next_seed_64;

    int frames = run_simulation(seed, grid, w1, w2, nullptr, NoOpCallback{});

    int current_max = max_frames_atomic.load(std::memory_order_relaxed);
    if (frames > current_max) {
      std::lock_guard<std::mutex> lock(best_mutex);
      if (frames > max_frames_atomic.load(std::memory_order_relaxed)) {
        max_frames_atomic.store(frames, std::memory_order_relaxed);
        best_seed = seed;
        std::cout << "\r[New Best] Seed: " << best_seed
                  << " | Frames: " << frames
                  << "                                             \n";
      }
    }
  }
}

int main(int argc, char *argv[]) {
  std::signal(SIGINT, handle_sigint);
  std::filesystem::create_directory("bmps");

  if (argc > 1) {
    try {
      best_seed = (uint32_t)std::stoul(argv[1]);
      std::cout << "Target seed provided: " << best_seed
                << ". Skipping search phase.\n";

      DynamicGrid grid;
      std::vector<Point> w1, w2;
      int frames =
          run_simulation(best_seed, grid, w1, w2, nullptr, NoOpCallback{});
      max_frames_atomic.store(frames);

    } catch (...) {
      std::cerr << "Invalid seed parameter.\n";
      return 1;
    }
  } else {
    int num_threads = std::max(1, (int)std::thread::hardware_concurrency());
    std::cout << "Starting optimized search on " << num_threads
              << " threads...\n";

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
      threads.emplace_back(search_worker);

    auto start_time = std::chrono::steady_clock::now();

    while (keep_searching) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      uint64_t current = global_seed_counter.load(std::memory_order_relaxed);
      if (current > 0xFFFFFFFFULL) {
        current = 0xFFFFFFFFULL;
      }

      auto now = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed = now - start_time;
      double elapsed_sec = elapsed.count();

      double seeds_per_sec = 0;
      if (elapsed_sec > 0) {
        seeds_per_sec = current / elapsed_sec;
      }

      uint64_t remaining = 0xFFFFFFFFULL - current;
      double eta_seconds = 0;
      if (seeds_per_sec > 0) {
        eta_seconds = remaining / seeds_per_sec;
      }

      int days = eta_seconds / 86400;
      int hours = ((int)eta_seconds % 86400) / 3600;
      int minutes = ((int)eta_seconds % 3600) / 60;
      int seconds = ((int)eta_seconds % 60);

      std::cout << "\rSearching... Seed: " << current
                << " | Rate: " << (int)seeds_per_sec << " seeds/s"
                << " | ETA: " << days << "d " << std::setw(2)
                << std::setfill('0') << hours << "h " << std::setw(2)
                << std::setfill('0') << minutes << "m " << std::setw(2)
                << std::setfill('0') << seconds << "s     " << std::flush;

      if (current >= 0xFFFFFFFFULL) {
        break;
      }
    }
    for (auto &t : threads)
      t.join();
  }

  int final_max = max_frames_atomic.load();
  if (final_max > 0) {
    in_render_phase = true; // allow Ctrl+C to quit immediately
    std::cout << "\n\nReplaying best seed: " << best_seed << " (" << final_max
              << " frames)\n";

    DynamicGrid grid;
    std::vector<Point> w1, w2;
    Rect bound;

    // First pass to find bounding box
    run_simulation(best_seed, grid, w1, w2, &bound, NoOpCallback{});

    // Make sure dimensions are divisible by 2 for ffmpeg
    int W = (bound.width() + 1) & ~1;
    int H = (bound.height() + 1) & ~1;
    std::cout << "Resolution: " << W << "x" << H << "\n";

    run_simulation(best_seed, grid, w1, w2, nullptr,
                   [&](int frame, const std::vector<Point> &wave) {
                     char fname[64];
                     std::snprintf(fname, sizeof(fname), "bmps/state_%04d.bmp",
                                   frame);
                     save_bmp(fname, W, H, wave, bound.x1, bound.y1);
                     if (frame % 100 == 0)
                       std::cout << "\rRendering... " << frame << "/"
                                 << final_max << std::flush;
                   });

    std::cout << "\nRender complete in 'bmps/' directory.\n";
  } else {
    std::cout << "\nNo runs generated.\n";
  }

  return 0;
}
