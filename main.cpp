#include <atomic>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Simulation Configuration
const int GRID_W = 1000;
const int GRID_H = 1000;
const int MAX_ITERATIONS = 50000;

// Directions mapping (0: North, 1: East, 2: South, 3: West)
const int dx[] = {0, 1, 0, -1};
const int dy[] = {-1, 0, 1, 0};

// Atomic flag to handle the Ctrl+C interrupt gracefully
std::atomic<bool> keep_searching(true);

void handle_sigint(int sig) {
  std::cout << "\n\n[Interrupt] Ctrl+C caught! Stopping search and moving to "
               "render phase...\n";
  keep_searching = false;
}

// A simple 24-bit BMP Image class
class BMPImage {
  int width, height;
  std::vector<uint8_t> pixels;

public:
  BMPImage(int w, int h) : width(w), height(h), pixels(w * h * 3, 0) {}

  void set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || x >= width || y < 0 || y >= height)
      return;
    int idx = (y * width + x) * 3;
    pixels[idx] = b;
    pixels[idx + 1] = g;
    pixels[idx + 2] = r;
  }

  void save(const std::string &filename) {
    std::ofstream f(filename, std::ios::binary);
    int row_stride = (width * 3 + 3) & ~3;
    int file_size = 54 + row_stride * height;

    uint8_t file_header[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    file_header[2] = (uint8_t)(file_size);
    file_header[3] = (uint8_t)(file_size >> 8);
    file_header[4] = (uint8_t)(file_size >> 16);
    file_header[5] = (uint8_t)(file_size >> 24);

    uint8_t info_header[40] = {40, 0, 0, 0, 0, 0, 0,  0,
                               0,  0, 0, 0, 1, 0, 24, 0};
    info_header[4] = (uint8_t)(width);
    info_header[5] = (uint8_t)(width >> 8);
    info_header[6] = (uint8_t)(width >> 16);
    info_header[7] = (uint8_t)(width >> 24);
    info_header[8] = (uint8_t)(height);
    info_header[9] = (uint8_t)(height >> 8);
    info_header[10] = (uint8_t)(height >> 16);
    info_header[11] = (uint8_t)(height >> 24);

    f.write((char *)file_header, 14);
    f.write((char *)info_header, 40);

    std::vector<uint8_t> padding(row_stride - width * 3, 0);

    for (int y = height - 1; y >= 0; --y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 3;
        f.write((char *)&pixels[idx], 3);
      }
      if (!padding.empty())
        f.write((char *)padding.data(), padding.size());
    }
  }
};

bool has_connection(int pipe_type, int dir) {
  return (pipe_type == dir) || ((pipe_type + 1) % 4 == dir);
}

// Extracted the simulation logic into a reusable function
int run_simulation(uint32_t seed, bool save_frames) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 3);

  std::vector<std::vector<int>> grid(GRID_H, std::vector<int>(GRID_W));
  for (int y = 0; y < GRID_H; ++y) {
    for (int x = 0; x < GRID_W; ++x) {
      grid[y][x] = dist(rng);
    }
  }

  std::vector<std::pair<int, int>> current_wave;

  int start_x = GRID_W / 2;
  int start_y = GRID_H / 2;
  current_wave.push_back({start_x, start_y});
  current_wave.push_back({start_x + 1, start_y});
  current_wave.push_back({start_x, start_y + 1});
  current_wave.push_back({start_x + 1, start_y + 1});

  int frame_index = 0;

  while (!current_wave.empty() && frame_index < MAX_ITERATIONS) {
    BMPImage *img = nullptr;

    // Only allocate the image if we actually plan to save it
    if (save_frames) {
      img = new BMPImage(GRID_W, GRID_H);
    }

    std::vector<std::vector<bool>> in_next_wave(
        GRID_H, std::vector<bool>(GRID_W, false));

    for (auto &p : current_wave) {
      int x = p.first;
      int y = p.second;
      grid[y][x] = (grid[y][x] + 3) % 4; // Turn left

      if (save_frames) {
        img->set_pixel(x, y, 0, 255, 255);
      }
    }

    if (save_frames) {
      std::ostringstream filename;
      filename << "bmps/state_" << std::setfill('0') << std::setw(4)
               << frame_index << ".bmp";
      img->save(filename.str());
      std::cout << "Saved " << filename.str()
                << " (Pixels drawn: " << current_wave.size() << ")\n";
      delete img;
    }

    std::vector<std::pair<int, int>> next_wave;

    // Evaluate propagation to neighbors based on NEW orientations
    for (auto &p : current_wave) {
      int x = p.first;
      int y = p.second;
      int type = grid[y][x];

      int active_dirs[] = {type, (type + 1) % 4};

      for (int dir : active_dirs) {
        int nx = x + dx[dir];
        int ny = y + dy[dir];

        if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H)
          continue;

        int opposite_dir = (dir + 2) % 4;
        if (has_connection(grid[ny][nx], opposite_dir)) {
          if (!in_next_wave[ny][nx]) {
            next_wave.push_back({nx, ny});
            in_next_wave[ny][nx] = true;
          }
        }
      }
    }

    current_wave = next_wave;
    frame_index++;
  }

  return frame_index;
}

int main() {
  // 1. Register signal handler to catch Ctrl+C
  std::signal(SIGINT, handle_sigint);

  // 2. Ensure output directory exists to avoid crashes
  std::filesystem::create_directory("bmps");

  uint32_t current_seed = 0;
  uint32_t best_seed = 0;
  int max_frames = 0;

  std::cout << "Starting headless brute force search...\n";
  std::cout << "Press Ctrl+C at any time to halt the search and render the "
               "best seed.\n\n";

  // 3. Headless Search Phase
  while (keep_searching) {
    int frames = run_simulation(current_seed, false);

    if (frames > max_frames) {
      max_frames = frames;
      best_seed = current_seed;
      std::cout << "\r[New Best] Seed: " << std::setw(8) << best_seed
                << " | Frames: " << max_frames << "           \n";
    }

    current_seed++;

    // Print progress overlay so the terminal doesn't look frozen
    if (current_seed % 500 == 0) {
      std::cout << "\rSearching... Current Seed: " << current_seed
                << std::flush;
    }
  }

  // 4. Replay and Render Phase
  if (max_frames > 0) {
    std::cout << "\n============================================\n";
    std::cout << "Search halted. Replaying the best scenario.\n";
    std::cout << "Best Seed: " << best_seed << "\n";
    std::cout << "Expected Frames: " << max_frames << "\n";
    std::cout << "============================================\n";

    run_simulation(best_seed, true);

    std::cout << "\nRender complete! Images are in the 'bmps' directory.\n";
  } else {
    std::cout << "\nNo valid runs found before interruption.\n";
  }

  return 0;
}
