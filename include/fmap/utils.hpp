#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace fmap {

// Every expensive loop here is independent, so this covers them all without
// OpenMP. `body` must not throw -- it would cross a thread boundary.
template <typename Body>
void parallel_for(std::size_t count, const Body& body,
                  std::size_t min_per_thread = 256) {
  if (count == 0) return;

  unsigned hardware = std::thread::hardware_concurrency();
  if (hardware == 0) hardware = 1;

  const std::size_t max_useful = std::max<std::size_t>(1, count / min_per_thread);
  const std::size_t threads =
      std::min<std::size_t>(hardware, std::max<std::size_t>(1, max_useful));

  if (threads <= 1) {
    for (std::size_t i = 0; i < count; ++i) body(i);
    return;
  }

  const std::size_t chunk = (count + threads - 1) / threads;
  std::vector<std::thread> workers;
  workers.reserve(threads);

  for (std::size_t t = 0; t < threads; ++t) {
    const std::size_t begin = t * chunk;
    const std::size_t end = std::min(count, begin + chunk);
    if (begin >= end) break;
    workers.emplace_back([&body, begin, end] {
      for (std::size_t i = begin; i < end; ++i) body(i);
    });
  }
  for (std::thread& w : workers) w.join();
}

class Timer {
 public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  void reset() { start_ = std::chrono::steady_clock::now(); }

  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace fmap
