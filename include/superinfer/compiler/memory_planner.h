#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <superinfer/base/result.hpp>

namespace superinfer::compiler {

enum class ArenaKind { device, workspace };
enum class AllocationClass { persistent_weight, constant, kv_state, decode_state, activation, scratch };

/** Half-open command interval [first, last) during which an allocation is live. */
struct Lifetime final {
  std::uint64_t first{0};
  std::uint64_t last{0};
};

/** A compiler-owned allocation request; no device address is present at this level. */
struct AllocationRequest final {
  std::uint64_t id{0};
  std::string name;
  ArenaKind arena{ArenaKind::device};
  std::uint64_t bytes{0};
  std::uint64_t alignment{0};
  Lifetime lifetime{};
  AllocationClass allocation_class{AllocationClass::activation};
};

/** An immutable physical placement with the request's lifetime retained for auditing. */
struct Allocation final {
  std::uint64_t id{0};
  std::string name;
  ArenaKind arena{ArenaKind::device};
  AllocationClass allocation_class{AllocationClass::activation};
  std::uint64_t offset{0};
  std::uint64_t bytes{0};
  std::uint64_t alignment{0};
  Lifetime lifetime{};
};

/** Deterministic resource ledger emitted alongside a Physical Plan. */
struct MemoryPlan final {
  std::uint64_t device_arena_bytes{0};
  std::uint64_t workspace_bytes{0};
  std::vector<Allocation> allocations;

  /** Rechecks bounds, alignment, overflow, and liveness-safe non-overlap. */
  [[nodiscard]] base::Status verify() const;

  /** Returns a stable human-readable ledger suitable for golden evidence. */
  [[nodiscard]] std::string dump() const;
};

/**
 * Performs first-fit aligned liveness reuse for device and workspace arenas.
 *
 * The planner owns no buffers and is thread-safe when used concurrently because each call builds a
 * fresh value result. It never touches a device. Requests are copied into the result so callers can
 * retain or destroy their input collection after planning.
 */
class MemoryPlanner final {
 public:
  MemoryPlanner(std::uint64_t device_budget_bytes, std::uint64_t workspace_budget_bytes)
      : device_budget_bytes_(device_budget_bytes), workspace_budget_bytes_(workspace_budget_bytes) {}

  [[nodiscard]] base::Result<MemoryPlan> plan(const std::vector<AllocationRequest>& requests) const;

 private:
  std::uint64_t device_budget_bytes_;
  std::uint64_t workspace_budget_bytes_;
};

}  // namespace superinfer::compiler
