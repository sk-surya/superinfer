#include <superinfer/compiler/memory_planner.h>

#include <algorithm>
#include <limits>
#include <sstream>

#include <superinfer/base/checked_math.hpp>

namespace superinfer::compiler {
namespace {

bool overlaps(const Lifetime left, const Lifetime right) {
  return left.first < right.last && right.first < left.last;
}

bool ranges_overlap(const Allocation& left, std::uint64_t offset, std::uint64_t bytes) {
  const std::uint64_t left_end = left.offset + left.bytes;
  const std::uint64_t right_end = offset + bytes;
  return offset < left_end && left.offset < right_end;
}

std::uint64_t budget_for(ArenaKind arena, std::uint64_t device, std::uint64_t workspace) {
  return arena == ArenaKind::device ? device : workspace;
}

const char* arena_name(ArenaKind arena) {
  return arena == ArenaKind::device ? "device" : "workspace";
}

const char* allocation_class_name(AllocationClass allocation_class) {
  switch (allocation_class) {
    case AllocationClass::persistent_weight: return "persistent_weight";
    case AllocationClass::constant: return "constant";
    case AllocationClass::kv_state: return "kv_state";
    case AllocationClass::decode_state: return "decode_state";
    case AllocationClass::activation: return "activation";
    case AllocationClass::scratch: return "scratch";
  }
  return "unknown";
}

}  // namespace

base::Status MemoryPlan::verify() const {
  for (std::size_t index = 0; index < allocations.size(); ++index) {
    const Allocation& allocation = allocations[index];
    if (allocation.name.empty() || allocation.bytes == 0 || allocation.alignment == 0 ||
        allocation.lifetime.first >= allocation.lifetime.last ||
        allocation.offset % allocation.alignment != 0) {
      return base::Status::invalid_argument("memory allocation has invalid identity, size, alignment, or lifetime");
    }
    if (allocation.offset > std::numeric_limits<std::uint64_t>::max() - allocation.bytes) {
      return base::Status::overflow("memory allocation end overflows");
    }
    const std::uint64_t end = allocation.offset + allocation.bytes;
    if (end > budget_for(allocation.arena, device_arena_bytes, workspace_bytes)) {
      return base::Status::resource_exhausted("memory allocation exceeds arena budget");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      const Allocation& other = allocations[prior];
      if (allocation.id == other.id) {
        return base::Status::invalid_argument("memory allocation IDs are not unique");
      }
      if (allocation.arena == other.arena && overlaps(allocation.lifetime, other.lifetime) &&
          ranges_overlap(other, allocation.offset, allocation.bytes)) {
        return base::Status::failed_precondition("live memory allocations overlap");
      }
    }
  }
  return {};
}

std::string MemoryPlan::dump() const {
  std::ostringstream output;
  output << "memory-plan:v1 peak device=" << device_arena_bytes
         << " workspace=" << workspace_bytes << "\n";
  for (const Allocation& allocation : allocations) {
    output << "allocation id=" << allocation.id << " name=" << allocation.name
           << " class=" << allocation_class_name(allocation.allocation_class)
           << " arena=" << arena_name(allocation.arena) << " offset=" << allocation.offset
           << " bytes=" << allocation.bytes << " alignment=" << allocation.alignment
           << " lifetime=" << allocation.lifetime.first << ".." << allocation.lifetime.last << "\n";
  }
  return output.str();
}

base::Result<MemoryPlan> MemoryPlanner::plan(const std::vector<AllocationRequest>& requests) const {
  MemoryPlan result;
  std::vector<std::size_t> order;
  order.reserve(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const AllocationRequest& request = requests[index];
    if (request.name.empty() || request.bytes == 0 || request.alignment == 0 ||
        request.lifetime.first >= request.lifetime.last) {
      return base::Status::invalid_argument("memory allocation request is incomplete");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (requests[prior].id == request.id) {
        return base::Status::invalid_argument("memory allocation request IDs are not unique");
      }
    }
    order.push_back(index);
  }
  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    if (requests[left].lifetime.first != requests[right].lifetime.first) {
      return requests[left].lifetime.first < requests[right].lifetime.first;
    }
    return requests[left].id < requests[right].id;
  });

  for (const std::size_t index : order) {
    const AllocationRequest& request = requests[index];
    std::uint64_t cursor = 0;
    std::uint64_t chosen = 0;
    while (true) {
      const auto aligned = base::checked_align_up(cursor, request.alignment);
      if (!aligned.has_value()) {
        base::Status error = aligned.error();
        return error.with_context(request.name);
      }
      chosen = aligned.value();
      const auto end = base::checked_add(chosen, request.bytes);
      if (!end.has_value()) {
        base::Status error = end.error();
        return error.with_context(request.name);
      }
      bool conflict = false;
      std::uint64_t next_cursor = chosen;
      for (const Allocation& placed : result.allocations) {
        if (placed.arena != request.arena || !overlaps(placed.lifetime, request.lifetime) ||
            !ranges_overlap(placed, chosen, request.bytes)) {
          continue;
        }
        conflict = true;
        const auto placed_end = base::checked_add(placed.offset, placed.bytes);
        if (!placed_end.has_value()) {
          base::Status error = placed_end.error();
          return error.with_context(placed.name);
        }
        next_cursor = std::max(next_cursor, placed_end.value());
      }
      if (!conflict) {
        if (end.value() > budget_for(request.arena, device_budget_bytes_, workspace_budget_bytes_)) {
          return base::Status::resource_exhausted(std::string("memory budget exceeded for ") + request.name);
        }
        result.allocations.push_back({request.id, request.name, request.arena, request.allocation_class,
                                      chosen, request.bytes, request.alignment, request.lifetime});
        break;
      }
      if (next_cursor <= cursor) return base::Status::internal("memory planner failed to advance");
      cursor = next_cursor;
    }
  }

  std::sort(result.allocations.begin(), result.allocations.end(),
            [](const Allocation& left, const Allocation& right) { return left.id < right.id; });
  for (const Allocation& allocation : result.allocations) {
    const auto end = base::checked_add(allocation.offset, allocation.bytes);
    if (!end.has_value()) {
      base::Status error = end.error();
      return error.with_context(allocation.name);
    }
    std::uint64_t& peak = allocation.arena == ArenaKind::device ? result.device_arena_bytes
                                                                 : result.workspace_bytes;
    peak = std::max(peak, end.value());
  }
  base::Status status = result.verify();
  if (!status.ok()) return status.with_context("memory planner");
  return result;
}

}  // namespace superinfer::compiler
