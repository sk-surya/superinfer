#pragma once

#include <superinfer/ir/semantic/module.hpp>

namespace superinfer::ir::semantic {

/** Stateless CPU-only verifier entry point for parser, converter, and fuzz callers. */
inline base::Status verify(const Module& module) { return module.verify(); }

}  // namespace superinfer::ir::semantic

