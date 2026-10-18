#pragma once

#include <string>

#include "engine/engine_api.h"
#include "engine/generate.h"

namespace bnk {

// Runs the stdin/stdout line protocol until "quit" or EOF.
int serve_main(EngineApi & eng, const GenOptions & gopt, const std::string & model_name);

}  // namespace bnk
